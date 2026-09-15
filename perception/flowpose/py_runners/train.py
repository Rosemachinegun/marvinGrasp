from tqdm import tqdm
import sys, os
import torch
import gc
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.train_loader import get_train_dataloader, get_train_dataloader_raw
from dataset.augmentation import ProcessBatch
from networks.dino.dino import DinoLoader
from args import parse_arguments
from configs import instantiate_model

def train_step(model, data, *args, **kwargs):
    loss = model.forward_with_loss(data, *args, **kwargs)
    loss.backward(create_graph=False)
    return loss

def train_flow(args, train_loader, flow_model, dino_loader, batch_processor, teacher_model=None):

    flow_model.clock.epoch = 0
    # Training loop
    for epoch in range(args.n_epochs):
        torch.cuda.empty_cache()
        progressbar = tqdm(train_loader)
        successful_batches = 0
        flow_model.train()

        for batch_idx, batch in enumerate(progressbar):
            if flow_model.clock.step < args.warmup:
                flow_model.update_learning_rate()
            # Process batch - move to device and compute derived quantities
            processed_batch = batch_processor(batch)

            dino_loader.extract_features(processed_batch['roi_rgb'])  # Extract DINO features and store in DinoLoader.feat
            processed_batch['dino_feat'] = dino_loader.get_feature()   # Add 'dino_feat' to processed_batch

            flow_model.encode_func(processed_batch)  # Adds 'pts_feat' to processed_batch
            
            # Train flow using features WITH gradients for PointNet2
            losses = flow_model.train_flow_one_step(data=processed_batch, compiled_train_step=train_step, teacher=teacher_model)

            # ADD MEMORY CLEANUP
            del processed_batch
            if batch_idx % 4 == 0:  # Clear cache every 4 iterations
                torch.cuda.empty_cache()
                gc.collect()
                
            progressbar.set_description(f"EPOCH_{epoch+1}[{batch_idx}][loss: {losses.item():.4f}][successful: {successful_batches}]")
            flow_model.clock.tick()
            successful_batches += 1

        print(f"Epoch {epoch+1}: Successfully processed {successful_batches} batches")
        # update lr
        flow_model.update_learning_rate()
        flow_model.clock.tock()
        flow_model.save_ckpt()

def train_scale(args, train_loader, scale_model, dino_loader, batch_processor, flow_model):

    flow_model.eval()
    scale_model.clock.epoch = 0
    # Training loop
    for epoch in range(args.n_epochs):
        torch.cuda.empty_cache()
        progressbar = tqdm(train_loader)
        successful_batches = 0
        scale_model.train()

        for batch_idx, batch in enumerate(progressbar):
            if scale_model.clock.step < args.warmup:
                scale_model.update_learning_rate()
            # Process batch - move to device and compute derived quantities
            processed_batch = batch_processor(batch)

            dino_loader.extract_features(processed_batch['roi_rgb'])  # Extract DINO features and store in DinoLoader.feat
            processed_batch['dino_feat'] = dino_loader.get_feature()   # Add 'dino_feat' to processed_batch
            
            with torch.no_grad():
                flow_model.encode_func(data=processed_batch)
            losses = scale_model.train_scale_one_step(data=processed_batch)
                
            progressbar.set_description(f"EPOCH_{epoch+1}[{batch_idx}][loss: {[value.item() for key, value in losses.items()]}][successful: {successful_batches}]")
            scale_model.clock.tick()
            successful_batches += 1

        print(f"Epoch {epoch+1}: Successfully processed {successful_batches} batches")
        # update lr
        scale_model.update_learning_rate()
        scale_model.clock.tock()
        scale_model.save_ckpt()

def main():
    args = parse_arguments()
    device = 'cuda'
    shard_path = args.shard_path

    # Create dataloader
    if getattr(args, 'raw', False):
        train_loader = get_train_dataloader_raw(
            args=args,
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        train_loader = get_train_dataloader(
            args=args,
            shard_path=shard_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    # Create batch processor
    batch_processor = ProcessBatch(
        device=device,
        pose_mode=args.pose_mode if hasattr(args, 'pose_mode') else 'quat_wxyz'
    )

    # Load flow model
    model = instantiate_model(args) # instantiate meanflow or scalenet
    # Load DINO model
    dino_loader = DinoLoader(model_name='dinov2_vits14', device=device)

    flow_model = None

    # flow model
    if args.arch == 'pointnet':
        if args.pretrained_flow_model_path is not None:
            if args.use_pretrain:
                model.load_ckpt(model_dir=args.pretrained_flow_model_path, load_model_only=False)
                flow_model = model
        else:
            flow_model = None
        train_flow(args, train_loader, model, dino_loader, batch_processor, teacher_model=flow_model)

    # scale model
    elif args.arch == 'scalenet':
        # 复现神人操作
        args.arch = 'pointnet'  # temp set arch to pointnet to load flow model
        flow_model = instantiate_model(args)
        flow_model.load_ckpt(model_dir=args.pretrained_flow_model_path, load_model_only=True)

        args.arch = 'scalenet'  # reset arch to scalenet
        if args.pretrained_scale_model_path is not None:
                model.load_ckpt(model_dir=args.pretrained_scale_model_path, load_model_only=False)
        train_scale(args, train_loader, model, dino_loader, batch_processor, flow_model)
    
    else:
        print("Wut are you doing???")

if __name__ == '__main__':
    main()