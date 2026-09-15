import torch
import os
import sys
from args_kp import parse_args
from dataset.dataset import rawDataset
from torch.utils.data import DataLoader
from utils.trainer import KeypointTrainer
from configs import instantiate_model
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.dino.dino import DinoLoader

def main():
    args = parse_args()
    args.is_train = True
    args.arch = 'keypoint'
    # Normalize data path: args.py stores it as data_path, args_kp.py as data_dir
    if not hasattr(args, 'data_dir') or args.data_dir == '?':
        args.data_dir = getattr(args, 'data_path', None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino = DinoLoader(model_name=args.dino_model, device=device)
    args.dino = dino
    torch.manual_seed(args.seed)

    # Get dataset size and create train/val splits
    total_size = len(rawDataset(
        path=args.data_dir,
        keypoint=True,
        img_size=args.img_size,
        augment=False,
        num_classes=args.num_classes,
        num_keypoints_per_class=args.num_keypoints
    ))

    train_size = int(total_size * args.train_val_split)
    val_size = total_size - train_size

    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_set = rawDataset(
        path=args.data_dir,
        keypoint=True,
        img_size=args.img_size,
        augment=True,
        num_classes=args.num_classes,
        num_keypoints_per_class=args.num_keypoints
    )

    val_set = rawDataset(
        path=args.data_dir,
        keypoint=True,
        img_size=args.img_size,
        augment=False,
        num_classes=args.num_classes,
        num_keypoints_per_class=args.num_keypoints
    )

    # 使用Subset来指定训练集和验证集的索引
    from torch.utils.data import Subset
    train_set = Subset(train_set, train_indices)
    val_set = Subset(val_set, val_indices)

    print(f"总样本数: {total_size}")
    print(f"训练集: {train_size} 样本 ({args.train_val_split*100:.0f}%)")
    print(f"验证集: {val_size} 样本 ({(1-args.train_val_split)*100:.0f}%)")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Init model
    model = instantiate_model(args)
    
    model.to(device)

    # Trainer
    trainer = KeypointTrainer(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=model.optimizer,
        scheduler=model.scheduler,
        loss_fn=model.loss_fn,
        device=device,
        save_dir=args.save_dir,
        max_grad_norm=args.max_grad_norm,
        visualize_validation=not args.no_val_vis,  # 默认启用，除非指定--no-val-vis
        num_val_vis_samples=args.val_vis_samples,
        val_every_n_epochs=args.val_every_n_epochs
    )

    # Continue training from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        start_epoch, trainer.global_step, trainer.best_val_loss = model.load_ckpt(args.resume)
        start_epoch += 1  # resume from next epoch
        print(f"✅ 从epoch {start_epoch - 1}恢复，继续训练")

    # Start training
    print("\n开始训练...")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"初始学习率: {args.lr}")
    print(f"Backbone学习率: {args.lr_backbone}")

    trainer.train(args.epochs - start_epoch)


if __name__ == "__main__":
    main()

