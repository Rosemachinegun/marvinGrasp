import torch
import random
import numpy as np
import os, sys
import pickle
import gc
from sklearn.cluster import DBSCAN
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.val_loader import get_validation_dataloader_raw, get_validation_dataloader_raw_iterable
from args import parse_arguments
from dataset.dataset import OmniXValDataset, array_to_CameraIntrinsicsBase, array_to_SymLabel
from networks.flow.meanflow_inference import MeanFlow
from dataset.augmentation import ProcessBatch
from utils.transforms.rotation import get_rot_matrix, matrix_to_quaternion, quaternion_to_matrix
from utils.misc import average_quaternion_batch
from configs import instantiate_model

from cutoop.eval_utils import DetectMatch, Metrics
from cutoop.rotation import SymLabel
from networks.dino.dino import DinoLoader

def _model_tag(path: str | None, default: str) -> str:
    if not path:
        return default
    return "_".join(path.split("/")[-2:])

def _res_path(args, name: str) -> str:
    return f"results/evaluation_results/{args.result_dir}/{name}"

def _sample_keys(val_batch):
    """Generate unique per-object keys from a batch using WebDataset sample keys."""
    return [f"{o}" for p, o in zip(val_batch['path'], val_batch['object_name'])]

def unified_inference(batch_processor, val_loader, dino_loader, flow_model, scale_model, args, 
                     flow_save_path, aggregate_save_path, scale_save_path, dm_save_path):
    """Unified inference pipeline: flow -> aggregate -> scale -> detect_match in single pass."""
    
    # Check if results already exist
    if all(os.path.exists(p) for p in [flow_save_path, aggregate_save_path, scale_save_path, dm_save_path]):
        print("All results already exist, skipping inference.")
        return
    
    pred_pose_dict = {}
    flow_feature_dict = {}
    final_pose_dict = {}
    final_length_dict = {}
    all_dm = []
    
    flow_model.eval()
    if args.pretrained_scale_model_path is not None:
        scale_model.eval()
    
    print("Running unified inference pipeline...")
    batch_count = 0
    for i, val_batch in enumerate(tqdm(val_loader, desc="unified inference")):
        batch_count += 1
        keys = _sample_keys(val_batch)
        sample = batch_processor(val_batch)
        
        # ============ FLOW INFERENCE ============
        dino_loader.extract_features(sample['roi_rgb'].to(dino_loader.device))
        sample['dino_feat'] = dino_loader.get_feature()
        
        with torch.no_grad():
            pred_pose, _ = flow_model.pred_func(data=sample)
            pts_feat = sample['pts_feat'].cpu()
            
            for j, key in enumerate(keys):
                pred_pose_dict[key] = pred_pose[j].cpu()
                flow_feature_dict[key] = {'pts_feat': pts_feat[j]}
        
        # ============ POSE AGGREGATION ============
        aggregated_poses_batch = {}
        for j, key in enumerate(keys):
            # pred_pose[j] is a single pose vector, aggregate with itself (no actual aggregation needed)
            pred_pose_j = pred_pose[j].cpu()  # [1, pose_dim]
            
            rot_matrix = get_rot_matrix(pred_pose_j[:, :-3], args.pose_mode)  # [1, 3, 3]
            quat_wxyz = matrix_to_quaternion(rot_matrix)  # [1, 4]
            aggregated_quat = average_quaternion_batch(quat_wxyz.unsqueeze(0))[0]  # [4]
            aggregated_trans = torch.mean(pred_pose_j[:, -3:], axis=0)  # [3]
            
            num_samples = pred_pose.shape[0]
            retain_num = max(1, int(num_samples * args.retain_ratio))
            
            if True:
                # https://math.stackexchange.com/a/90098
                # 1 - ⟨q1, q2⟩ ^ 2 = (1 - cos theta) / 2
                pairwise_distance = 1 - torch.sum(quat_wxyz.unsqueeze(0) * quat_wxyz.unsqueeze(1), dim=2) ** 2
                dbscan = DBSCAN(eps=args.clustering_eps, min_samples=int(args.clustering_minpts * retain_num)).fit(pairwise_distance.cpu().numpy())
                labels = dbscan.labels_
                if np.any(labels >= 0):
                    bins = np.bincount(labels[labels >= 0])
                    best_label = np.argmax(bins)
                    aggregated_quat = average_quaternion_batch(quat_wxyz[labels == best_label].unsqueeze(0))[0]
            
            aggregated_pose = torch.zeros(4, 4)
            aggregated_pose[3, 3] = 1
            aggregated_pose[:3, :3] = quaternion_to_matrix(aggregated_quat.unsqueeze(0))[0]
            aggregated_pose[:3, 3] = aggregated_trans
            aggregated_poses_batch[key] = aggregated_pose
        
        # ============ SCALE INFERENCE ============
        for j, key in enumerate(keys):
            if key not in aggregated_poses_batch:
                continue
            
            agg_pose = aggregated_poses_batch[key]
            
            if args.pretrained_scale_model_path is None:
                # Compute bbox length from point cloud
                pcl_j = sample['pts'][j:j+1]  # [1, 1024, 3]
                rotation = agg_pose[:3, :3].to(pcl_j.device)
                translation = agg_pose[:3, 3].to(pcl_j.device)
                
                pcl_transformed = pcl_j - translation.unsqueeze(0)
                pcl_transformed = torch.matmul(rotation.unsqueeze(0), pcl_transformed.unsqueeze(-1)).squeeze(-1)
                bbox_length, _ = torch.max(torch.abs(pcl_transformed), dim=1)
                bbox_length = bbox_length[0] * 2
            else:
                # Use scale model
                sample_j = {k: v[j:j+1] if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                sample_j['pts_feat'] = flow_feature_dict[key]['pts_feat'].unsqueeze(0).to(args.device)
                sample_j['axes'] = agg_pose[:3, :3].to(args.device).unsqueeze(0)
                
                cal_mat, length = scale_model.pred_scale_func(sample_j)
                agg_pose = agg_pose.clone()
                agg_pose[:3, :3] = cal_mat[0].cpu()
                bbox_length = length[0].cpu()
            
            final_pose_dict[key] = agg_pose
            final_length_dict[key] = bbox_length
        
        # ============ DETECT MATCH ============
        valid_indices = [j for j, k in enumerate(keys) if k in final_pose_dict]
        if valid_indices:
            pred_pose_batch = torch.stack([final_pose_dict[keys[j]] for j in valid_indices]).numpy()
            pred_length_batch = torch.clamp(
                torch.stack([final_length_dict[keys[j]] for j in valid_indices]), min=1e-3
            ).numpy()
            
            gt_pose_batch = sample['affine'].cpu()[valid_indices].numpy()
            gt_length_batch = val_batch['bbox_side_len'].cpu()[valid_indices].numpy()
            
            batch_size = len(valid_indices)
            valid_class_labels = [val_batch['class_label'][j] for j in valid_indices]
            valid_paths = [val_batch['path'][j] for j in valid_indices]
            valid_intrinsics = val_batch['intrinsics'].cpu()[valid_indices]
            
            dm = DetectMatch(
                gt_affine=gt_pose_batch, gt_size=gt_length_batch,
                gt_sym_labels=[SymLabel(False, 'none', 'none', 'none')] * batch_size,
                gt_class_labels=valid_class_labels,
                pred_affine=pred_pose_batch, pred_size=pred_length_batch,
                image_path=[path + 'color.png' for path in valid_paths],
                camera_intrinsics=array_to_CameraIntrinsicsBase(valid_intrinsics)
            )
            all_dm.append(dm)
        
        if i % 4 == 3:
            gc.collect()
    
    print(f"\nProcessed {batch_count} batches")
    
    # Save results
    print(f"Flow inference done: {len(pred_pose_dict)} samples.")
    pickle.dump((pred_pose_dict, flow_feature_dict), open(flow_save_path, 'wb'))
    
    print(f"Aggregation done: {len(aggregated_poses_batch)} samples.")
    pickle.dump(aggregated_poses_batch, open(aggregate_save_path, 'wb'))
    
    print(f"Scale inference done: {len(final_pose_dict)} samples.")
    pickle.dump((final_pose_dict, final_length_dict), open(scale_save_path, 'wb'))
    
    all_dm = DetectMatch.concat(all_dm) if all_dm else DetectMatch.concat([])
    pickle.dump(all_dm, open(dm_save_path, 'wb'))
    print(f"DetectMatch done: {len(all_dm)} samples.")


def get_criterion(dm_path, criterion_save_path):
    if os.path.exists(criterion_save_path):
        return
    assert os.path.exists(dm_path)
    all_dm: DetectMatch = pickle.load(open(dm_path, 'rb'))
    criterion = all_dm.criterion()
    pickle.dump(criterion, open(criterion_save_path, 'wb'))

def print_metrics(dm_path, criterion_path, metrics_save_path):
    assert os.path.exists(dm_path)
    all_dm: DetectMatch = pickle.load(open(dm_path, 'rb'))
    assert os.path.exists(criterion_path)
    criterion = pickle.load(open(criterion_path, 'rb'))
    
    metrics: Metrics = all_dm.metrics(
        criterion=criterion,
        iou_auc_ranges=[
            (0.25, 1, 0.075),
            (0.5, 1, 0.005),
            (0.75, 1, 0.0025),
        ],
        pose_auc_ranges=[
            ((0, 5, 0.05), (0, 2, 0.02)),
            ((0, 5, 0.05), (0, 5, 0.05)),
            ((0, 10, 0.1), (0, 2, 0.02)),
            ((0, 10, 0.1), (0, 5, 0.05)),
        ],
    )
    print("iou_mean:", metrics.class_means.iou_mean)
    print("iou_acc (0.25, 0.50, 0.75):", metrics.class_means.iou_acc)
    print("deg_mean:", metrics.class_means.deg_mean)
    print("sht_mean:", metrics.class_means.sht_mean)
    print("pose_acc [(5, 2), (5, 5), (10, 2), (10, 5)]:", metrics.class_means.pose_acc)
    print("AUC @ IoU 25:", metrics.class_means.iou_auc[0].auc)
    print("AUC @ IoU 50:", metrics.class_means.iou_auc[1].auc)
    print("AUC @ IoU 75:", metrics.class_means.iou_auc[2].auc)
    print("VUS @ 5 deg 2 cm:", metrics.class_means.pose_auc[0].auc)
    print("VUS @ 5 deg 5 cm:", metrics.class_means.pose_auc[1].auc)
    print("VUS @ 10 deg 2 cm:", metrics.class_means.pose_auc[2].auc)
    print("VUS @ 10 deg 5 cm:", metrics.class_means.pose_auc[3].auc)
    
    # Convert metrics to dict and handle numpy types
    import json
    from dataclasses import asdict
    
    def convert_to_serializable(obj):
        """Recursively convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_serializable(obj.tolist())
        else:
            return obj
    
    metrics_dict = asdict(metrics)
    metrics_dict = convert_to_serializable(metrics_dict)
    
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

def main():
    args = parse_arguments()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda'

    val_loader = get_validation_dataloader_raw(
        args=args, 
        shard_path=args.shard_path, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    batch_processor = ProcessBatch(
        device=device,
        pose_mode=args.pose_mode if hasattr(args, 'pose_mode') else 'quat_wxyz'
    )

    # Load Dino model (for inference only, no training, so we can load it here without worrying about GPU memory for training)
    dino_loader = DinoLoader(model_name='dinov2_vits14', device=device)

    # Load flow model
    args.arch = 'pointnet'
    flow_model = instantiate_model(args)
    flow_model.load_ckpt(model_dir=args.pretrained_flow_model_path, load_model_only=True)
    flow_model.to(device)
    
    # Load scale model
    args.arch = 'scalenet'
    scale_model = instantiate_model(args)
    if args.pretrained_scale_model_path is not None:
        scale_model.load_ckpt(model_dir=args.pretrained_scale_model_path, load_model_only=True)
    scale_model.to(device)

    os.makedirs(f'results/evaluation_results/{args.result_dir}', exist_ok=True)
    os.makedirs(_res_path(args, ""), exist_ok=True)

    flow_save_path = _res_path(args, f"flow_prediction_{_model_tag(args.pretrained_flow_model_path, 'flow')}.pkl")
    aggregate_save_path = _res_path(args, f"aggregation.pkl")
    scale_save_path = _res_path(args, f"scale_prediction_{_model_tag(args.pretrained_scale_model_path, 'scale')}.pkl")
    dm_save_path = _res_path(args, "detect_match.pkl")

    # Run unified inference pipeline (single pass through dataloader)
    unified_inference(batch_processor, val_loader, dino_loader, flow_model, scale_model, args,
                      flow_save_path, aggregate_save_path, scale_save_path, dm_save_path)

    criterion_save_path = _res_path(args, "criterion.pkl")
    get_criterion(dm_save_path, criterion_save_path)

    metrics_save_path = _res_path(args, "metrics.json")
    print_metrics(dm_save_path, criterion_save_path, metrics_save_path)

if __name__ == "__main__":
    main()