import torch
import random
import numpy as np
import os, sys
import pickle
import gc
from sklearn.cluster import DBSCAN
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.val_loader import get_validation_dataloader
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
    return [f"{p}_{o}" for p, o in zip(val_batch['path'], val_batch['object_name'])]

def inference_flow(batch_processor, val_loader, dino_loader, flow_model:MeanFlow, save_path, args):
    if os.path.exists(save_path):
        return
    
    pred_pose_dict = {}   # key -> pred_pose [num_samples, pose_dim]
    flow_feature_dict = {}  # key -> {'pts_feat': tensor}

    flow_model.eval()
    for i, val_batch in enumerate(tqdm(val_loader, desc="inference flow")):
        # print(val_batch['affine'][0])
        keys = _sample_keys(val_batch)
        sample = batch_processor(val_batch)
        dino_loader.extract_features(sample['roi_rgb'].to(dino_loader.device))
        sample['dino_feat'] = dino_loader.get_feature()
        with torch.no_grad():
            pred_pose, _ = flow_model.pred_func(data=sample)
            # print(get_rot_matrix(pred_pose[0,:5,:-3] ,args.pose_mode), args.pose_mode)
            # print(pred_pose[0,0,-3:])
            pts_feat = sample['pts_feat'].cpu()
            for j, key in enumerate(keys):
                pred_pose_dict[key] = pred_pose[j].cpu()
                flow_feature_dict[key] = {'pts_feat': pts_feat[j]}
            if i % 4 == 3:
                gc.collect()
    # quit()
    print(f"Flow inference done: {len(pred_pose_dict)} samples.")
    pickle.dump((pred_pose_dict, flow_feature_dict), open(save_path, 'wb'))

def aggregate_pose(args, flow_save_path, save_path):
    if os.path.exists(save_path):
        return
    
    pred_pose_dict, _ = pickle.load(open(flow_save_path, 'rb'))
    aggregated_pose_dict = {}  # key -> [4, 4] aggregated pose

    for i, (key, pred_pose) in enumerate(tqdm(pred_pose_dict.items(), desc="aggregate pose")):
        # pred_pose: [num_samples, pose_dim]
        num_samples = pred_pose.shape[0]
        retain_num = max(1, int(num_samples * args.retain_ratio))
        good_pose = pred_pose[:retain_num, :]  # [retain_num, pose_dim]
        rot_matrix = get_rot_matrix(good_pose[:, :-3], args.pose_mode)  # [retain_num, 3, 3]
        quat_wxyz = matrix_to_quaternion(rot_matrix)  # [retain_num, 4]
        aggregated_quat = average_quaternion_batch(quat_wxyz.unsqueeze(0))[0]  # [4]
        if args.clustering:
            # https://math.stackexchange.com/a/90098
            # 1 - ⟨q1, q2⟩ ^ 2 = (1 - cos theta) / 2
            pairwise_distance = 1 - torch.sum(quat_wxyz.unsqueeze(0) * quat_wxyz.unsqueeze(1), dim=2) ** 2
            dbscan = DBSCAN(eps=args.clustering_eps, min_samples=int(args.clustering_minpts * retain_num)).fit(pairwise_distance.cpu().numpy())
            labels = dbscan.labels_
            if np.any(labels >= 0):
                bins = np.bincount(labels[labels >= 0])
                best_label = np.argmax(bins)
                aggregated_quat = average_quaternion_batch(quat_wxyz[labels == best_label].unsqueeze(0))[0]
        aggregated_trans = torch.mean(good_pose[:, -3:], dim=0)  # [3]
        aggregated_pose = torch.zeros(4, 4)
        aggregated_pose[3, 3] = 1
        aggregated_pose[:3, :3] = quaternion_to_matrix(aggregated_quat.unsqueeze(0))[0]
        aggregated_pose[:3, 3] = aggregated_trans
        aggregated_pose_dict[key] = aggregated_pose
        if i % 100 == 99:
            gc.collect()

    print(f"Aggregation done: {len(aggregated_pose_dict)} samples.")
    pickle.dump(aggregated_pose_dict, open(save_path, 'wb'))

def inference_scale(args, batch_processor, val_loader, flow_save_path, aggregate_path, scale_model, save_path):
    if os.path.exists(save_path):
        return
    
    _, flow_feature_dict = pickle.load(open(flow_save_path, 'rb'))
    aggregated_pose_dict = pickle.load(open(aggregate_path, 'rb'))

    final_pose_dict = {}   # key -> [4, 4] final pose
    final_length_dict = {} # key -> [3] bbox side lengths

    if args.pretrained_scale_model_path is None:
        for i, val_batch in enumerate(tqdm(val_loader, desc="inference scale")):
            keys = _sample_keys(val_batch)
            # Skip samples not seen during flow inference
            valid = [j for j, k in enumerate(keys) if k in aggregated_pose_dict]
            if not valid:
                continue
            sample = batch_processor(val_batch)
            pcl: torch.Tensor = sample['pts']  # [bs, 1024, 3]
            agg_poses = torch.stack([aggregated_pose_dict[keys[j]] for j in valid])  # [n, 4, 4]
            rotation = agg_poses[:, :3, :3].to(pcl.device)  # [n, 3, 3]
            rotation_t = torch.transpose(rotation, 1, 2)
            translation = agg_poses[:, :3, 3].to(pcl.device)  # [n, 3]

            pcl_valid = pcl[valid]  # [n, 1024, 3]
            n_pts = pcl_valid.shape[1]
            pcl_valid = pcl_valid - translation.unsqueeze(1)
            pcl_valid = pcl_valid.reshape(-1, 3, 1)
            rotation_t = torch.repeat_interleave(rotation_t, n_pts, dim=0)
            pcl_valid = torch.bmm(rotation_t, pcl_valid).reshape(-1, n_pts, 3)

            bbox_length, _ = torch.max(torch.abs(pcl_valid), dim=1)
            bbox_length *= 2

            for idx, j in enumerate(valid):
                key = keys[j]
                final_pose_dict[key] = agg_poses[idx]
                final_length_dict[key] = bbox_length[idx].cpu()

            if i % 10 == 9:
                gc.collect()
        pickle.dump((final_pose_dict, final_length_dict), open(save_path, 'wb'))
        return

    scale_model.eval()
    for i, val_batch in enumerate(tqdm(val_loader, desc="inference scale")):
        keys = _sample_keys(val_batch)
        valid = [j for j, k in enumerate(keys) if k in aggregated_pose_dict]
        if not valid:
            continue
        sample = batch_processor(val_batch)
        # Reconstruct flow features for matched samples
        pts_feat = torch.stack([flow_feature_dict[keys[j]]['pts_feat'] for j in valid])
        agg_poses = torch.stack([aggregated_pose_dict[keys[j]] for j in valid])
        sample['pts_feat'] = pts_feat.to(args.device)
        sample['axes'] = agg_poses[:, :3, :3].to(args.device)
        # Select only valid samples from the batch for the model
        for k in ('pts', 'zero_mean_pts', 'pts_center', 'roi_rgb', 'roi_xs', 'roi_ys', 'roi_center_dir'):
            if k in sample:
                sample[k] = sample[k][valid]
        cal_mat, length = scale_model.pred_scale_func(sample)
        for idx, j in enumerate(valid):
            key = keys[j]
            final_pose = agg_poses[idx].clone()
            final_pose[:3, :3] = cal_mat[idx].cpu()
            final_pose_dict[key] = final_pose
            final_length_dict[key] = length[idx].cpu()
        if i % 4 == 3:
            gc.collect()
    
    print(f"Scale inference done: {len(final_pose_dict)} samples.")
    pickle.dump((final_pose_dict, final_length_dict), open(save_path, 'wb'))

def get_detect_match(val_loader, batch_processor, cls_save_path, dm_save_path):
    if os.path.exists(dm_save_path):
        return
    
    assert os.path.exists(cls_save_path)
    final_pose_dict, final_length_dict = pickle.load(open(cls_save_path, 'rb'))
    
    all_dm = []
    
    for i, val_batch in enumerate(tqdm(val_loader, desc="detect match")):
        keys = _sample_keys(val_batch)
        # Only include samples that have predictions
        valid = [j for j, k in enumerate(keys) if k in final_pose_dict]
        if not valid:
            continue

        sample = batch_processor(val_batch)

        pred_pose = torch.stack([final_pose_dict[keys[j]] for j in valid]).numpy()
        pred_length = torch.clamp(
            torch.stack([final_length_dict[keys[j]] for j in valid]), min=1e-3
        ).numpy()

        # Move tensors to CPU before converting to numpy
        gt_pose = sample['affine'].cpu()[valid].numpy()
        gt_length = val_batch['bbox_side_len'].cpu()[valid].numpy()

        batch_size = len(valid)
        valid_class_labels = [val_batch['class_label'][j] for j in valid]
        valid_paths = [val_batch['path'][j] for j in valid]
        valid_intrinsics = val_batch['intrinsics'].cpu()[valid]
        
        dm = DetectMatch(
            gt_affine=gt_pose, gt_size=gt_length, 
            gt_sym_labels=[SymLabel(False, 'none', 'none', 'none')] * batch_size,
            gt_class_labels=valid_class_labels,
            pred_affine=pred_pose, pred_size=pred_length,
            image_path=[path + 'color.png' for path in valid_paths],
            camera_intrinsics=array_to_CameraIntrinsicsBase(valid_intrinsics)
        )
        all_dm.append(dm)
        if i % 10 == 9:
            gc.collect()

    all_dm = DetectMatch.concat(all_dm)
    pickle.dump(all_dm, open(dm_save_path, 'wb'))

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
    # shard_path = "/media/kewei/Training/testROPE/test-{000000..000010}.tar"
    val_loader = get_validation_dataloader(
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

    inference_flow(batch_processor, val_loader, dino_loader, flow_model, flow_save_path, args)

    aggregate_pose(args, flow_save_path, aggregate_save_path)

    inference_scale(args, batch_processor, val_loader, flow_save_path, aggregate_save_path, scale_model, scale_save_path)

    get_detect_match(val_loader, batch_processor, scale_save_path, dm_save_path)

    criterion_save_path = _res_path(args, "criterion.pkl")
    get_criterion(dm_save_path, criterion_save_path)

    metrics_save_path = _res_path(args, "metrics.json")
    print_metrics(dm_save_path, criterion_save_path, metrics_save_path)

if __name__ == "__main__":
    main()