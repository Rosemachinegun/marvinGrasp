import numpy as np
import torch
import cv2
import random
from utils.transforms.mask import defor_2D, aug_bbox_DZI, get_2d_coord_np, crop_resize_by_warp_affine
from utils.transforms.box import get_bbox
from utils.transforms.rotation import add_noise_to_R, quaternion_to_matrix, get_pose_representation
from utils.transforms.metadata import ImageMetaData
from utils.transforms.pixel import pixel2xyz

def rgb_transform(rgb):
    rgb_ = np.transpose(rgb, (2, 0, 1)) / 255.0
    _mean = (0.485, 0.456, 0.406)
    _std = (0.229, 0.224, 0.225)
    for i in range(3):
        rgb_[i, :, :] = (rgb_[i, :, :] - _mean[i]) / _std[i]
    return rgb_

def depth_to_pcl(depth, K, xymap, valid):
    K = K.reshape(-1)
    cx, cy, fx, fy = K[2], K[5], K[0], K[4]
    depth = depth.reshape(-1).astype(np.float32)[valid]
    x_map = xymap[0].reshape(-1)[valid]
    y_map = xymap[1].reshape(-1)[valid]
    real_x = (x_map - cx) * depth / fx
    real_y = (y_map - cy) * depth / fy
    pcl = np.stack((real_x, real_y, depth), axis=-1)
    return pcl.astype(np.float32)

def sample_points(pcl, n_pts):
    total_pts_num = pcl.shape[0]
    if total_pts_num < n_pts:
        pcl = np.concatenate([
            np.tile(pcl, (n_pts // total_pts_num, 1)),
            pcl[:n_pts % total_pts_num]
        ], axis=0)
        ids = np.concatenate([
            np.tile(np.arange(total_pts_num), n_pts // total_pts_num),
            np.arange(n_pts % total_pts_num)
        ], axis=0)
    else:
        ids = np.random.permutation(total_pts_num)[:n_pts]
        pcl = pcl[ids]
    return ids, pcl
    
# Modular augmentation transforms
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample):
        for transform in self.transforms:
            if sample is None:  # Skip if previous transform failed
                return None
            # per_object mode returns a list
            if isinstance(sample, list):
                results = []
                for s in sample:
                    result = transform(s)
                    if result is not None:
                        results.append(result)
                sample = results if results else None
            else:
                sample = transform(sample)
        return sample

class ParseMetaData:
    """Extract object annotations from meta.json and select object(s)"""
    def __init__(self, per_object=False):
        """
        Args:
            per_object: If True, returns a list of samples (one per valid object).
                       Used for evaluation to process each object exactly once.
                       If False, randomly selects one object (for training).
        """
        self.per_object = per_object
        
    def __call__(self, sample):
        # Parse metadata
        meta_dict = sample['meta']
        
        # Remove 'rotation' field if present
        for obj_k, obj_v in meta_dict.get('objects', {}).items():
            if isinstance(obj_v, dict):
                obj_v.pop('rotation', None)
        
        gts = ImageMetaData(**meta_dict)
        
        # Check for valid objects
        valid_objects = [obj for obj in gts.objects if obj.is_valid]

        if not valid_objects:
            return None
        if self.per_object:
            # Return list of samples, one per object (for evaluation)
            samples = []
            for obj in valid_objects:
                obj_sample = sample.copy()
                obj_sample['gts'] = gts
                obj_sample['selected_object'] = obj
                obj_sample['inst_name'] = obj.meta.oid
                samples.append(obj_sample)
            return samples
        else:
            # Select one object randomly (for training)
            obj = random.choice(valid_objects)
            sample['gts'] = gts
            sample['selected_object'] = obj
            sample['inst_name'] = obj.meta.oid
            
            return sample

class CropAndResize:
    """Crop object region and resize"""
    def __init__(self, img_size=224, dynamic_zoom_params=None):
        self.img_size = img_size
        self.dynamic_zoom_params = dynamic_zoom_params
        
    def __call__(self, sample):
        rgb = sample['rgb']
        depth = sample['depth']
        mask = sample['mask']
        obj = sample['selected_object']
        gts = sample['gts']
        
        # Clip depth values
        depth[depth > 1e3] = 0
        
        # Verify shape consistency
        if not (mask.shape[:2] == depth.shape[:2] == rgb.shape[:2]):
            return None
        
        # Get camera intrinsics
        intrinsics = gts.camera.intrinsics
        img_resize_scale = rgb.shape[0] / intrinsics.height
        
        # Build camera matrix
        mat_K = np.array([
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        mat_K[:2, :] *= img_resize_scale
        
        im_H, im_W = rgb.shape[0], rgb.shape[1]
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)
        
        # Get object mask
        object_mask = np.equal(mask, obj.mask_id)
        if not np.any(object_mask):
            return None
        
        # Get bounding box
        ys, xs = np.argwhere(object_mask).transpose(1, 0)
        rmin, rmax, cmin, cmax = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        rmin, rmax, cmin, cmax = get_bbox([rmin, cmin, rmax, cmax], im_H, im_W)
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        
        # Apply dynamic zoom-in augmentation
        bbox_center, scale = aug_bbox_DZI(
            self.dynamic_zoom_params, bbox_xyxy, im_H, im_W
        )
        
        # Crop and resize
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, self.img_size,
            interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)
        
        roi_rgb_ = crop_resize_by_warp_affine(
            rgb, bbox_center, scale, self.img_size,
            interpolation=cv2.INTER_LINEAR
        )
        
        # RGB normalization
        roi_rgb = rgb_transform(roi_rgb_)
        
        # Create binary mask
        mask_target = mask.copy().astype(np.float32)
        mask_target[mask != obj.mask_id] = 0.0
        mask_target[mask == obj.mask_id] = 1.0
        
        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, self.img_size,
            interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)
        
        roi_depth = crop_resize_by_warp_affine(
            depth, bbox_center, scale, self.img_size,
            interpolation=cv2.INTER_NEAREST
        )
        roi_depth = np.expand_dims(roi_depth, axis=0)
        
        # Validate depth
        depth_valid = roi_depth > 0
        if np.sum(depth_valid) <= 1.0:
            return None
        
        roi_m_d_valid = roi_mask.astype(np.bool_) * depth_valid
        if np.sum(roi_m_d_valid) <= 1.0:
            return None
        
        # Store results
        sample['roi_rgb'] = roi_rgb
        sample['roi_rgb_'] = roi_rgb_
        sample['roi_mask'] = roi_mask
        sample['roi_depth'] = roi_depth
        sample['roi_coord_2d'] = roi_coord_2d
        sample['mat_K'] = mat_K
        sample['bbox_center'] = bbox_center
        sample['im_H'] = im_H
        sample['im_W'] = im_W
        sample['intrinsics'] = intrinsics
        
        return sample

class GeneratePointCloud:
    """Convert depth to point cloud"""
    def __init__(self, n_pts=1024, deform_2d_params=None):
        self.n_pts = n_pts
        self.deform_2d_params = deform_2d_params
        
    def __call__(self, sample):
        roi_depth = sample['roi_depth']
        roi_mask = sample['roi_mask']
        mat_K = sample['mat_K']
        roi_coord_2d = sample['roi_coord_2d']
        
        # Apply 2D deformation to mask
        roi_mask_def = defor_2D(
            roi_mask,
            rand_r=self.deform_2d_params['roi_mask_r'],
            rand_pro=self.deform_2d_params['roi_mask_pro']
        )
        
        # Get valid points
        valid = (np.squeeze(roi_depth, axis=0) > 0) * roi_mask_def > 0
        xs, ys = np.argwhere(valid).transpose(1, 0)
        valid = valid.reshape(-1)
        
        # Convert depth to point cloud
        pcl_in = depth_to_pcl(roi_depth, mat_K, roi_coord_2d, valid)
        
        if len(pcl_in) < 50:
            return None
        
        # Sample points
        ids, pcl_in = sample_points(pcl_in, self.n_pts)
        xs, ys = xs[ids], ys[ids]
        
        sample['pcl_in'] = pcl_in
        sample['roi_xs'] = xs
        sample['roi_ys'] = ys
        
        return sample

class DinoAugmentation:
    def __call__(self, sample):
        # TODO: Implement DINO-specific augmentations if needed
        return sample

class ToTensor:
    """Convert to PyTorch tensors and build final data dict"""
    def __init__(self, args=None):
        self.args = args
        
    def __call__(self, sample):
        obj = sample['selected_object']
        intrinsics = sample['intrinsics']
        bbox_center = sample['bbox_center']
        im_H = sample['im_H']
        im_W = sample['im_W']
        
        # Get rotation and translation
        rotation = quaternion_to_matrix(torch.tensor(obj.quaternion_wxyz))
        translation = torch.tensor(obj.translation, dtype=torch.float32)
        
        # Build affine matrix
        affine = torch.eye(4)
        affine[:3, :3] = rotation
        affine[:3, 3] = translation

        # # sym
        # sym_info = obj.meta.tag.symmetry
        # sym_idx = {'none': 0, 'any': 1, 'half': 2, 'quarter': 3}
        # sym_info = [int(sym_info.any), sym_idx[sym_info.x], sym_idx[sym_info.y], sym_idx[sym_info.z]]
        
        data_dict = {
            'pcl_in': torch.as_tensor(sample['pcl_in'], dtype=torch.float32).contiguous(),
            'rotation': rotation.to(torch.float32).contiguous(),
            'translation': translation.contiguous(),
            'affine': affine.to(torch.float32).contiguous(),
            'handle_visibility': torch.as_tensor(1, dtype=torch.int8).contiguous(),
            # 'sym_info': torch.as_tensor(sym_info, dtype=torch.int8).contiguous(),
            'roi_rgb': torch.as_tensor(sample['roi_rgb'], dtype=torch.float32).contiguous(),
            'roi_rgb_': torch.as_tensor(sample['roi_rgb_'], dtype=torch.uint8).contiguous(),
            'roi_xs': torch.as_tensor(sample['roi_xs'], dtype=torch.int64).contiguous(),
            'roi_ys': torch.as_tensor(sample['roi_ys'], dtype=torch.int64).contiguous(),
            'roi_center_dir': torch.as_tensor(
                pixel2xyz(im_H, im_W, bbox_center, intrinsics),
                dtype=torch.float32
            ).contiguous(),
            'intrinsics': torch.as_tensor([
                intrinsics.fx, intrinsics.fy, intrinsics.cx,
                intrinsics.cy, intrinsics.width, intrinsics.height
            ], dtype=torch.float32).contiguous(),
            'bbox_side_len': torch.as_tensor(
                obj.meta.bbox_side_len, dtype=torch.float32
            ).contiguous(),
            'pose': torch.as_tensor(
                obj.quaternion_wxyz + obj.translation, dtype=torch.float32
            ).contiguous(),
            'path': sample.get('__key__', ''),
            'class_label': obj.meta.class_label,
            'class_name': obj.meta.class_name,
            'object_name': sample['inst_name']
        }
        
        # Add scale-specific data if needed
        if self.args and hasattr(self.args, 'agent_type') and self.args.agent_type == 'scale':
            length_training = torch.as_tensor(obj.meta.bbox_side_len, dtype=torch.float32)
            axes4x4_training = torch.zeros(self.args.scale_batch_size, 4, 4)
            axes4x4_training[:, :3, :3] = rotation.unsqueeze(0).repeat_interleave(
                self.args.scale_batch_size, dim=0
            )
            length_training = length_training.unsqueeze(0).repeat_interleave(
                self.args.scale_batch_size, dim=0
            )
            axes4x4_training[:, 3, 3] = 1
            axes_training = add_noise_to_R(axes4x4_training, r=10)[:, :3, :3]
            data_dict['axes_training'] = axes_training.contiguous()
            data_dict['length_training'] = length_training.contiguous()
        
        return data_dict

class ProcessBatch:
    """Process batched samples for training"""
    def __init__(self, device, pose_mode='quat_wxyz'):
        """
        Args:
            device: torch device (cuda or cpu)
            pose_mode: one of ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix']
        """
        assert pose_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix']
        self.device = device
        self.pose_mode = pose_mode
    
    def __call__(self, batch_sample):
        """
        Process a batch of samples
        
        Args:
            batch_sample: Dictionary with batched tensors from DataLoader
            
        Returns:
            processed_sample: Dictionary with processed data ready for model
        """
        # Move all tensors to device
        batch_sample = {
            k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch_sample.items()
        }
        
        PC_da = batch_sample['pcl_in']
        gt_R_da = batch_sample['rotation']
        gt_t_da = batch_sample['translation']

        # Create batched affine matrices
        batch_size = gt_R_da.shape[0]
        affine = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        affine[:, :3, :3] = gt_R_da
        affine[:, :3, 3] = gt_t_da
        
        # Build processed sample
        processed_sample = {
            'class_label': batch_sample['class_label'],
            'affine': affine,
            'pts': PC_da,
            'roi_rgb': batch_sample['roi_rgb'],
            'roi_xs': batch_sample['roi_xs'],
            'roi_ys': batch_sample['roi_ys'],
            'roi_center_dir': batch_sample['roi_center_dir'],
        }
        
        # Scale-specific data
        if 'axes_training' in batch_sample:
            processed_sample['axes_training'] = batch_sample['axes_training']
            processed_sample['length_training'] = batch_sample['length_training']
        
        # Convert rotation to desired pose representation
        rot = get_pose_representation(gt_R_da, self.pose_mode)
        location = gt_t_da
        processed_sample['gt_pose'] = torch.cat([rot.float(), location.float()], dim=-1)
        
        # Compute zero-mean point cloud and pose
        pts = processed_sample['pts']
        zero_mean = pts[:, :, :3].mean(dim=1)  # [B, 3]
        processed_sample['pts_center'] = zero_mean
        
        # Zero-centered point cloud
        processed_sample['zero_mean_pts'] = pts.clone()
        processed_sample['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1)
        
        # Zero-centered pose (adjust translation)
        processed_sample['zero_mean_gt_pose'] = processed_sample['gt_pose'].clone()
        processed_sample['zero_mean_gt_pose'][:, -3:] -= zero_mean
        
        return processed_sample
