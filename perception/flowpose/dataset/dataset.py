import torch.utils.data as data
import webdataset as wds
import pickle
import numpy as np
import json
import os
import cv2
import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from dataset.augmentation import rgb_transform, depth_to_pcl, sample_points
from utils.transforms.rotation import SymLabel
from utils.transforms.mask import get_2d_coord_np, crop_resize_by_warp_affine, get_affine_transform
from utils.transforms.box import get_bbox, aug_bbox_eval
from utils.transforms.pixel import pixel2xyz
from utils.transforms.metadata import ImageMetaData, ViewInfo, CameraIntrinsicsBase

def load_color(path_or_img: "str | os.PathLike | np.ndarray") -> np.ndarray:
        if path_or_img is None:
            return None
        
        if isinstance(path_or_img, np.ndarray):
            data = path_or_img
            return data
        else:
            data = cv2.imread(str(path_or_img))[:, :, ::-1]  # RGB order
            return data

def load_depth(path_or_img: "str | os.PathLike | np.ndarray") -> np.ndarray:
        if path_or_img is None:
            return None
        if isinstance(path_or_img, np.ndarray):
            data = path_or_img
            return data
        else:
            data = cv2.imread(str(path_or_img), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if len(data.shape) == 3:
                data = data[:, :, 0]
            
            return data  # unit: m
        
def load_mask(path_or_img: "str | os.PathLike | np.ndarray") -> np.ndarray:
        if path_or_img is None:
            return None
        
        if isinstance(path_or_img, np.ndarray):
            data = path_or_img
            return data
        else:
            data = cv2.imread(str(path_or_img), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if len(data.shape) == 3:
                data = data[:, :, 2]
            return np.array(data * 255, dtype=np.uint8)

def array_to_CameraIntrinsicsBase(intrinsics_list):
    return [CameraIntrinsicsBase(*item) for item in intrinsics_list]

def array_to_SymLabel(arr_Nx4: np.ndarray):
    syms_N = []
    tags = ['none', 'any', 'half', 'quarter']
    for a, x, y, z in arr_Nx4:
        syms_N.append(SymLabel(bool(a), tags[x], tags[y], tags[z]))
    return syms_N

class OmniXTrainDataset(data.IterableDataset):
    def __init__(self, shard_path, transform=None, shuffle=1000):
        """
        Args:
            shard_path: Path pattern to shards
            transform: Optional transforms
            shuffle: Shuffle buffer size (0 for no shuffle)
        """
        self.shard_path = shard_path
        self.transform = transform
        self.shuffle = shuffle
        
    def __iter__(self):
        dataset = (
            wds.WebDataset(self.shard_path)
            .shuffle(self.shuffle)
            .decode(self._pose_data_decoder)
        )
        
        for sample in dataset:
            if self.transform:
                sample = self.transform(sample)
            yield sample
    
    @staticmethod
    def _pose_data_decoder(sample):
        """Custom decoder to reconstruct numpy arrays from bytes"""
        try:
            rgb_bytes = sample["rgb.pyd"]
            rgb_shape = pickle.loads(sample["rgb_shape.pyd"])
            rgb_dtype = np.dtype(sample["rgb_dtype.txt"].decode('utf-8'))
            rgb = np.frombuffer(rgb_bytes, dtype=rgb_dtype).reshape(rgb_shape).copy()
            
            depth_bytes = sample["depth.pyd"]
            depth_shape = pickle.loads(sample["depth_shape.pyd"])
            depth_dtype = np.dtype(sample["depth_dtype.txt"].decode('utf-8'))
            depth = np.frombuffer(depth_bytes, dtype=depth_dtype).reshape(depth_shape).copy()
            
            mask_bytes = sample["mask.pyd"]
            mask_shape = pickle.loads(sample["mask_shape.pyd"])
            mask_dtype = np.dtype(sample["mask_dtype.txt"].decode('utf-8'))
            mask = np.frombuffer(mask_bytes, dtype=mask_dtype).reshape(mask_shape).copy()
            
            meta = json.loads(sample["meta.json"].decode('utf-8'))
            
            return {
                "rgb": rgb,
                "depth": depth,
                "mask": mask,
                "meta": meta
            }
        except Exception as e:
            print(f"Decoder error: {e}")
            print(f"Sample keys: {list(sample.keys())}")
            raise

class OmniXInferDataset(object):
    def __init__(self, data: dict, img_size: int=224, device='cuda', n_pts=1024):
        """
        Args:
            data (dict): dictionary containing depth, color, mask, and meta data
                depth (np.ndarray): depth image
                color (np.ndarray): color image
                mask (np.ndarray): mask image
                meta (dict): camera intrinsics
            img_size (int): size of the image to be used for the network
            device (str): device to be used for the network
            n_pts (int): number of points to be used for the network
        """
        self._depth: np.ndarray = data['depth']
        self._color: np.ndarray = data['color']
        self._mask: np.ndarray = data['mask']

        self.intrinsics= {
            "fx": 606.5540161132812,
            "fy": 606.3988647460938,
            "cx": 325.6007080078125,
            "cy": 252.87457275390625,
            "width": 640,
            "height": 480
            }
        
        camera_intrinsics = CameraIntrinsicsBase(
            fx=self.intrinsics['fx'],
            fy=self.intrinsics['fy'],
            cx=self.intrinsics['cx'],
            cy=self.intrinsics['cy'],
            width=self.intrinsics['width'],
            height=self.intrinsics['height']
        )

        camera = ViewInfo(None, None, camera_intrinsics, None, None, None, None, None)
        self._meta: ImageMetaData = ImageMetaData(None, camera, None, None, None, None, None, None, None, None)
        self._img_size = img_size
        self._device = device
        self._n_pts = n_pts

    @classmethod
    def alternetive_init(cls, data, img_size: int=224, device='cuda', n_pts=1024, intrinsics=None, mode = 'infer'):
        """
        Requires depth in meters and mask in uint8 with 0 as background and non-zero as object
        
        """
        meta = None
        if isinstance(data, dict):
            depth = load_depth(data.get("depth")).astype(np.float32) / 1000.0
            color = load_color(data.get("color"))
            mask = load_mask(data.get("mask"))
            if mask is not None:
                mask = mask.copy()
                mask[mask == 255] = 0
        else:
            prefix = data if data.endswith(os.sep) else data + os.sep
            prefix = data.rstrip("/\\")
            depth_file = prefix + "depth.exr"
            color_file = prefix + "color.png"
            mask_file  = prefix + "mask.exr"
            meta_file = prefix + "meta.json"
            # Verify depth file exists before loading
            if depth_file is not None and os.path.exists(depth_file):
                depth = load_depth(depth_file).astype(np.float32) / 1000.0
            else:
                print(f"Warning: Depth file not found at {depth_file}")
                depth = None
            color = load_color(color_file)
            mask = load_mask(mask_file)
            
            # Try to load meta.json if it exists
            if os.path.exists(meta_file):
                try:
                    meta = json.loads(open(meta_file, 'r').read())
                except Exception as e:
                    print(f"Warning: Failed to load meta.json: {e}")
                    meta = None
            
            if mask is not None:
                mask = mask.copy()
                mask[mask == 255] = 0
            
        if depth is None:
            print("Warning: depth is None")
        if color is None:
            print("Warning: color is None")
        if mask is None:
            print("Warning: mask is None")
        # print(f'mask shape: {mask.shape}, unique values: {np.unique(mask)[:10]}')
        
        # Create dataset instance
        dataset = cls({'depth': depth, 'color': color, 'mask': mask}, 
                  img_size=img_size, device=device, n_pts=n_pts)
        
        # Use intrinsics from meta.json if available, otherwise use provided intrinsics
        if meta is not None and mode != 'rs':  # only use meta intrinsics in infer mode, for eval we want to use the same intrinsics for all frames
            try:
                cam_meta = meta['camera']['intrinsics']
                ## SOPE
                if mode == "sope":
                    meta_intrinsics = {
                        'fx': cam_meta['fx']/3,
                        'fy': cam_meta['fy']/3,
                        'cx': cam_meta['cx']/3,
                        'cy': cam_meta['cy']/3,
                        'width': cam_meta['width']/3,
                        'height': cam_meta['height']/3
                    }
                ## ROPE
                if mode == "rope":
                    meta_intrinsics = {
                        'fx': cam_meta['fx']/2,
                        'fy': cam_meta['fy']/2,
                        'cx': cam_meta['cx']/2,
                        'cy': cam_meta['cy']/2,
                        'width': cam_meta['width']/2,
                        'height': cam_meta['height']/2
                    }
                ## default to using the intrinsics as is for infer mode
                if mode == "isaac":
                    meta_intrinsics = {
                        'fx': cam_meta['fx'],
                        'fy': cam_meta['fy'],
                        'cx': cam_meta['cx'],
                        'cy': cam_meta['cy'],
                        'width': cam_meta['width'],
                        'height': cam_meta['height']
                    }
                
                # dataset.intrinsics = meta_intrinsics
                camera_intrinsics = CameraIntrinsicsBase(
                    fx=meta_intrinsics['fx'],
                    fy=meta_intrinsics['fy'],
                    cx=meta_intrinsics['cx'],
                    cy=meta_intrinsics['cy'],
                    width=meta_intrinsics['width'],
                    height=meta_intrinsics['height']
                )
                dataset._meta.camera.intrinsics = camera_intrinsics
                print(f"[INFO] Using camera intrinsics from meta.json")
            except Exception as e:
                print(f"Warning: Failed to extract intrinsics from meta.json: {e}")
                # Fall back to provided intrinsics
                if intrinsics is not None:
                    dataset.intrinsics = intrinsics
                    camera_intrinsics = CameraIntrinsicsBase(
                        fx=intrinsics['fx'],
                        fy=intrinsics['fy'],
                        cx=intrinsics['cx'],
                        cy=intrinsics['cy'],
                        width=intrinsics['width'],
                        height=intrinsics['height']
                    )
                    dataset._meta.camera.intrinsics = camera_intrinsics
        elif intrinsics is not None:
            dataset.intrinsics = intrinsics
            # Update the meta camera intrinsics
            camera_intrinsics = CameraIntrinsicsBase(
                fx=intrinsics['fx'],
                fy=intrinsics['fy'],
                cx=intrinsics['cx'],
                cy=intrinsics['cy'],
                width=intrinsics['width'],
                height=intrinsics['height']
            )
            # Update the meta camera intrinsics
            dataset._meta.camera.intrinsics = camera_intrinsics
    
        return dataset
    
    def get_per_object(self, obj_idx): ##
        object_mask = np.equal(self._mask, obj_idx)
        if not object_mask.any():
            assert False, f"Object {obj_idx} not found in mask"
        max_depth = 3 ###!!!
        ###
        if self._depth is not None:
            self._depth[self._depth > max_depth] = 0
        ###
        if self._depth is not None:
            if not (self._mask.shape[:2] == self._depth.shape[:2] == self._color.shape[:2]):
                assert False, "depth, mask, and rgb should have the same shape"
        ###
        intrinsics = self._meta.camera.intrinsics
        intrinsic_matrix = np.array([
            [intrinsics.fx, 0,             intrinsics.cx], 
            [0,             intrinsics.fy, intrinsics.cy], 
            [0,             0,             1]
            ], dtype=np.float32)
        
        img_width, img_height = self._color.shape[1], self._color.shape[0]
        scale_x = img_width / intrinsics.width
        scale_y = img_height / intrinsics.height
        intrinsic_matrix[0] *= scale_x
        intrinsic_matrix[1] *= scale_y

        coord_2d = get_2d_coord_np(img_width, img_height).transpose(1, 2, 0)

        ys, xs = np.argwhere(object_mask).transpose(1, 0)
        rmin, rmax, cmin, cmax = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        rmin, rmax, cmin, cmax = get_bbox([rmin, cmin, rmax, cmax], img_height, img_width)

        # here resize and crop to a fixed size 224 x 224
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bbox_center, scale = aug_bbox_eval(bbox_xyxy, img_height, img_width)

        # crop and resize
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

        roi_rgb_ = crop_resize_by_warp_affine(
            self._color, bbox_center, scale, self._img_size, interpolation=cv2.INTER_LINEAR
        )
        roi_rgb = rgb_transform(roi_rgb_)
        
        # Compute ROI -> original image affine transform (2x3) so callers can map ROI pixel coords back
        try:
            roi_to_img_trans = get_affine_transform(bbox_center, scale, 0, self._img_size, inv=True)
        except Exception as e:
            print(f"Error computing affine transform: {e}")
        mask_target = self._mask.copy().astype(np.float32)
        mask_target[self._mask != obj_idx] = 0.0
        mask_target[self._mask == obj_idx] = 1.0

        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)
        if self._depth is not None:
            roi_depth = crop_resize_by_warp_affine(
                self._depth, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST
            )
            roi_depth = np.expand_dims(roi_depth, axis=0)
            valid = (np.squeeze(roi_depth, axis=0) > 0) * (np.squeeze(roi_mask, axis=0) > 0)
            xs, ys = np.argwhere(valid).transpose(1, 0)
            valid = valid.reshape(-1)
            pcl_in = depth_to_pcl(roi_depth, intrinsic_matrix, roi_coord_2d, valid)

            if len(pcl_in) < 10:
                # assert False, f"Not enough points for pose estimation. {len(pcl_in)} points found"
                print(f"Warning: Not enough points for pose estimation. {len(pcl_in)} points found")
                return None
            ids, pcl_in = sample_points(pcl_in, self._n_pts)
            xs, ys = xs[ids], ys[ids]

        data = {}
        if self._depth is not None:
            data['pcl_in'] = torch.as_tensor(pcl_in.astype(np.float32)).contiguous()
            data['roi_xs'] = torch.as_tensor(np.ascontiguousarray(xs), dtype=torch.int64).contiguous()
            data['roi_ys'] = torch.as_tensor(np.ascontiguousarray(ys), dtype=torch.int64).contiguous()
        data['roi_rgb'] = torch.as_tensor(np.ascontiguousarray(roi_rgb), dtype=torch.float32).contiguous()
        data['roi_rgb_'] = torch.as_tensor(np.ascontiguousarray(roi_rgb_), dtype=torch.uint8).contiguous()
        data['roi_to_img_trans'] = torch.as_tensor(np.ascontiguousarray(roi_to_img_trans), dtype=torch.float32).contiguous()
        data['roi_center_dir'] = torch.as_tensor(pixel2xyz(img_height, img_height, bbox_center, intrinsics), dtype=torch.float32).contiguous()

        return data
    
    def get_objects(self, mode='full'): ##
        obj_idx = np.unique(self._mask)
        obj_idx = obj_idx[(obj_idx != 0) & (obj_idx != 255)]  # remove background and unknown/invalid labels
        objects = {}
        labels = []

        for idx in obj_idx:
            obj = self.get_per_object(idx)
            if obj is None:
                continue
            
            labels.append(int(idx))

            for key, value in obj.items():
                if key not in objects:
                    objects[key] = []
                objects[key].append(value)
        
        for key, value in objects.items():
            objects[key] = torch.stack(value, dim=0)
            if mode != 'rgb':
                if 'pcl_in' not in objects:
                    raise ValueError(f"No valid objects found / no pcl_in produced. Mask unique values: {np.unique(self._mask)}")
                
        if mode != 'rgb':
            try:
                PC_da = objects['pcl_in'].to(self._device)
            except:
                print(f"Warning: No valid pcl_in found for any object. Mask unique values: {np.unique(self._mask)}")
                return None
        else:
            PC_da = None

        data = {}
        data['labels'] = labels                     # list of object ids
        data['pts'] = PC_da                         # [bs, 1024, 3]
        data['pts_color'] = PC_da                   # [bs, 1024, 3]
        data['roi_rgb'] = objects['roi_rgb'].to(self._device)   # [bs, 3, imgsize, imgsize]
        # also expose the unnormalized ROI uint8 images and ROI->image transform
        if 'roi_rgb_' in objects:
            data['roi_rgb_'] = objects['roi_rgb_'].to(self._device)  # [bs, H, W, 3] uint8
        if 'roi_to_img_trans' in objects:
            data['roi_to_img_trans'] = objects['roi_to_img_trans'].to(self._device)  # [bs, 2, 3]
        assert data['roi_rgb'].shape[-1] == data['roi_rgb'].shape[-2]
        assert data['roi_rgb'].shape[-1] % 14 == 0

        data['roi_center_dir'] = objects['roi_center_dir'].to(self._device)     # [bs, 3]

        """ zero center """
        if mode != 'rgb':
            data['roi_xs'] = objects['roi_xs'].to(self._device)     # [bs, 1024]
            data['roi_ys'] = objects['roi_ys'].to(self._device)     # [bs, 1024]
            num_pts = data['pts'].shape[1]
            zero_mean = torch.mean(data['pts'][:, :, :3], dim=1)
            # data['zero_mean_pts'] = copy.deepcopy(data['pts'])
            data['zero_mean_pts'] = data['pts'].clone()
            data['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
            data['pts_center'] = zero_mean

        return data

    @property
    def cam_intrinsics(self):
        return self._meta.camera.intrinsics
    
    @cam_intrinsics.setter
    def cam_intrinsics(self, intrinsics):
        self._meta.camera.intrinsics = intrinsics

class OmniXValDataset(data.IterableDataset):
    def __init__(self, shard_path, transform=None, drop_step=1, per_object=False):
        """
        Args:
            shard_path: Path pattern to shards
            transform: Optional transforms
            drop_step: Keep 1 every drop_step samples (e.g., drop_step=5 keeps every 5th sample)
            per_object: If True, yield each object in an image separately (for evaluation).
        """
        self.shard_path = shard_path
        self.transform = transform
        self.drop_step = max(1, drop_step)
        self.per_object = per_object
        
    def __iter__(self):
        dataset = wds.WebDataset(self.shard_path, shardshuffle=False)
        
        count = 0
        for sample in dataset:
            # keep every Nth sample
            if self.drop_step > 1:
                if count % self.drop_step != 0:
                    count += 1
                    continue
            
            # Decode the sample manually
            try:
                decoded_sample = self._decode_sample(sample)
            except Exception as e:
                print(f"Error decoding sample: {e}")
                count += 1
                continue
            
            if self.transform:
                decoded_sample = self.transform(decoded_sample)
            
            count += 1
            # Handle per_object mode
            if self.per_object and isinstance(decoded_sample, list):
                for s in decoded_sample:
                    if s is not None:
                        yield s
            elif decoded_sample is not None:
                yield decoded_sample
    
    @staticmethod
    def _decode_sample(sample):
        """Decode webdataset sample to standard format"""
        try:
            rgb_bytes = sample["rgb.pyd"]
            rgb_shape = pickle.loads(sample["rgb_shape.pyd"])
            rgb_dtype = np.dtype(sample["rgb_dtype.txt"].decode('utf-8'))
            rgb = np.frombuffer(rgb_bytes, dtype=rgb_dtype).reshape(rgb_shape).copy()

            
            depth_bytes = sample["depth.pyd"]
            depth_shape = pickle.loads(sample["depth_shape.pyd"])
            depth_dtype = np.dtype(sample["depth_dtype.txt"].decode('utf-8'))
            depth = np.frombuffer(depth_bytes, dtype=depth_dtype).reshape(depth_shape).copy()
            
            mask_bytes = sample["mask.pyd"]
            mask_shape = pickle.loads(sample["mask_shape.pyd"])
            mask_dtype = np.dtype(sample["mask_dtype.txt"].decode('utf-8'))
            mask = np.frombuffer(mask_bytes, dtype=mask_dtype).reshape(mask_shape).copy()
            
            meta = json.loads(sample["meta.json"].decode('utf-8'))
            
            return {
                "rgb": rgb,
                "depth": depth,
                "mask": mask,
                "meta": meta
            }
        except Exception as e:
            print(f"Decoder error: {e}")
            print(f"Sample keys: {list(sample.keys())}")
            raise

class rawValDataset(data.IterableDataset):
    """Iterable dataset that loads raw image samples from a root directory for validation/evaluation.
    
    Similar to OmniXValDataset but works with raw file format instead of webdataset tars.
    Supports per_object mode for evaluation where each object is processed separately.
    """
    
    def __init__(self, data_path, transform=None, drop_step=1, per_object=False):
        """
        Args:
            data_path: Path to root directory containing sample directories
            transform: Optional transforms to apply
            drop_step: Keep 1 every drop_step samples
            per_object: If True, yield each object separately (for evaluation)
        """
        self.data_path = data_path
        self.transform = transform
        self.drop_step = max(1, drop_step)
        self.per_object = per_object
        
    def __iter__(self):
        # Discover all sample directories
        directories = rawDataset._collect_directories(self.data_path, split=None)
        
        count = 0
        for directory in directories:
            # Discover samples in this directory
            raw_samples = rawDataset._discover_samples(directory)
            
            for idx_str, color_path, depth_path, mask_path, meta_path in raw_samples:
                # Apply drop_step filtering
                if self.drop_step > 1:
                    if count % self.drop_step != 0:
                        count += 1
                        continue
                
                count += 1
                
                # Load the sample (similar to _decode_sample)
                try:
                    rgb = load_color(color_path)
                    depth = load_depth(depth_path)
                    mask = load_mask(mask_path) if mask_path is not None else None
                    
                    if meta_path is not None:
                        with open(meta_path) as f:
                            meta = json.load(f)
                    else:
                        meta = {}
                    
                    sample = {
                        "rgb": rgb,
                        "depth": depth,
                        "mask": mask,
                        "meta": meta,
                        "index": idx_str,
                    }
                except Exception as e:
                    print(f"Error loading sample from {directory}: {e}")
                    continue
                
                # Apply transform
                if self.transform:
                    sample = self.transform(sample)
                
                # Handle per_object mode
                if self.per_object and isinstance(sample, list):
                    for s in sample:
                        if s is not None:
                            yield s
                elif sample is not None:
                    yield sample

class rawDataset(data.Dataset):
    """Dataset that loads raw image samples from a root directory.

    The root directory may contain multiple sub-directories, each holding
    samples for one scene/sequence.  A flat root (samples directly inside
    it) is also supported.

    Filename convention (zero-padded index) inside each sample directory:
        <index>_color.png       – RGB image (required)
        <index>_depth.exr/.png  – depth map, metres (required)
        <index>_mask.png/.exr   – instance mask (optional)
        <index>_meta.json       – metadata (optional)
        <index>_kp.json         – keypoint annotations (optional, keypoint mode,
                                  also accepts _keypoints.json as fallback)
                                  Format: [{"class": int, "x": float, "y": float}, ...]

    When keypoint=False (default), __getitem__ returns:
        {"rgb", "depth", "mask", "meta", "index"}
        matching the transform pipeline (ParseMetaData, CropAndResize, …).

    When keypoint=True, the dataset expands each frame into per-instance
    samples at init time (same strategy as InstanceKeypointDataset), and
    __getitem__ returns:
        {"image", "coords", "visibility", "image_id"}
    """

    _MASK_SUFFIXES  = ("_mask.png",  "_mask.exr")
    _DEPTH_SUFFIXES = ("_depth.exr", "_depth.png")

    def __init__(self, path: str, transform=None, keypoint: bool = False,
                 img_size: int = 224, augment: bool = True,
                 num_classes: int = 2, num_keypoints_per_class: int = 1,
                 split: str = None, per_object: bool = False):
        self.path = path
        self.transform = transform
        self.keypoint = keypoint
        self.img_size = img_size
        self.augment = augment
        self.per_object = per_object
        self.num_classes = num_classes
        self.num_keypoints_per_class = num_keypoints_per_class
        self.split = split

        directories = self._collect_directories(path, split=split)


        raw_samples = []
        for d in directories:
            raw_samples.extend(self._discover_samples(d))

        if not raw_samples:
            raise FileNotFoundError(
                f"No valid samples found under '{path}'. "
                "Expected files like 000000_color.png / 000000_depth.exr."
            )

        if keypoint:
            # _expand_keypoint_samples needs (samples, num_classes); directory is
            # embedded in each sample's color_path so no separate arg needed.
            self.samples = self._expand_keypoint_samples(raw_samples, num_classes)
            if not self.samples:
                raise FileNotFoundError(
                    f"No valid instance keypoint samples found under '{path}'. "
                    "Ensure mask files exist alongside the color images."
                )
        else:
            self.samples = raw_samples

        print(f"[rawDataset] Loaded {len(self.samples)} samples from '{path}'"
              + (f" (split='{split}')" if split else ""))
        
    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_directories(root: str, split: str = None) -> list:
        """Return directories that directly contain sample files.

        Walks the directory tree recursively, so it supports both flat layouts
        (colour images directly under *root* or one level below) and
        hierarchical layouts such as:
            {root}/{subset}/{train|val}/{scene}/{index}/

        When *split* is given (e.g. ``'train'`` or ``'val'``) only directories
        whose absolute path contains that segment between two separators are
        included, enabling per-split loading from a shared root.
        """
        import re
        color_pattern = re.compile(r'^\d+_color\.png$')

        # Fast path: root itself contains color images
        try:
            entries = os.listdir(root)
        except PermissionError:
            return []
        if any(color_pattern.match(e) for e in entries):
            return [root]

        # Recursively find all directories that directly contain color images
        sample_dirs = []
        # Normalise split token for path-segment matching
        split_seg = (os.sep + split + os.sep) if split else None
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()  # deterministic traversal order
            if not any(color_pattern.match(f) for f in filenames):
                continue
            if split_seg is not None:
                # Check that the split segment appears between two separators
                normalised = os.path.abspath(dirpath) + os.sep
                if split_seg not in normalised:
                    continue
            sample_dirs.append(dirpath)

        return sorted(sample_dirs)

    @classmethod
    def _discover_samples(cls, directory: str) -> list:
        """Return sorted list of (index_str, color_path, depth_path, mask_path, meta_path)."""
        import re
        pattern = re.compile(r'^(\d+)_color\.png$')
        samples = []
        for fname in sorted(os.listdir(directory)):
            m = pattern.match(fname)
            if m is None:
                continue
            idx = m.group(1)
            color_path = os.path.join(directory, fname)

            depth_path = None
            for suf in cls._DEPTH_SUFFIXES:
                candidate = os.path.join(directory, idx + suf)
                if os.path.exists(candidate):
                    depth_path = candidate
                    break
            # depth is optional for keypoint-only data

            mask_path = None
            for suf in cls._MASK_SUFFIXES:
                candidate = os.path.join(directory, idx + suf)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            meta_path = os.path.join(directory, idx + "_meta.json")
            if not os.path.exists(meta_path):
                meta_path = None

            samples.append((idx, color_path, depth_path, mask_path, meta_path))
        return samples

    @classmethod
    def _expand_keypoint_samples(cls, raw_samples: list,
                                  num_classes: int) -> list:
        """Expand frame-level entries into per-instance keypoint samples.

        Loads each mask at init time to enumerate object instances, matching
        the InstanceKeypointDataset approach so __len__ is always accurate.
        The directory is derived from each sample's color_path so samples
        from multiple directories are handled transparently.
        """
        kp_samples = []
        for (idx_str, color_path, depth_path, mask_path, meta_path) in raw_samples:
            if mask_path is None:
                continue

            # Optional per-frame keypoints file (same directory as color image)
            sample_dir = os.path.dirname(color_path)
            # Accept both _kp.json and _keypoints.json
            kp_path = os.path.join(sample_dir, idx_str + "_kp.json")
            if not os.path.exists(kp_path):
                kp_path = os.path.join(sample_dir, idx_str + "_keypoints.json")
                # print(kp_path)
                
            keypoints = []
            if os.path.exists(kp_path):
                with open(kp_path) as f:
                    kp_data = json.load(f)
                # Support both a bare list and {"keypoints": [...]} wrapper
                if isinstance(kp_data, list):
                    keypoints = kp_data
                else:
                    keypoints = kp_data.get("keypoints", [])
            else:
                # print("error, no kp data")
                continue

            mask = load_mask(mask_path)  # H×W uint8
            if mask is None:
                continue

            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids > 0]  # exclude background

            for obj_id in unique_ids:
                binary_mask = (mask == int(obj_id))
                if not binary_mask.any():
                    continue

                ys, xs = np.where(binary_mask)
                rmin, rmax = int(ys.min()), int(ys.max())
                cmin, cmax = int(xs.min()), int(xs.max())

                h = rmax - rmin
                w = cmax - cmin
                padding_x = int(w * 0.15)
                padding_y = int(h * 0.15)
                rmin = max(0, rmin - padding_y)
                rmax = min(mask.shape[0], rmax + padding_y)
                cmin = max(0, cmin - padding_x)
                cmax = min(mask.shape[1], cmax + padding_x)

                # Keep only keypoints that fall inside this instance's mask
                instance_kpts = []
                for kps in keypoints:
                    for kp in kps['keypoints']:
                        c = kp['kp_class']  # Default to class 0 if not present
                        kx, ky = int(kp['x']), int(kp['y'])
                        if not (0 <= kx < mask.shape[1] and 0 <= ky < mask.shape[0]):
                            continue
                        if binary_mask[ky, kx] and c < num_classes:
                            instance_kpts.append({
                                'class': c,
                                'x': kx - cmin,  # local bbox coords
                                'y': ky - rmin,
                            })

                kp_samples.append({
                    'color_path':  color_path,
                    'bbox':        (cmin, rmin, cmax, rmax),
                    'keypoints':   instance_kpts,
                    'mask_id': int(obj_id),
                    'index':       idx_str,
                })

        return kp_samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        if self.keypoint:
            return self._getitem_keypoint(index)
        else:
            return self._getitem_pose(index)

    # ------------------------------------------------------------------
    # Per-mode item getters
    # ------------------------------------------------------------------

    def _getitem_pose(self, index: int) -> dict:
        idx_str, color_path, depth_path, mask_path, meta_path = self.samples[index]

        rgb   = load_color(color_path)
        depth = load_depth(depth_path)
        mask  = load_mask(mask_path) if mask_path is not None else None
        # mask[mask == 255] = 0

        if meta_path is not None:
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        sample = {
            "rgb":   rgb,
            "depth": depth,
            "mask":  mask,
            "meta":  meta,
            "index": idx_str,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        # Return the sample as-is (could be a single dict or a list of dicts)
        # The DataLoader's flatten_per_object function will handle lists
        return sample

    def _getitem_keypoint(self, index: int) -> dict:
        sample = self.samples[index]

        image = cv2.imread(sample['color_path'])
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {sample['color_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cmin, rmin, cmax, rmax = sample['bbox']
        cmax = min(cmax, image.shape[1])
        rmax = min(rmax, image.shape[0])
        cropped = image[rmin:rmax, cmin:cmax]
        orig_h, orig_w = cropped.shape[:2]

        coords_pixel  = [[kp['x'], kp['y']] for kp in sample['keypoints']]
        visibility_list = [1.0] * len(coords_pixel)
        class_ids     = [kp['class'] for kp in sample['keypoints']]

        if self.augment:
            cropped, coords_pixel, visibility_list = self._apply_augmentation(
                cropped, coords_pixel, visibility_list, orig_w, orig_h
            )

        image_resized = cv2.resize(cropped, (self.img_size, self.img_size))
        scale_x = self.img_size / orig_w if orig_w > 0 else 1.0
        scale_y = self.img_size / orig_h if orig_h > 0 else 1.0

        coords     = torch.zeros(self.num_classes, self.num_keypoints_per_class, 2)
        visibility = torch.zeros(self.num_classes, self.num_keypoints_per_class)
        class_counts = [0] * self.num_classes

        for coord, vis, c in zip(coords_pixel, visibility_list, class_ids):
            k = class_counts[c]
            if k >= self.num_keypoints_per_class:
                continue
            coords[c, k]     = torch.tensor([coord[0] * scale_x / self.img_size,
                                              coord[1] * scale_y / self.img_size])
            visibility[c, k] = vis
            class_counts[c] += 1

        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return {
            'image':      image_tensor,
            'coords':     coords,
            'visibility': visibility,
            'mask_id':   sample['mask_id'],
        }

    # ------------------------------------------------------------------
    # Augmentation (mirrors InstanceKeypointDataset._apply_augmentation)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_augmentation(image, coords, visibility, w, h):
        flip_prob = np.random.rand()
        if flip_prob < 0.3:
            image  = cv2.flip(image, 1)
            coords = [[w - x, y] for x, y in coords]
        elif flip_prob < 0.4:
            image  = cv2.flip(image, 0)
            coords = [[x, h - y] for x, y in coords]
        elif flip_prob < 0.5:
            image  = cv2.flip(image, -1)
            coords = [[w - x, h - y] for x, y in coords]

        if np.random.rand() < 0.7:
            alpha = np.random.uniform(0.8, 1.2)
            beta  = np.random.uniform(-20, 20)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        if np.random.rand() < 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] += np.random.uniform(-10, 10)
            hsv[:, :, 0]  = np.clip(hsv[:, :, 0], 0, 179)
            hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)
            hsv[:, :, 1]  = np.clip(hsv[:, :, 1], 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if np.random.rand() < 0.6:
            angle  = np.random.uniform(-15, 15)
            scale  = np.random.uniform(0.9, 1.1)
            tx     = np.random.uniform(-0.1, 0.1) * w
            ty     = np.random.uniform(-0.1, 0.1) * h
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            new_coords, new_visibility = [], []
            for (x, y), vis in zip(coords, visibility):
                transformed = M @ np.array([x, y, 1.0])
                new_x, new_y = transformed[0], transformed[1]
                if 0 <= new_x < w and 0 <= new_y < h:
                    new_coords.append([new_x, new_y])
                    new_visibility.append(vis)
                else:
                    new_coords.append([np.clip(new_x, 0, w - 1), np.clip(new_y, 0, h - 1)])
                    new_visibility.append(0.0)
            coords, visibility = new_coords, new_visibility

        return image, coords, visibility

