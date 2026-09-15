import numpy as np
import torch
import sys, os
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset.infer_loader import get_infer_dataloader

_EX_T = torch.tensor(
    [
        [0.0, -0.70710678, 0.70710678],
        [-1.0, 0.0, 0.0],
        [0.0, -0.70710678, -0.70710678],
    ],
    dtype=torch.float32,
)
_X_UP_FIX = torch.tensor(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=torch.float32,
)
_Y_UP_FIX = torch.tensor(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)
_NEGATIVE_X_FIX = torch.tensor(
    [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)
_UP_AXIS_THRESHOLD_COS = 0.85

# return combined mask and obj_ids (list of [mask_label, box_id])
def _build_obj_ids(mask: np.ndarray) -> list:
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[(unique_ids != 0) & (unique_ids != 255)]
    return [[int(mid), int(mid)] for mid in unique_ids]


def _normalize_axis(axis: torch.Tensor) -> torch.Tensor:
    return axis / torch.clamp(torch.linalg.norm(axis), min=1e-6)


def _project_to_rotation(rot: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(rot)
    fixed = u @ vh
    if torch.det(fixed) < 0:
        u = u.clone()
        u[:, -1] *= -1
        fixed = u @ vh
    return fixed


def _calibrate_pose(pose_mats):
    pose_t = torch.as_tensor(pose_mats, dtype=torch.float32).clone()
    if pose_t.ndim == 2:
        pose_t = pose_t.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    ex_t = _EX_T.to(device=pose_t.device, dtype=pose_t.dtype)
    x_up_fix = _X_UP_FIX.to(device=pose_t.device, dtype=pose_t.dtype)
    y_up_fix = _Y_UP_FIX.to(device=pose_t.device, dtype=pose_t.dtype)
    negative_x_fix = _NEGATIVE_X_FIX.to(device=pose_t.device, dtype=pose_t.dtype)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=pose_t.device, dtype=pose_t.dtype)
    world_negative_x = torch.tensor([-1.0, 0.0, 0.0], device=pose_t.device, dtype=pose_t.dtype)

    for i in range(pose_t.shape[0]):
        rot_world = ex_t @ pose_t[i, :3, :3]
        x_axis_up = torch.dot(_normalize_axis(rot_world[:, 0]), world_up)
        y_axis_up = torch.dot(_normalize_axis(rot_world[:, 1]), world_up)

        if x_axis_up > _UP_AXIS_THRESHOLD_COS and x_axis_up >= y_axis_up:
            rot_world = rot_world @ x_up_fix
        elif y_axis_up > _UP_AXIS_THRESHOLD_COS:
            rot_world = rot_world @ y_up_fix

        x_axis_negative = torch.dot(rot_world[:, 0], world_negative_x)
        if x_axis_negative > 0:
            rot_world = rot_world @ negative_x_fix

        pose_t[i, :3, :3] = _project_to_rotation(ex_t.transpose(0, 1) @ rot_world)

    return pose_t[0] if squeeze_out else pose_t

# PoseInferece Object
class PoseInferenceSession:

    # PoseInference Constructor
    def __init__(
        self,
        flow,
        args,
        intrinsics: Optional[dict] = None,
    ) -> None:
        self.args = args
        # self.flow = Flow(args=self.args)
        self.flow = flow
        default_intrinsics = {
            "fx": 606.5540161132812,
            "fy": 606.3988647460938,
            "cx": 325.6007080078125,
            "cy": 252.87457275390625,
            "width": 640,
            "height": 480
        }
        self.intrinsics = intrinsics or default_intrinsics
        self.enable_tracking = self.args.enable_tracking
        self.enable_calibration = getattr(self.args, 'enable_calibration', True)
        self.frame_idx = 0
        self.last_raw_pose = None

    # Inference Function
    def infer(
        self,
        dino_loader,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        obj_ids: Optional[list] = None,
        frame_idx: Optional[int] = None,
        depth_scale: float = 0.001,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[list]]:
        
        obj_ids = obj_ids or []
        if len(obj_ids) == 0:
            obj_ids = _build_obj_ids(mask)
        
        frame_data = {
            "color": rgb,
            "depth": (depth.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16), # legacy depth scaling (Mr.Kang)
            "mask": mask,
        }

        intr = {
            "fx": self.intrinsics["fx"], "fy": self.intrinsics["fy"],
            "cx": self.intrinsics["cx"], "cy": self.intrinsics["cy"],
            "width": self.intrinsics["width"], "height": self.intrinsics["height"]
        }
        data = get_infer_dataloader(frame_data, self.args, intrinsics=intr)

        idx = self.frame_idx
        # print("Frame: ", idx)

        if self.args.enable_tracking:
            pose, length, labels = self.flow.inference(data, dino_loader=dino_loader, obj_ids=obj_ids, frame_idx=idx, enable_tracking=True)
        else:
            pose, length, labels = self.flow.inference(data, dino_loader=dino_loader, obj_ids=obj_ids, frame_idx=idx, enable_tracking=False)

        self.frame_idx = idx + 1

        if not pose or not length or pose[0] is None or length[0] is None:
            print("Inference failed for frame_idx:", idx)
            self.last_raw_pose = None
            return None, None, obj_ids

        # Build obj_ids matching actual pose outputs (only objects that succeeded)
        actual_obj_ids = [[lbl, lbl] for lbl in labels]

        self.last_raw_pose = pose[0].detach().clone() if isinstance(pose[0], torch.Tensor) else pose[0].copy()
        if self.enable_calibration:
            pose[0] = _calibrate_pose(pose[0])

        return pose, length, actual_obj_ids
