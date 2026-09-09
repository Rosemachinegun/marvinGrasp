#!/usr/bin/env python3
"""感知层：封装 SAM3 分割结果后处理和 FlowPose 6D 位姿推理流程。"""

from __future__ import annotations

import argparse
import contextlib
import importlib.machinery
import json
import os
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from grasp_core.core.pose_math import select_ik_hand
from grasp_core.core.long_object_axes import canonical_long_object_pose, is_long_object
from grasp_core.core.robot_target_pose import make_target_object_pose
from grasp_core.perception.realsense_sam3 import (
    DEFAULT_BBOX_CONTAINMENT_THRESHOLD,
    DEFAULT_CONTAINMENT_MIN_AREA_RATIO,
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_DEDUP_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_SERIAL,
    CameraIntrinsics,
    CaptureBundle,
    Detection3D,
    Sam3Runner,
    imwrite_checked,
    make_artist_overlay,
    resolve_checkpoint_path,
    save_inference_result,
)
from grasp_core.tasks.cube_z_symmetry_grasp_policy import (
    apply_cube_z_symmetry_grasp_policy,
    local_minus_x_base,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLOWPOSE_ROOT = PROJECT_ROOT / "FlowPose"
FLOWPOSE_PY_RUNNER_DIR = FLOWPOSE_ROOT / "py_runners"
DEFAULT_FLOW_MODEL_PATH = PROJECT_ROOT / "model" / "FlowNet3.pth"
DEFAULT_SCALE_MODEL_PATH = PROJECT_ROOT / "model" / "ScaleNet3.pth"
DEFAULT_CAPTURE_DIR = PROJECT_ROOT / "captures" / "flowpose_realsense"
DEFAULT_DINO_REPO_CANDIDATES = [
    PROJECT_ROOT / "model" / "facebookresearch_dinov2_main",
    Path(
        "/home/kewei/anygrasp/auto_app-main/FlowPoseDocker/model/facebookresearch_dinov2_main"
    ),
    Path("/home/kewei/TJFusion/FlowPoseDocker/model/facebookresearch_dinov2_main"),
    Path("/home/kewei/.cache/torch/hub/facebookresearch_dinov2_main"),
]
DEFAULT_DINO_CKPT_CANDIDATES = [
    PROJECT_ROOT / "model" / "dinov2_vits14_pretrain.pth",
    Path(
        "/home/kewei/anygrasp/auto_app-main/FlowPoseDocker/model/dinov2_vits14_pretrain.pth"
    ),
    Path("/home/kewei/TJFusion/FlowPoseDocker/model/dinov2_vits14_pretrain.pth"),
    Path("/home/kewei/TJFusion/FlowPoseDocker/dinov2_vits14_pretrain.pth"),
    Path("/home/kewei/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"),
]


@dataclass(frozen=True)
class Sam3FrameResult:
    bundle: CaptureBundle
    prediction: dict[str, Any]
    object_count: int
    overlay: np.ndarray
    result_path: Path
    labels: list[str]
    runner_init_sec: float
    infer_sec: float
    postprocess_sec: float
    save_sec: float
    elapsed_sec: float


@dataclass(frozen=True)
class Sam3FlowPoseInput:
    """Minimal SAM3 output consumed by FlowPose.

    SAM3 supplies only segmentation metadata here. Depth and intrinsics stay on
    the capture bundle and are interpreted once by FlowPose.
    """

    color_bgr: np.ndarray
    depth_z16: np.ndarray
    depth_scale: float
    intrinsics: CameraIntrinsics
    masks: np.ndarray
    labels: list[str]
    scores: np.ndarray


@dataclass(frozen=True)
class Sam3PostprocessResult:
    """SAM3 postprocess output kept independent from runner/cache logic."""

    prediction: dict[str, Any]
    detections: list[Detection3D]
    overlay: np.ndarray
    labels: list[str]
    object_count: int


@dataclass(frozen=True)
class FlowPoseObject:
    name: str
    obj_id: list[int]
    pose: list[list[float]]
    size: list[float]
    score: float | None


@dataclass(frozen=True)
class FlowPoseResult:
    objects: list[FlowPoseObject]
    pose_all: list[Any] | None
    length_all: list[Any] | None
    elapsed_sec: float
    sam3_elapsed_sec: float
    runner_init_sec: float
    visualize_save_sec: float
    total_elapsed_sec: float
    visualization: np.ndarray
    raw_visualization: np.ndarray
    result_path: Path
    visualization_path: Path


def _is_path_under(path: str, root: Path) -> bool:
    if not path:
        return False
    try:
        return os.path.commonpath([os.path.abspath(path), str(root.resolve())]) == str(
            root.resolve()
        )
    except Exception:
        return False


def _clear_non_flowpose_cached_modules() -> None:
    prefixes = ("utils", "dataset", "inference", "networks")
    for name in list(sys.modules):
        if not any(
            name == prefix or name.startswith(prefix + ".") for prefix in prefixes
        ):
            continue
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", "") if module is not None else ""
        if not _is_path_under(module_file, FLOWPOSE_ROOT):
            sys.modules.pop(name, None)


def _register_flowpose_namespace_package(name: str, package_dir: Path) -> None:
    if not package_dir.is_dir():
        return
    module = types.ModuleType(name)
    module.__path__ = [str(package_dir)]
    module.__package__ = name
    module.__file__ = str(package_dir / "__init__.py")
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]
    module.__spec__ = spec
    sys.modules[name] = module


def prepare_flowpose_imports() -> None:
    for path in (FLOWPOSE_PY_RUNNER_DIR, FLOWPOSE_ROOT):
        path_str = str(path)
        with contextlib.suppress(ValueError):
            sys.path.remove(path_str)
        if path.is_dir():
            sys.path.insert(0, path_str)

    _clear_non_flowpose_cached_modules()
    for name in ("utils", "dataset", "inference", "networks"):
        _register_flowpose_namespace_package(name, FLOWPOSE_ROOT / name)
    _register_flowpose_namespace_package(
        "utils.transforms", FLOWPOSE_ROOT / "utils" / "transforms"
    )


def resolve_existing_path(
    raw_path: str | Path, candidates: list[Path] | None = None
) -> Path:
    path = Path(raw_path).expanduser()
    all_candidates = [path]
    if not path.is_absolute():
        all_candidates.append(PROJECT_ROOT / path)
    all_candidates.extend(candidates or [])
    for candidate in all_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Path not found. Tried: "
        + ", ".join(str(candidate) for candidate in all_candidates)
    )


def optional_existing_path(
    raw_path: str | Path | None, candidates: list[Path]
) -> Path | None:
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.exists():
            return path
        if not path.is_absolute() and (PROJECT_ROOT / path).exists():
            return PROJECT_ROOT / path
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def unpack_flowpose_output(output: Any) -> tuple[Any, Any, list[list[int]] | None]:
    if not isinstance(output, (list, tuple)):
        raise RuntimeError(f"Unexpected FlowPose output type: {type(output)}")
    if len(output) == 3:
        pose_out, length_out, actual_obj_ids = output
    elif len(output) == 2:
        pose_out, length_out = output
        actual_obj_ids = None
    else:
        raise RuntimeError(f"Unexpected FlowPose output length: {len(output)}")

    pose_all = (
        pose_out[0] if isinstance(pose_out, (list, tuple)) and pose_out else pose_out
    )
    length_all = (
        length_out[0]
        if isinstance(length_out, (list, tuple)) and length_out
        else length_out
    )
    return pose_all, length_all, actual_obj_ids


# ---------------------------------------------------------------------------
# Basic data-shaping helpers
# ---------------------------------------------------------------------------


def normalize_instance_masks(masks: Any) -> np.ndarray:
    """Return masks as a boolean ``(N, H, W)`` array with no needless copy."""

    if masks is None or len(masks) == 0:
        return np.empty((0, 0, 0), dtype=np.bool_)
    mask_array = np.asarray(masks)
    if mask_array.dtype == np.bool_:
        return mask_array
    return mask_array.astype(np.bool_, copy=False)


def as_array_dtype(array: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    """Convert ndarray dtype only when needed.

    ``ndarray.astype`` copies by default, so all FlowPose boundary conversions
    go through this helper to keep same-dtype inputs zero-copy.
    """

    return array if array.dtype == dtype else array.astype(dtype, copy=False)


def create_combined_mask(masks: np.ndarray) -> tuple[np.ndarray | None, list[list[int]]]:
    """Pack boolean instance masks into FlowPose's uint8 id mask.

    The mask normalization happens once before the loop. The loop only writes
    instance ids into the destination array, avoiding repeated ``asarray`` and
    ``astype(bool)`` calls for every object.
    """

    masks_bool = normalize_instance_masks(masks)
    if masks_bool.size == 0:
        return None, []
    h, w = masks_bool.shape[-2:]
    combined = np.zeros((h, w), dtype=np.uint8)
    obj_ids: list[list[int]] = []
    for index, mask in enumerate(masks_bool):
        obj_id = index + 1
        combined[mask] = obj_id
        obj_ids.append([obj_id, obj_id])
    return combined, obj_ids


def make_2d_detection_records(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    prompt: str,
) -> list[Detection3D]:
    """Build SAM3 result records without depth projection.

    FlowPose performs the RGB-D 3D reasoning, so this avoids the earlier extra
    median-depth localization pass while preserving the saved SAM3 JSON schema.
    """

    masks_bool = normalize_instance_masks(masks)
    detections: list[Detection3D] = []
    for object_id, (mask, box, score) in enumerate(zip(masks_bool, boxes, scores)):
        detections.append(
            Detection3D(
                object_id=object_id,
                prompt=prompt,
                score=float(score),
                bbox_xyxy=[float(value) for value in box],
                mask_area_px=int(mask.sum()),
                pixel_center_xy=None,
                depth_m=None,
                point_xyz_m=None,
            )
        )
    return detections


def postprocess_sam3_prediction(
    color_bgr: np.ndarray,
    prediction: dict[str, Any],
    prompt: str,
) -> Sam3PostprocessResult:
    """Normalize SAM3 outputs and build UI/save artifacts.

    This step intentionally stays 2D-only. It prepares masks, labels and the
    SAM3 overlay, but leaves RGB-D geometry to FlowPose so the pipeline has one
    owner for 3D pose reasoning.
    """

    masks = normalize_instance_masks(prediction["masks"])
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    detections = make_2d_detection_records(masks, boxes, scores, prompt)
    labels = make_labels(prompt, len(scores))
    overlay = make_artist_overlay(
        color_bgr, masks, boxes, scores, detections, prompt
    )
    return Sam3PostprocessResult(
        prediction={**prediction, "masks": masks},
        detections=detections,
        overlay=overlay,
        labels=labels,
        object_count=len(scores),
    )


def build_flowpose_input(sam_result: Sam3FrameResult) -> Sam3FlowPoseInput:
    """Extract only the SAM3 fields FlowPose needs."""

    bundle = sam_result.bundle
    prediction = sam_result.prediction
    return Sam3FlowPoseInput(
        color_bgr=bundle.color_image,
        depth_z16=bundle.depth_image,
        depth_scale=bundle.depth_scale,
        intrinsics=bundle.intrinsics,
        masks=normalize_instance_masks(prediction["masks"]),
        labels=sam_result.labels,
        scores=prediction["scores"],
    )


class FlowPoseRunner:
    """Reusable FlowPose interface for aligned RGB-D and instance masks."""

    def __init__(
        self,
        flow_model_path: Path,
        scale_model_path: Path,
        dino_repo_path: Path | None,
        dino_ckpt_path: Path | None,
        device: str,
    ) -> None:
        prepare_flowpose_imports()
        import torch
        from api_runner import PoseInferenceSession
        from inference.inference_helper import Flow
        from networks.dino.dino import DinoLoader
        from utils.yomni_vis import visualize_detections

        self.torch = torch
        self.PoseInferenceSession = PoseInferenceSession
        self.visualize_detections = visualize_detections
        self.device = (
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        args = argparse.Namespace(
            pretrained_flow_model_path=str(flow_model_path),
            pretrained_scale_model_path=str(scale_model_path),
            device=self.device,
            img_size=224,
            n_pts=1024,
            frame_gap_threshold=10,
            eval_repeat_num=25,
            retain_ratio=0.4,
            enable_tracking=False,
            enable_calibration=True,
            seed=0,
            dropout=0,
            use_edm_aug=False,
            log_dir="debug",
            use_pretrain=False,
            is_train=False,
            pose_mode="rot_matrix",
            optimizer="Adam",
            lr=1e-2,
            lr_decay=0.98,
            num_points=1024,
            scale_embedding=180,
            ema_rate=0.999,
            repeat_num=20,
            clustering=1,
            clustering_eps=0.01,
            clustering_minpts=0.1667,
        )
        self.args = args

        print(f"[FlowPose] loading FlowNet={flow_model_path}", flush=True)
        print(f"[FlowPose] loading ScaleNet={scale_model_path}", flush=True)
        self.flow = Flow(args)

        print(
            f"[FlowPose] loading DINO repo={dino_repo_path} ckpt={dino_ckpt_path}",
            flush=True,
        )
        self.dino_loader = DinoLoader(
            model_name="dinov2_vits14",
            device=self.device,
            local_repo_path=str(dino_repo_path) if dino_repo_path else None,
            ckpt_path=str(dino_ckpt_path) if dino_ckpt_path else None,
        )
        self.inferencer = self.PoseInferenceSession(
            self.flow,
            args,
            intrinsics=self._intrinsics_dict(
                CameraIntrinsics(fx=606.554, fy=606.399, cx=325.601, cy=252.875),
                width=640,
                height=480,
            ),
        )
        self.last_intrinsics = self.build_cam_intrinsics(
            CameraIntrinsics(fx=606.554, fy=606.399, cx=325.601, cy=252.875),
            640,
            480,
        )
        print("[FlowPose] ready", flush=True)

    @staticmethod
    def _intrinsics_dict(
        intrinsics: CameraIntrinsics, width: int, height: int
    ) -> dict[str, float | int]:
        return {
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
            "width": int(width),
            "height": int(height),
        }

    @staticmethod
    def build_cam_intrinsics(
        intrinsics: CameraIntrinsics, width: int, height: int
    ) -> SimpleNamespace:
        data = SimpleNamespace()
        data.fx = float(intrinsics.fx)
        data.fy = float(intrinsics.fy)
        data.cx = float(intrinsics.cx)
        data.cy = float(intrinsics.cy)
        data.width = int(width)
        data.height = int(height)
        data.intrinsic_matrix = np.array(
            [
                [data.fx, 0.0, data.cx],
                [0.0, data.fy, data.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        data.K = data.intrinsic_matrix
        return data

    def infer_instances(
        self,
        color_bgr: np.ndarray,
        depth_z16: np.ndarray,
        depth_scale: float,
        intrinsics: CameraIntrinsics,
        masks: np.ndarray,
        labels: list[str],
        scores: np.ndarray,
    ) -> dict[str, Any]:
        combined_mask, obj_ids = create_combined_mask(masks)
        if combined_mask is None or not obj_ids:
            return {
                "objects": [],
                "pose_all": None,
                "length_all": None,
                "elapsed_sec": 0.0,
            }

        height, width = color_bgr.shape[:2]
        self.inferencer.intrinsics = self._intrinsics_dict(intrinsics, width, height)
        self.last_intrinsics = self.build_cam_intrinsics(intrinsics, width, height)

        t0 = time.time()
        output = self.inferencer.infer(
            dino_loader=self.dino_loader,
            rgb=as_array_dtype(color_bgr, np.uint8),
            depth=as_array_dtype(depth_z16, np.float32),
            mask=combined_mask,
            obj_ids=obj_ids,
            depth_scale=float(depth_scale or 0.001),
        )
        pose_all_raw, length_all_raw, actual_obj_ids = unpack_flowpose_output(output)
        pose_all = to_jsonable(pose_all_raw)
        length_all = to_jsonable(length_all_raw)
        actual_obj_ids = actual_obj_ids or obj_ids
        objects = self._build_objects(
            pose_all, length_all, actual_obj_ids, labels, scores
        )
        return {
            "objects": objects,
            "pose_all": pose_all,
            "length_all": length_all,
            # Calibration may permute pose axes without permuting length_all.
            # Long objects must use the original, paired pose and dimensions.
            "raw_pose_all": to_jsonable(self.inferencer.last_raw_pose),
            "raw_length_all": length_all,
            "elapsed_sec": round(time.time() - t0, 4),
        }

    @staticmethod
    def _build_objects(
        pose_all: Any,
        length_all: Any,
        obj_ids: list[list[int]],
        labels: list[str],
        scores: np.ndarray,
    ) -> list[FlowPoseObject]:
        if pose_all is None or length_all is None:
            return []
        n = min(len(pose_all), len(length_all), len(obj_ids))
        objects: list[FlowPoseObject] = []
        for i in range(n):
            obj_id = obj_ids[i]
            box_id = (
                int(obj_id[1])
                if isinstance(obj_id, (list, tuple)) and len(obj_id) > 1
                else i + 1
            )
            label_index = max(0, box_id - 1)
            name = labels[label_index] if label_index < len(labels) else f"obj_{box_id}"
            score = float(scores[label_index]) if label_index < len(scores) else None
            objects.append(
                FlowPoseObject(
                    name=name,
                    obj_id=[int(obj_id[0]), int(obj_id[1])],
                    pose=pose_all[i],
                    size=length_all[i],
                    score=score,
                )
            )
        return objects

    def visualize(
        self,
        image_bgr: np.ndarray,
        masks: np.ndarray,
        labels: list[str],
        pose_all: Any,
        length_all: Any,
        normalized_indices: list[int] | None = None,
    ) -> np.ndarray:
        vis = image_bgr.copy()
        masks_bool = normalize_instance_masks(masks)
        for i, mask_bool in enumerate(masks_bool):
            color = FLOWPOSE_PALETTE[i % len(FLOWPOSE_PALETTE)]
            tinted = np.zeros_like(vis)
            tinted[mask_bool] = color
            blend = cv2.addWeighted(vis, 0.78, tinted, 0.22, 0)
            vis[mask_bool] = blend[mask_bool]
            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.drawContours(vis, contours, -1, color, 2, cv2.LINE_AA)

        if pose_all is not None and length_all is not None:
            poses = np.asarray(pose_all, dtype=np.float32)
            lengths = np.asarray(length_all, dtype=np.float32)
            if len(poses) > 0 and len(lengths) > 0:
                raw_pose = getattr(self.inferencer, "last_raw_pose", None)
                bbox_poses = None
                if raw_pose is not None:
                    with contextlib.suppress(Exception):
                        if hasattr(raw_pose, "detach"):
                            raw_pose = raw_pose.detach().cpu().numpy()
                        bbox_poses = np.asarray(raw_pose, dtype=np.float32)
                        if bbox_poses.shape[0] != poses.shape[0]:
                            bbox_poses = None
                        elif normalized_indices:
                            bbox_poses = bbox_poses.copy()
                            for index in normalized_indices:
                                bbox_poses[index] = poses[index]
                vis = self.visualize_detections(
                    vis,
                    poses,
                    lengths,
                    self.last_intrinsics,
                    color=None,
                    thickness=1,
                    axes_length=0.035,
                    bbox_poses=bbox_poses,
                )
                for i, pose in enumerate(poses):
                    if i >= len(labels):
                        break
                    center = pose[:3, 3]
                    if abs(float(center[2])) < 1e-8:
                        continue
                    u = int(
                        (self.last_intrinsics.fx * center[0] / center[2])
                        + self.last_intrinsics.cx
                    )
                    v = int(
                        (self.last_intrinsics.fy * center[1] / center[2])
                        + self.last_intrinsics.cy
                    )
                    draw_pose_label(
                        vis,
                        labels[i],
                        (u, v),
                        FLOWPOSE_PALETTE[i % len(FLOWPOSE_PALETTE)],
                    )
        return vis


FLOWPOSE_PALETTE = [
    (86, 180, 233),
    (230, 159, 0),
    (0, 158, 115),
    (213, 94, 0),
    (204, 121, 167),
    (0, 114, 178),
]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def draw_pose_label(
    image: np.ndarray, label: str, center: tuple[int, int], color: tuple[int, int, int]
) -> None:
    text = str(label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(center[0] - tw // 2, image.shape[1] - tw - 8))
    y = max(th + 8, min(center[1] - 16, image.shape[0] - baseline - 8))
    cv2.rectangle(image, (x - 5, y - th - 6), (x + tw + 5, y + baseline + 5), color, -1)
    cv2.putText(
        image, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def make_labels(prompt: str, count: int) -> list[str]:
    return [f"{prompt}_{i + 1}" for i in range(count)]


def split_prompts(prompts: str) -> list[str]:
    parts = [part.strip() for part in str(prompts).split(",")]
    return [part for part in parts if part] or [str(prompts).strip()]


def best_prompt_prediction(
    runner: Sam3Runner,
    color_bgr: np.ndarray,
    prompts: str,
) -> dict[str, Any]:
    """Run comma-separated prompts independently and keep the strongest result."""

    best: dict[str, Any] | None = None
    best_score = -1.0
    for prompt in split_prompts(prompts):
        prediction = runner.infer(color_bgr, prompt)
        scores = np.asarray(prediction.get("scores", []), dtype=np.float32)
        top_score = float(scores.max()) if scores.size else -1.0
        if best is None or top_score > best_score:
            best = prediction
            best_score = top_score
    if best is None:
        return runner.infer(color_bgr, str(prompts).strip())
    return best


def filter_prediction_by_roi(
    prediction: dict[str, Any],
    roi_xyxy: tuple[int, int, int, int] | list[int] | None,
    *,
    enabled: bool,
) -> tuple[dict[str, Any], int, int]:
    """Keep only predictions whose bbox center falls inside the configured ROI."""

    boxes = np.asarray(prediction.get("boxes", []), dtype=np.float32)
    original_count = len(boxes)
    if not enabled or roi_xyxy is None or original_count == 0:
        return prediction, original_count, original_count

    x_min, y_min, x_max, y_max = (float(value) for value in roi_xyxy)
    centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
    centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
    keep = (
        (centers_x >= x_min)
        & (centers_x <= x_max)
        & (centers_y >= y_min)
        & (centers_y <= y_max)
    )

    filtered = dict(prediction)
    for key in ("masks", "boxes", "scores"):
        values = prediction.get(key)
        if values is not None:
            filtered[key] = values[keep]
    return filtered, original_count, int(np.count_nonzero(keep))


# ---------------------------------------------------------------------------
# Pipeline execution functions
# ---------------------------------------------------------------------------


def run_sam3_job(
    runner_cache: dict[str, Sam3Runner | None],
    runner_kwargs: dict[str, Any],
    bundle: CaptureBundle,
    prompt: str,
    capture_meta_path: Path,
    capture_metadata: dict[str, Any],
    args: argparse.Namespace | None = None,
) -> Sam3FrameResult:
    job_start = time.perf_counter()
    runner_init_sec = 0.0
    if runner_cache["runner"] is None:
        init_start = time.perf_counter()
        runner_cache["runner"] = Sam3Runner(**runner_kwargs)
        runner_init_sec = time.perf_counter() - init_start
    runner = runner_cache["runner"]
    infer_start = time.perf_counter()
    prediction = best_prompt_prediction(runner, bundle.color_image, prompt)
    roi_xyxy = getattr(args, "sam3_roi_xyxy", None) if args is not None else None
    roi_filter_enabled = (
        bool(getattr(args, "sam3_roi_filter", False)) if args is not None else False
    )
    prediction, raw_count, roi_count = filter_prediction_by_roi(
        prediction,
        roi_xyxy,
        enabled=roi_filter_enabled,
    )
    if roi_filter_enabled and roi_xyxy is not None:
        print(
            "[SAM3] ROI filter "
            f"xyxy={tuple(int(value) for value in roi_xyxy)} "
            f"kept={roi_count}/{raw_count}",
            flush=True,
        )
    selected_prompt = str(prediction.get("prompt") or prompt)
    infer_sec = time.perf_counter() - infer_start

    postprocess_start = time.perf_counter()
    postprocessed = postprocess_sam3_prediction(
        bundle.color_image, prediction, selected_prompt
    )
    postprocess_sec = time.perf_counter() - postprocess_start

    save_start = time.perf_counter()
    timing = {
        "runner_init_sec": round(runner_init_sec, 4),
        "infer_sec": round(infer_sec, 4),
        "postprocess_sec": round(postprocess_sec, 4),
    }
    result_path = save_inference_result(
        capture_meta_path,
        {
            **capture_metadata,
            "sam3_timing": timing,
            "sam3_roi_filter": {
                "enabled": roi_filter_enabled,
                "roi_xyxy": list(roi_xyxy) if roi_xyxy is not None else None,
                "raw_count": raw_count,
                "kept_count": roi_count,
            },
        },
        postprocessed.detections,
        postprocessed.overlay,
    )
    save_sec = time.perf_counter() - save_start
    elapsed_sec = time.perf_counter() - job_start
    return Sam3FrameResult(
        bundle=bundle,
        prediction=postprocessed.prediction,
        object_count=postprocessed.object_count,
        overlay=postprocessed.overlay,
        result_path=result_path,
        labels=postprocessed.labels,
        runner_init_sec=round(runner_init_sec, 4),
        infer_sec=round(infer_sec, 4),
        postprocess_sec=round(postprocess_sec, 4),
        save_sec=round(save_sec, 4),
        elapsed_sec=round(elapsed_sec, 4),
    )


def run_flowpose_job(
    runner_cache: dict[str, FlowPoseRunner | None],
    runner_kwargs: dict[str, Any],
    sam_result: Sam3FrameResult,
    args: argparse.Namespace | None = None,
    base_to_camera: np.ndarray | None = None,
) -> FlowPoseResult:
    job_start = time.perf_counter()
    runner_init_sec = 0.0
    if runner_cache["runner"] is None:
        init_start = time.perf_counter()
        runner_cache["runner"] = FlowPoseRunner(**runner_kwargs)
        runner_init_sec = time.perf_counter() - init_start
    runner = runner_cache["runner"]
    flowpose_input = build_flowpose_input(sam_result)
    output = runner.infer_instances(
        color_bgr=flowpose_input.color_bgr,
        depth_z16=flowpose_input.depth_z16,
        depth_scale=flowpose_input.depth_scale,
        intrinsics=flowpose_input.intrinsics,
        masks=flowpose_input.masks,
        labels=flowpose_input.labels,
        scores=flowpose_input.scores,
    )
    raw_visualization = runner.visualize(
        flowpose_input.color_bgr,
        flowpose_input.masks,
        flowpose_input.labels,
        output.get("raw_pose_all") or output["pose_all"],
        output.get("raw_length_all") or output["length_all"],
    )
    output = apply_long_object_axes_to_flowpose_output(output, base_to_camera)
    output = apply_cube_z_symmetry_to_flowpose_output(output, args, base_to_camera)
    visualize_save_start = time.perf_counter()
    visualization = runner.visualize(
        flowpose_input.color_bgr,
        flowpose_input.masks,
        flowpose_input.labels,
        output["pose_all"],
        output["length_all"],
        normalized_indices=output.get("long_object_normalized_indices"),
    )
    result_path, visualization_path = save_flowpose_result(
        sam_result,
        {
            **output,
            "runner_init_sec": round(runner_init_sec, 4),
        },
        visualization,
    )
    visualize_save_sec = time.perf_counter() - visualize_save_start
    total_elapsed_sec = time.perf_counter() - job_start
    return FlowPoseResult(
        objects=output["objects"],
        pose_all=output["pose_all"],
        length_all=output["length_all"],
        elapsed_sec=output["elapsed_sec"],
        sam3_elapsed_sec=sam_result.elapsed_sec,
        runner_init_sec=round(runner_init_sec, 4),
        visualize_save_sec=round(visualize_save_sec, 4),
        total_elapsed_sec=round(total_elapsed_sec, 4),
        visualization=visualization,
        raw_visualization=raw_visualization,
        result_path=result_path,
        visualization_path=visualization_path,
    )


def apply_long_object_axes_to_flowpose_output(
    output: dict[str, Any],
    base_to_camera: np.ndarray | None,
) -> dict[str, Any]:
    """Canonicalize long objects before visualization, JSON export and IK.

    This is independent of cube symmetry and the selected hand. For historical
    output without raw poses, use the legacy calibrated X-long convention;
    never interpret its unpermuted dimensions as source-axis indices.
    """
    objects = list(output.get("objects") or [])
    if not any(is_long_object(obj.name) for obj in objects):
        return output
    if base_to_camera is None:
        raise ValueError("long-object Z-up normalization requires camera extrinsics")
    transform = np.asarray(base_to_camera, dtype=np.float64)
    inverse = np.linalg.inv(transform)
    poses = list(output.get("pose_all") or [])
    sizes = list(output.get("length_all") or [])
    raw_poses = output.get("raw_pose_all")
    raw_sizes = output.get("raw_length_all")
    normalized_indices = []
    for index, obj in enumerate(objects):
        if not is_long_object(obj.name):
            continue
        has_raw = raw_poses is not None and raw_sizes is not None
        source_pose = raw_poses[index] if has_raw else obj.pose
        source_size = raw_sizes[index] if has_raw else None
        pose, dimensions = canonical_long_object_pose(
            transform @ np.asarray(source_pose, dtype=np.float64),
            source_size,
            source_long_axis=None if has_raw else 0,
        )
        camera_pose = (inverse @ pose).tolist()
        size = dimensions.tolist() if dimensions is not None else list(obj.size)
        objects[index] = FlowPoseObject(obj.name, list(obj.obj_id), camera_pose, size, obj.score)
        poses[index] = camera_pose
        sizes[index] = size
        normalized_indices.append(index)
    return {
        **output,
        "objects": objects,
        "pose_all": poses,
        "length_all": sizes,
        "long_object_normalized_indices": normalized_indices,
    }


def apply_cube_z_symmetry_to_flowpose_output(
    output: dict[str, Any],
    args: argparse.Namespace | None,
    base_to_camera: np.ndarray | None,
) -> dict[str, Any]:
    if args is None or base_to_camera is None:
        return output
    if not bool(getattr(args, "use_cube_z_symmetry_grasp_policy", False)):
        return output

    base_to_camera = np.asarray(base_to_camera, dtype=np.float64)
    camera_to_base = np.linalg.inv(base_to_camera)
    objects = list(output.get("objects") or [])
    if not objects:
        return output

    adjusted_objects: list[FlowPoseObject] = []
    adjusted_pose_all = list(output.get("pose_all") or [])
    for index, obj in enumerate(objects):
        camera_pose = np.asarray(obj.pose, dtype=np.float64)
        if camera_pose.shape != (4, 4) or not np.all(np.isfinite(camera_pose)):
            adjusted_objects.append(obj)
            continue

        target = make_target_object_pose(
            label=obj.name,
            frame_id=f"{obj.name}_{index + 1}",
            camera_pose=camera_pose,
            base_to_camera=base_to_camera,
            size=np.asarray(obj.size, dtype=np.float64) if obj.size else None,
            score=obj.score,
        )
        hand = select_ik_hand(target.base_xyz, args.ik_hand)
        selection = apply_cube_z_symmetry_grasp_policy(target, hand=hand, args=args)
        if selection is None:
            adjusted_objects.append(obj)
            continue

        corrected_camera_pose = camera_to_base @ np.asarray(
            selection.target.base_pose,
            dtype=np.float64,
        )
        if index < len(adjusted_pose_all):
            adjusted_pose_all[index] = corrected_camera_pose.tolist()
        adjusted_objects.append(
            FlowPoseObject(
                name=obj.name,
                obj_id=list(obj.obj_id),
                pose=corrected_camera_pose.tolist(),
                size=list(obj.size),
                score=obj.score,
            )
        )
        raw_dir = local_minus_x_base(selection.raw_candidate.pose)
        selected_dir = local_minus_x_base(selection.candidate.pose)
        side = "Y+" if selection.desired_y_sign > 0.0 else "Y-"
        print(
            "[cube_z_symmetry] "
            f"{obj.name}_{index + 1}: hand={hand} desired_side={side} "
            f"selected={selection.candidate.name} "
            f"angle={selection.candidate.angle_deg:.0f}deg "
            f"raw_minus_x=({raw_dir[0]:.3f},{raw_dir[1]:.3f},{raw_dir[2]:.3f}) "
            f"selected_minus_x=({selected_dir[0]:.3f},{selected_dir[1]:.3f},{selected_dir[2]:.3f})",
            flush=True,
        )

    return {
        **output,
        "objects": adjusted_objects,
        "pose_all": adjusted_pose_all,
    }


def save_flowpose_result(
    sam_result: Sam3FrameResult, output: dict[str, Any], visualization: np.ndarray
) -> tuple[Path, Path]:
    result_path = sam_result.result_path.with_name(
        sam_result.result_path.name.replace("_sam3.json", "_flowpose.json")
    )
    visualization_path = sam_result.result_path.with_name(
        sam_result.result_path.name.replace("_sam3.json", "_flowpose.png")
    )
    imwrite_checked(visualization_path, visualization)
    payload = {
        "source_sam3_result": str(sam_result.result_path),
        "frame_id": sam_result.bundle.frame_id,
        "depth_scale": sam_result.bundle.depth_scale,
        "color_intrinsics": asdict(sam_result.bundle.intrinsics),
        "objects": [asdict(obj) for obj in output["objects"]],
        "pose_all": output["pose_all"],
        "length_all": output["length_all"],
        "raw_pose_all": output.get("raw_pose_all"),
        "raw_length_all": output.get("raw_length_all"),
        "long_object_axis_convention": "base_z_up_x_long_y_short_v1",
        "elapsed_sec": output["elapsed_sec"],
        "flowpose_timing": {
            "runner_init_sec": output.get("runner_init_sec", 0.0),
            "infer_sec": output["elapsed_sec"],
        },
        "flowpose_visualization": str(visualization_path),
    }
    result_path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    return result_path, visualization_path
