#!/usr/bin/env python3
"""感知运行层：组装模型参数、收集异步 SAM3/FlowPose 结果并转换机器人目标。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import traceback
from concurrent.futures import Future
from pathlib import Path

import cv2
import numpy as np

from grasp_core.perception.flowpose_pipeline import (
    DEFAULT_DINO_CKPT_CANDIDATES,
    DEFAULT_DINO_REPO_CANDIDATES,
    FlowPoseObject,
    Sam3FrameResult,
    optional_existing_path,
    resolve_checkpoint_path,
    resolve_existing_path,
)
from grasp_core.core.types.robot_target_pose import TargetObjectPose, make_child_frame_ids, make_target_object_pose
from grasp_core.tools.request_ik_ui import (
    draw_base_target_overlay,
    print_base_target_objects,
    print_flowpose_objects,
    print_flowpose_timing,
    print_sam3_timing,
)
from grasp_core.perception.realsense_sam3 import CaptureBundle

def build_runner_kwargs(args: argparse.Namespace) -> tuple[dict, dict, Path]:
    sam3_checkpoint_path = resolve_checkpoint_path(args.sam3_checkpoint_path)
    flow_model_path = resolve_existing_path(args.flow_model_path)
    scale_model_path = resolve_existing_path(args.scale_model_path)
    dino_repo_path = optional_existing_path(
        args.dino_repo_path, DEFAULT_DINO_REPO_CANDIDATES
    )
    dino_ckpt_path = optional_existing_path(
        args.dino_ckpt_path, DEFAULT_DINO_CKPT_CANDIDATES
    )

    sam3_runner_kwargs = {
        "checkpoint_path": sam3_checkpoint_path,
        "score_threshold": args.score_threshold,
        "dedup_iou_threshold": args.dedup_iou_threshold,
        "suppress_contained_masks": args.suppress_contained_masks,
        "containment_threshold": args.containment_threshold,
        "bbox_containment_threshold": args.bbox_containment_threshold,
        "containment_min_area_ratio": args.containment_min_area_ratio,
        "device": args.sam3_device,
        "resolution": args.sam3_resolution,
        "sam3_root": args.sam3_root,
    }
    flowpose_runner_kwargs = {
        "flow_model_path": flow_model_path,
        "scale_model_path": scale_model_path,
        "dino_repo_path": dino_repo_path,
        "dino_ckpt_path": dino_ckpt_path,
        "device": args.flowpose_device,
    }
    return (
        sam3_runner_kwargs,
        flowpose_runner_kwargs,
        Path(args.capture_dir).expanduser(),
    )

def collect_sam3_results(
    futures: list[Future],
    latest_result: Sam3FrameResult | None,
    latest_overlay: np.ndarray | None,
    latest_flowpose_overlay: np.ndarray | None,
    status: str,
) -> tuple[Sam3FrameResult | None, np.ndarray | None, np.ndarray | None, str]:
    for future in futures:
        if not future.done():
            continue
        try:
            latest_result = future.result()
            latest_overlay = latest_result.overlay
            latest_flowpose_overlay = None
            status = (
                f"SAM3 done: {latest_result.object_count} object(s), "
                f"infer={latest_result.infer_sec:.2f}s total={latest_result.elapsed_sec:.2f}s. "
                "Press F for FlowPose"
            )
            print_sam3_timing(latest_result)
            print(f"[SAM3] saved result: {latest_result.result_path}", flush=True)
        except Exception:
            status = "SAM3 failed; see terminal traceback"
            traceback.print_exc()
    return latest_result, latest_overlay, latest_flowpose_overlay, status


def collect_flowpose_results(
    futures: list[Future],
    latest_overlay: np.ndarray | None,
    latest_base_targets: list[TargetObjectPose],
    status: str,
    base_to_camera: np.ndarray,
    show_base_targets: bool,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, list[TargetObjectPose], str]:
    for future in futures:
        if not future.done():
            continue
        try:
            flowpose_result = future.result()
            latest_overlay = flowpose_result.visualization
            latest_base_targets = build_base_target_objects(
                flowpose_result.objects, base_to_camera
            )
            latest_base_targets = apply_forced_object_z_to_targets(
                latest_base_targets,
                args,
            )
            if show_base_targets:
                latest_overlay = draw_base_target_overlay(
                    latest_overlay, latest_base_targets
                )
            status = (
                f"FlowPose done: {len(flowpose_result.objects)} object(s), "
                f"infer={flowpose_result.elapsed_sec:.2f}s "
                f"total={flowpose_result.total_elapsed_sec:.2f}s"
            )
            print_flowpose_objects(flowpose_result.objects)
            print_base_target_objects(latest_base_targets)
            if latest_base_targets:
                target = latest_base_targets[0]
                x, y, z = target.base_xyz.tolist()
                status += f" | base {target.frame_id}=({x:.3f},{y:.3f},{z:.3f})m"
            print_flowpose_timing(flowpose_result)
            print(f"[FlowPose] saved result: {flowpose_result.result_path}", flush=True)
        except Exception:
            latest_base_targets = []
            status = "FlowPose failed; see terminal traceback"
            traceback.print_exc()
    return latest_overlay, latest_base_targets, status


def build_base_target_objects(
    objects: list[FlowPoseObject],
    base_to_camera: np.ndarray,
) -> list[TargetObjectPose]:
    frame_ids = make_child_frame_ids(obj.name for obj in objects)
    targets: list[TargetObjectPose] = []
    for obj, frame_id in zip(objects, frame_ids, strict=False):
        pose = np.asarray(obj.pose, dtype=np.float64)
        if pose.shape != (4, 4):
            continue
        size = np.asarray(obj.size, dtype=np.float64) if obj.size else None
        targets.append(
            make_target_object_pose(
                label=obj.name,
                frame_id=frame_id,
                camera_pose=pose,
                base_to_camera=base_to_camera,
                size=size,
                score=obj.score,
            )
        )
    return targets


def apply_forced_object_z_to_targets(
    targets: list[TargetObjectPose],
    args: argparse.Namespace,
) -> list[TargetObjectPose]:
    if not bool(getattr(args, "force_object_z", False)):
        return targets

    forced_z = float(getattr(args, "forced_object_z_m", 0.0))
    if not np.isfinite(forced_z):
        print(
            f"[base_link] invalid forced object z={forced_z}; using detected z",
            flush=True,
        )
        return targets

    adjusted: list[TargetObjectPose] = []
    for target in targets:
        adjusted.append(replace_target_object_z(target, forced_z))
        print(
            "[base_link] object z override "
            f"{target.frame_id}: detected={target.base_xyz[2]:.4f}m "
            f"forced={forced_z:.4f}m",
            flush=True,
        )
    if adjusted:
        print(
            "[base_link] forced object z "
            f"to {forced_z:.4f} m for {len(adjusted)} target(s); "
            "edit tool.yaml defaults.forced_object_z_m or pass --forced-object-z-m to change it",
            flush=True,
        )
    return adjusted


def replace_target_object_z(
    target: TargetObjectPose, forced_z: float
) -> TargetObjectPose:
    """Return a copy of a detected target with only base-frame object z replaced."""
    base_pose = np.asarray(target.base_pose, dtype=np.float64).copy()
    base_pose[2, 3] = float(forced_z)
    camera_pose = np.asarray(target.camera_pose, dtype=np.float64).copy()
    return TargetObjectPose(
        label=target.label,
        frame_id=target.frame_id,
        camera_pose=camera_pose,
        base_pose=base_pose,
        size=target.size.copy() if isinstance(target.size, np.ndarray) else target.size,
        score=target.score,
    )

def freeze_bundle(bundle: CaptureBundle) -> CaptureBundle:
    return CaptureBundle(
        color_image=bundle.color_image.copy(),
        depth_image=bundle.depth_image.copy(),
        depth_scale=bundle.depth_scale,
        frame_id=bundle.frame_id,
        device_timestamp_ms=bundle.device_timestamp_ms,
        host_receive_timestamp_ns=bundle.host_receive_timestamp_ns,
        intrinsics=bundle.intrinsics,
    )
