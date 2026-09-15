#!/usr/bin/env python3
"""界面层：负责 OpenCV dashboard 拼图、状态文字和感知结果预览显示。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functools import lru_cache

import cv2
import numpy as np

from grasp_core.perception.flowpose_pipeline import FlowPoseObject, Sam3FrameResult
from grasp_core.core.types.robot_target_pose import TargetObjectPose

PANEL_COLORS = {
    "live": (45, 45, 45),
    "sam3": (0, 158, 115),
    "flowpose": (213, 94, 0),
}

def make_dashboard(
    live_bgr: np.ndarray,
    sam_bgr: np.ndarray | None,
    flowpose_bgr: np.ndarray | None,
    status: str,
    prompt: str,
    sam_pending: int,
    flowpose_pending: int,
    place_enabled: bool = True,
) -> np.ndarray:
    panel_h, panel_w = live_bgr.shape[:2]
    live_panel = fit_panel(live_bgr, panel_w, panel_h)
    sam_panel = (
        fit_panel(sam_bgr, panel_w, panel_h)
        if sam_bgr is not None
        else empty_panel(panel_w, panel_h, "SAM3", "Press A or Z to capture and segment")
    )
    flowpose_panel = (
        fit_panel(flowpose_bgr, panel_w, panel_h)
        if flowpose_bgr is not None
        else empty_panel(panel_w, panel_h, "FlowPose", "Press B after SAM3 finishes")
    )

    annotate_panel(live_panel, "RealSense Live", PANEL_COLORS["live"])
    annotate_panel(sam_panel, "SAM3 Instances", PANEL_COLORS["sam3"])
    annotate_panel(flowpose_panel, "FlowPose 6D Pose", PANEL_COLORS["flowpose"])

    body = np.hstack([live_panel, sam_panel, flowpose_panel])
    footer_h = 42
    footer = np.full((footer_h, body.shape[1], 3), 26, dtype=np.uint8)
    place_text = "auto place on" if place_enabled else "auto place off"
    text = (
        f"A:auto grasp | Z:SAM3+FlowPose | B:FlowPose | C:target | S:pause | "
        f"H:both home | L/P:grip/release | "
        f"{place_text} | Q/Esc: quit | "
        f"SAM3 pending={sam_pending} FlowPose pending={flowpose_pending} | {status}"
    )
    cv2.putText(
        footer,
        text,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([body, footer])


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image.copy()
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def empty_panel(width: int, height: int, title: str, hint: str) -> np.ndarray:
    return _empty_panel_cached(width, height, title, hint).copy()


@lru_cache(maxsize=8)
def _empty_panel_cached(width: int, height: int, title: str, hint: str) -> np.ndarray:
    panel = np.full((height, width, 3), 38, dtype=np.uint8)
    cv2.rectangle(panel, (12, 12), (width - 13, height - 13), (82, 82, 82), 1)
    cv2.putText(
        panel,
        title,
        (28, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        hint,
        (28, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    return panel


def annotate_panel(image: np.ndarray, title: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (0, 0, 0), -1)
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), color, 2)
    cv2.putText(
        image,
        title,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def print_flowpose_objects(objects: list[FlowPoseObject]) -> None:
    if not objects:
        print("[FlowPose] no valid pose output", flush=True)
        return
    for obj in objects:
        pose = np.asarray(obj.pose, dtype=np.float32)
        size = np.asarray(obj.size, dtype=np.float32)
        translation = pose[:3, 3].tolist() if pose.shape == (4, 4) else None
        print(
            f"[FlowPose] {obj.name}: score={obj.score} "
            f"translation_m={translation} size_m={size.tolist()}",
            flush=True,
        )


def print_base_target_objects(objects: list[TargetObjectPose]) -> None:
    if not objects:
        print("[base_link] no valid target pose", flush=True)
        return
    for obj in objects:
        x, y, z = obj.base_xyz.tolist()
        print(
            f"[base_link] {obj.frame_id}: x={x:.4f} y={y:.4f} z={z:.4f} m",
            flush=True,
        )


def draw_base_target_overlay(
    image: np.ndarray,
    targets: list[TargetObjectPose],
) -> np.ndarray:
    canvas = image.copy()
    lines = ["base_link targets"]
    if targets:
        for obj in targets[:6]:
            x, y, z = obj.base_xyz.tolist()
            lines.append(f"{obj.frame_id}: x={x:.3f} y={y:.3f} z={z:.3f} m")
    else:
        lines.append("no valid target")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    line_h = 22
    x0, y0 = 14, 44
    max_width = 0
    for line in lines:
        (width, _), _ = cv2.getTextSize(line, font, scale, thickness)
        max_width = max(max_width, width)
    box_w = min(canvas.shape[1] - 20, max_width + 24)
    box_h = len(lines) * line_h + 16
    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (x0 - 8, y0 - 20),
        (x0 - 8 + box_w, y0 - 20 + box_h),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    cv2.rectangle(
        canvas,
        (x0 - 8, y0 - 20),
        (x0 - 8 + box_w, y0 - 20 + box_h),
        (0, 210, 160),
        1,
        cv2.LINE_AA,
    )
    for index, line in enumerate(lines):
        color = (90, 255, 210) if index == 0 else (245, 245, 245)
        cv2.putText(
            canvas,
            line,
            (x0, y0 + index * line_h),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return canvas


def print_sam3_timing(result: Sam3FrameResult) -> None:
    print(
        "[timing] SAM3 "
        f"init={result.runner_init_sec:.4f}s "
        f"infer={result.infer_sec:.4f}s "
        f"post={result.postprocess_sec:.4f}s "
        f"save={result.save_sec:.4f}s "
        f"total={result.elapsed_sec:.4f}s",
        flush=True,
    )


def print_flowpose_timing(result) -> None:
    combined = getattr(result, "sam3_elapsed_sec", None)
    suffix = (
        f" sam3+flowpose={combined + result.total_elapsed_sec:.4f}s" if combined else ""
    )
    print(
        "[timing] FlowPose "
        f"init={result.runner_init_sec:.4f}s "
        f"infer={result.elapsed_sec:.4f}s "
        f"vis_save={result.visualize_save_sec:.4f}s "
        f"total={result.total_elapsed_sec:.4f}s"
        f"{suffix}",
        flush=True,
    )
