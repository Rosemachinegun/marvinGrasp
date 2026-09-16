#!/usr/bin/env python3
"""感知设备层：负责 RealSense 取流、SAM3 推理封装、采集保存和图像叠加。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERIAL = "406122070773"
DEFAULT_PROMPTS = "pen"
DEFAULT_SCORE_THRESHOLD = 0.1
DEFAULT_DEDUP_IOU_THRESHOLD = 0.4
DEFAULT_CONTAINMENT_THRESHOLD = 0.75
DEFAULT_BBOX_CONTAINMENT_THRESHOLD = 0.75
DEFAULT_CONTAINMENT_MIN_AREA_RATIO = 1.0


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CaptureBundle:
    color_image: np.ndarray
    depth_image: np.ndarray
    depth_scale: float
    frame_id: int
    device_timestamp_ms: float
    host_receive_timestamp_ns: int
    intrinsics: CameraIntrinsics


@dataclass(frozen=True)
class Detection3D:
    object_id: int
    prompt: str
    score: float
    bbox_xyxy: list[float]
    mask_area_px: int
    pixel_center_xy: list[float] | None
    depth_m: float | None
    point_xyz_m: list[float] | None


def resolve_checkpoint_path(path: str) -> Path:
    requested = Path(path).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(PROJECT_ROOT / requested)
    if path in {"/model/sam3.pt", "/perception/models/sam3.pt"}:
        candidates.append(PROJECT_ROOT / "perception" / "models" / "sam3.pt")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "SAM3 checkpoint not found. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def maybe_add_sam3_source(sam3_root: str | None) -> None:
    candidates = []
    if sam3_root:
        candidates.append(Path(sam3_root).expanduser())
    if os.environ.get("SAM3_ROOT"):
        candidates.append(Path(os.environ["SAM3_ROOT"]).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT / "perception" / "sam3",
            Path("/home/kewei/repo/sam3"),
            Path("/home/kewei/anygrasp/auto_app-main/Sam3Docker/sam3-main"),
            Path("/home/kewei/TJFusion/Sam3Docker/sam3"),
            Path("/home/kewei/TJFusion/Sam3Docker/sam3-main"),
        ]
    )

    for candidate in candidates:
        if (candidate / "sam3" / "__init__.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


class RealSenseD435:
    def __init__(self, serial: str, width: int, height: int, fps: int) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.rs: Any = None
        self.pipeline: Any = None
        self.align: Any = None
        self.depth_scale = 0.0
        self.intrinsics: CameraIntrinsics | None = None

    def open(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not importable in this Python environment."
            ) from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        profile = self._start_first_supported_profile()
        self.align = rs.align(rs.stream.color)

        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.intrinsics = CameraIntrinsics(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
        )

        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(
            f"[RealSense] opened name={name} serial={serial} "
            f"{self.width}x{self.height}@{self.fps} depth_scale={self.depth_scale}",
            flush=True,
        )

    def _start_first_supported_profile(self) -> Any:
        candidates = [
            (self.width, self.height, self.fps),
            (640, 480, 30),
            (848, 480, 30),
            (640, 480, 15),
            (424, 240, 30),
        ]
        errors: list[str] = []
        seen: set[tuple[int, int, int]] = set()
        for width, height, fps in candidates:
            key = (width, height, fps)
            if key in seen:
                continue
            seen.add(key)
            config = self.rs.config()
            if self.serial:
                config.enable_device(self.serial)
            config.enable_stream(
                self.rs.stream.color, width, height, self.rs.format.bgr8, fps
            )
            config.enable_stream(
                self.rs.stream.depth, width, height, self.rs.format.z16, fps
            )
            try:
                profile = self.pipeline.start(config)
            except RuntimeError as exc:
                errors.append(f"{width}x{height}@{fps}: {exc}")
                continue
            self.width = width
            self.height = height
            self.fps = fps
            return profile
        raise RuntimeError(
            f"RealSense could not start aligned color+depth streams for "
            f"serial={self.serial or '(default)'}. Tried: {'; '.join(errors)}"
        )

    def read(self, timeout_ms: int = 1000) -> CaptureBundle | None:
        if self.pipeline is None or self.align is None or self.intrinsics is None:
            raise RuntimeError("RealSense pipeline is not opened.")

        frames = self.pipeline.wait_for_frames(timeout_ms)
        return self._bundle_from_frames(frames)

    def read_latest(
        self,
        timeout_ms: int = 1000,
        max_drain_frames: int = 30,
    ) -> CaptureBundle | None:
        """Read a frame, then drain already-buffered frames and return the newest."""
        if self.pipeline is None:
            raise RuntimeError("RealSense pipeline is not opened.")

        latest = self.read(timeout_ms)
        for _ in range(max(max_drain_frames - 1, 0)):
            frames = self.pipeline.poll_for_frames()
            if not frames:
                break
            bundle = self._bundle_from_frames(frames)
            if bundle is not None:
                latest = bundle
        return latest

    def _bundle_from_frames(self, frames) -> CaptureBundle | None:
        if self.align is None or self.intrinsics is None:
            raise RuntimeError("RealSense pipeline is not opened.")

        host_receive_timestamp_ns = time.time_ns()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        return CaptureBundle(
            color_image=np.asanyarray(color_frame.get_data()).copy(),
            depth_image=np.asanyarray(depth_frame.get_data()).copy(),
            depth_scale=self.depth_scale,
            frame_id=int(color_frame.get_frame_number()),
            device_timestamp_ms=float(color_frame.get_timestamp()),
            host_receive_timestamp_ns=host_receive_timestamp_ns,
            intrinsics=self.intrinsics,
        )

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()


class Sam3Runner:
    def __init__(
        self,
        checkpoint_path: Path,
        score_threshold: float,
        dedup_iou_threshold: float,
        suppress_contained_masks: bool,
        containment_threshold: float,
        bbox_containment_threshold: float,
        containment_min_area_ratio: float,
        device: str,
        resolution: int,
        sam3_root: str | None,
    ) -> None:
        maybe_add_sam3_source(sam3_root)
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.torch = torch
        self.device = device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print(
            f"[SAM3] loading checkpoint={checkpoint_path} device={self.device}",
            flush=True,
        )
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            device=self.device,
            load_from_HF=False,
        )
        self.processor = Sam3Processor(
            model,
            resolution=resolution,
            device=self.device,
            confidence_threshold=score_threshold,
        )
        self.dedup_iou_threshold = dedup_iou_threshold
        self.suppress_contained_masks = suppress_contained_masks
        self.containment_threshold = containment_threshold
        self.bbox_containment_threshold = bbox_containment_threshold
        self.containment_min_area_ratio = containment_min_area_ratio
        print("[SAM3] ready", flush=True)

    def infer(self, color_bgr: np.ndarray, prompt: str) -> dict[str, Any]:
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_rgb)
        context = (
            self.torch.autocast("cuda", dtype=self.torch.bfloat16)
            if self.device == "cuda"
            else _NullContext()
        )
        with context:
            state = self.processor.set_image(image)
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(state=state, prompt=prompt)

        masks, boxes, scores = self._extract_predictions(state)
        keep = filter_predictions(
            masks=masks,
            boxes=boxes,
            scores=scores,
            dedup_iou_threshold=self.dedup_iou_threshold,
            suppress_contained_masks=self.suppress_contained_masks,
            containment_threshold=self.containment_threshold,
            bbox_containment_threshold=self.bbox_containment_threshold,
            containment_min_area_ratio=self.containment_min_area_ratio,
        )
        return {
            "prompt": prompt,
            "masks": masks[keep],
            "boxes": boxes[keep],
            "scores": scores[keep],
        }

    @staticmethod
    def _extract_predictions(
        state: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        masks_t = state.get("masks")
        boxes_t = state.get("boxes")
        scores_t = state.get("scores")
        if masks_t is None or boxes_t is None or scores_t is None:
            return (
                np.zeros((0, 0, 0), dtype=bool),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        masks = (masks_t.detach().squeeze(1) > 0).cpu().numpy()
        boxes = boxes_t.detach().float().cpu().numpy()
        scores = scores_t.detach().float().cpu().numpy()
        return masks, boxes, scores


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> bool:
        return False


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    if intersection == 0:
        return 0.0
    union = int(mask_a.sum() + mask_b.sum() - intersection)
    return float(intersection / max(union, 1))


def bbox_containment(inner: np.ndarray, outer: np.ndarray) -> float:
    x0 = max(float(inner[0]), float(outer[0]))
    y0 = max(float(inner[1]), float(outer[1]))
    x1 = min(float(inner[2]), float(outer[2]))
    y1 = min(float(inner[3]), float(outer[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(0.0, float(inner[2] - inner[0])) * max(0.0, float(inner[3] - inner[1]))
    return float(inter / max(area, 1.0))


def filter_predictions(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    dedup_iou_threshold: float,
    suppress_contained_masks: bool,
    containment_threshold: float,
    bbox_containment_threshold: float,
    containment_min_area_ratio: float,
) -> np.ndarray:
    if len(scores) == 0:
        return np.zeros((0,), dtype=np.int64)

    order = np.argsort(-scores)
    areas = masks.reshape(len(masks), -1).sum(axis=1).astype(np.float64)
    kept: list[int] = []
    for idx in order:
        discard = False
        for kept_idx in kept:
            iou = mask_iou(masks[idx], masks[kept_idx])
            if iou >= dedup_iou_threshold:
                discard = True
                break

            if not suppress_contained_masks:
                continue
            candidate_area = max(float(areas[idx]), 1.0)
            kept_area = max(float(areas[kept_idx]), 1.0)
            if kept_area < candidate_area * containment_min_area_ratio:
                continue

            intersection = int(np.logical_and(masks[idx], masks[kept_idx]).sum())
            mask_contained = intersection / candidate_area >= containment_threshold
            box_contained = (
                bbox_containment(boxes[idx], boxes[kept_idx])
                >= bbox_containment_threshold
            )
            if mask_contained or box_contained:
                discard = True
                break
        if not discard:
            kept.append(int(idx))
    return np.array(kept, dtype=np.int64)


def localize_detections(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    bundle: CaptureBundle,
    prompt: str,
) -> list[Detection3D]:
    detections: list[Detection3D] = []
    depth_m = bundle.depth_image.astype(np.float32) * float(bundle.depth_scale)
    intr = bundle.intrinsics

    for object_id, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        valid = mask & (bundle.depth_image > 0)
        if not np.any(valid):
            detections.append(
                Detection3D(
                    object_id=object_id,
                    prompt=prompt,
                    score=float(score),
                    bbox_xyxy=[float(v) for v in box],
                    mask_area_px=int(mask.sum()),
                    pixel_center_xy=None,
                    depth_m=None,
                    point_xyz_m=None,
                )
            )
            continue

        ys, xs = np.nonzero(valid)
        z = float(np.median(depth_m[valid]))
        u = float(np.median(xs))
        v = float(np.median(ys))
        x = (u - intr.cx) * z / intr.fx
        y = (v - intr.cy) * z / intr.fy
        detections.append(
            Detection3D(
                object_id=object_id,
                prompt=prompt,
                score=float(score),
                bbox_xyxy=[float(value) for value in box],
                mask_area_px=int(mask.sum()),
                pixel_center_xy=[u, v],
                depth_m=z,
                point_xyz_m=[float(x), float(y), float(z)],
            )
        )
    return detections


def save_capture(
    bundle: CaptureBundle,
    capture_dir: Path,
    *,
    save_files: bool = True,
    save_color: bool = False,
) -> tuple[Path, dict[str, Any]]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_frame{bundle.frame_id:06d}"

    color_path = capture_dir / f"{stem}_color.png"
    depth_png_path = capture_dir / f"{stem}_depth.png"
    depth_npy_path = capture_dir / f"{stem}_depth.npy"
    meta_path = capture_dir / f"{stem}_meta.json"

    metadata = {
        "color_image": str(color_path),
        "depth_image": str(depth_png_path),
        "depth_image_npy": str(depth_npy_path),
        "depth_scale": bundle.depth_scale,
        "frame_id": bundle.frame_id,
        "device_timestamp_ms": bundle.device_timestamp_ms,
        "host_receive_timestamp_ns": bundle.host_receive_timestamp_ns,
        "color_intrinsics": asdict(bundle.intrinsics),
    }
    if save_files or save_color:
        capture_dir.mkdir(parents=True, exist_ok=True)
    if save_color:
        imwrite_checked(color_path, bundle.color_image)
    if save_files:
        imwrite_checked(depth_png_path, bundle.depth_image)
        np.save(depth_npy_path, bundle.depth_image)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        append_capture_csv(capture_dir / "captures.csv", metadata)
    return meta_path, metadata


def append_capture_csv(csv_path: Path, metadata: dict[str, Any]) -> None:
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_id",
                "device_timestamp_ms",
                "host_receive_timestamp_ns",
                "depth_scale",
                "fx",
                "fy",
                "cx",
                "cy",
                "color_image",
                "depth_image",
                "depth_image_npy",
            ],
        )
        if not file_exists:
            writer.writeheader()
        intr = metadata["color_intrinsics"]
        writer.writerow(
            {
                "frame_id": metadata["frame_id"],
                "device_timestamp_ms": metadata["device_timestamp_ms"],
                "host_receive_timestamp_ns": metadata["host_receive_timestamp_ns"],
                "depth_scale": metadata["depth_scale"],
                "fx": intr["fx"],
                "fy": intr["fy"],
                "cx": intr["cx"],
                "cy": intr["cy"],
                "color_image": metadata["color_image"],
                "depth_image": metadata["depth_image"],
                "depth_image_npy": metadata["depth_image_npy"],
            }
        )


def imwrite_checked(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def save_inference_result(
    capture_meta_path: Path,
    capture_metadata: dict[str, Any],
    detections: list[Detection3D],
    overlay: np.ndarray,
) -> Path:
    result_path = capture_meta_path.with_name(
        capture_meta_path.name.replace("_meta.json", "_sam3.json")
    )
    overlay_path = capture_meta_path.with_name(
        capture_meta_path.name.replace("_meta.json", "_sam3_artist.png")
    )
    imwrite_checked(overlay_path, overlay)
    payload = {
        **capture_metadata,
        "sam3_result_image": str(overlay_path),
        "detections": [asdict(det) for det in detections],
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result_path


def make_artist_overlay(
    color_bgr: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    detections: list[Detection3D],
    prompt: str,
) -> np.ndarray:
    canvas = color_bgr.copy()

    palette = np.array(
        [
            [86, 180, 233],
            [230, 159, 0],
            [0, 158, 115],
            [213, 94, 0],
            [204, 121, 167],
            [240, 228, 66],
            [0, 114, 178],
        ],
        dtype=np.uint8,
    )

    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        color = palette[i % len(palette)].tolist()
        tint = np.zeros_like(canvas, dtype=np.uint8)
        tint[mask] = color
        tinted_canvas = cv2.addWeighted(canvas, 0.72, tint, 0.28, 0)
        canvas[mask] = tinted_canvas[mask]

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), 5)
        cv2.drawContours(canvas, contours, -1, color, 2)

        x0, y0, x1, y1 = [int(round(v)) for v in box]
        draw_corner_box(canvas, x0, y0, x1, y1, color)
        # Keep the SAM3 overlay readable: show only the object name and its
        # instance number. Confidence and 3D coordinates remain available in
        # memory for FlowPose/grasping, but are not drawn on this image.
        label = f"{prompt}_{i}"
        draw_label(canvas, label, x0, max(18, y0 - 6), color)
        if i < len(detections) and detections[i].pixel_center_xy is not None:
            u, v = detections[i].pixel_center_xy
            cv2.circle(canvas, (int(round(u)), int(round(v))), 4, (255, 255, 255), -1)
            cv2.circle(canvas, (int(round(u)), int(round(v))), 2, color, -1)

    title = f'SAM3 prompt="{prompt}" objects={len(scores)}'
    draw_label(canvas, title, 12, 28, (30, 30, 30), large=True)
    return canvas


def draw_corner_box(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: list[int],
) -> None:
    h, w = image.shape[:2]
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    length = max(12, int(min(x1 - x0, y1 - y0) * 0.22))
    thickness = 3
    shadow = (20, 20, 20)

    segments = [
        ((x0, y0), (x0 + length, y0)),
        ((x0, y0), (x0, y0 + length)),
        ((x1, y0), (x1 - length, y0)),
        ((x1, y0), (x1, y0 + length)),
        ((x0, y1), (x0 + length, y1)),
        ((x0, y1), (x0, y1 - length)),
        ((x1, y1), (x1 - length, y1)),
        ((x1, y1), (x1, y1 - length)),
    ]
    for start, end in segments:
        cv2.line(image, start, end, shadow, thickness + 2, cv2.LINE_AA)
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] | list[int],
    large: bool = False,
) -> None:
    font_scale = 0.68 if large else 0.52
    thickness = 2 if large else 1
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x = max(0, min(x, image.shape[1] - width - 2))
    y = max(height + 2, min(y, image.shape[0] - baseline - 2))
    cv2.rectangle(
        image,
        (x, y - height - baseline - 5),
        (x + width + 8, y + baseline + 4),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_live_hud(frame: np.ndarray, text: str) -> np.ndarray:
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def process_capture(
    runner_cache: dict[str, Sam3Runner | None],
    runner_kwargs: dict[str, Any],
    bundle: CaptureBundle,
    prompt: str,
    capture_meta_path: Path,
    capture_metadata: dict[str, Any],
) -> tuple[np.ndarray, Path, list[Detection3D]]:
    if runner_cache["runner"] is None:
        runner_cache["runner"] = Sam3Runner(**runner_kwargs)
    runner = runner_cache["runner"]
    prediction = runner.infer(bundle.color_image, prompt)
    masks = prediction["masks"]
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    detections = localize_detections(masks, boxes, scores, bundle, prompt)
    overlay = make_artist_overlay(
        bundle.color_image, masks, boxes, scores, detections, prompt
    )
    result_path = save_inference_result(
        capture_meta_path, capture_metadata, detections, overlay
    )
    return overlay, result_path, detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RealSense D435 + SAM3 open-vocabulary 3D localization demo"
    )
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--checkpoint-path",
        default=str(PROJECT_ROOT / "perception" / "models" / "sam3.pt"),
    )
    parser.add_argument("--sam3-root", default=None)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD
    )
    parser.add_argument(
        "--dedup-iou-threshold", type=float, default=DEFAULT_DEDUP_IOU_THRESHOLD
    )
    parser.add_argument(
        "--suppress-contained-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--containment-threshold", type=float, default=DEFAULT_CONTAINMENT_THRESHOLD
    )
    parser.add_argument(
        "--bbox-containment-threshold",
        type=float,
        default=DEFAULT_BBOX_CONTAINMENT_THRESHOLD,
    )
    parser.add_argument(
        "--containment-min-area-ratio",
        type=float,
        default=DEFAULT_CONTAINMENT_MIN_AREA_RATIO,
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument(
        "--capture-dir", default=str(PROJECT_ROOT / "captures" / "sam3_realsense")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    capture_dir = Path(args.capture_dir).expanduser()

    camera = RealSenseD435(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    camera.open()

    runner_cache: dict[str, Sam3Runner | None] = {"runner": None}
    runner_kwargs = {
        "checkpoint_path": checkpoint_path,
        "score_threshold": args.score_threshold,
        "dedup_iou_threshold": args.dedup_iou_threshold,
        "suppress_contained_masks": args.suppress_contained_masks,
        "containment_threshold": args.containment_threshold,
        "bbox_containment_threshold": args.bbox_containment_threshold,
        "containment_min_area_ratio": args.containment_min_area_ratio,
        "device": args.device,
        "resolution": args.sam3_resolution,
        "sam3_root": args.sam3_root,
    }
    futures: list[Future] = []
    executor = ThreadPoolExecutor(max_workers=1)
    live_window = "RealSense D435 live - press R to capture, Q/Esc to quit"
    result_window = "SAM3 result - annotated object"
    status = "Ready"

    try:
        cv2.namedWindow(live_window, cv2.WINDOW_NORMAL)
        while True:
            bundle = camera.read()
            if bundle is None:
                continue

            still_pending: list[Future] = []
            for future in futures:
                if not future.done():
                    still_pending.append(future)
                    continue
                try:
                    overlay, result_path, detections = future.result()
                    cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
                    cv2.imshow(result_window, overlay)
                    status = (
                        f"SAM3 done: {len(detections)} object(s), {result_path.name}"
                    )
                    print(f"[SAM3] saved result: {result_path}", flush=True)
                except Exception:
                    status = "SAM3 failed; see terminal traceback"
                    traceback.print_exc()
            futures = still_pending

            hud = (
                f"frame={bundle.frame_id} depth_scale={bundle.depth_scale:.6f} "
                f"prompt='{args.prompts}' pending={len(futures)} | R capture | Q quit | {status}"
            )
            cv2.imshow(live_window, draw_live_hud(bundle.color_image, hud))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                meta_path, metadata = save_capture(bundle, capture_dir)
                print(f"[capture] saved metadata: {meta_path}", flush=True)
                frozen_bundle = CaptureBundle(
                    color_image=bundle.color_image.copy(),
                    depth_image=bundle.depth_image.copy(),
                    depth_scale=bundle.depth_scale,
                    frame_id=bundle.frame_id,
                    device_timestamp_ms=bundle.device_timestamp_ms,
                    host_receive_timestamp_ns=bundle.host_receive_timestamp_ns,
                    intrinsics=bundle.intrinsics,
                )
                futures.append(
                    executor.submit(
                        process_capture,
                        runner_cache,
                        runner_kwargs,
                        frozen_bundle,
                        args.prompts,
                        meta_path,
                        metadata,
                    )
                )
                status = f"Captured frame {bundle.frame_id}; SAM3 running"
    finally:
        camera.close()
        if futures:
            print(
                "[SAM3] waiting for queued inference job(s) before exit...", flush=True
            )
        executor.shutdown(wait=True, cancel_futures=True)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
