#!/usr/bin/env python3
"""Benchmark the real SAM3 + FlowPose deployment path in isolated processes.

Run this with the same Python environment used by the grasp application.  Each
scenario is launched in a fresh process so its before/after memory numbers are
not contaminated by CUDA's process-lifetime allocator cache.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.deployment.metrics import (  # noqa: E402
    NvidiaSmi,
    ResourceSampler,
    growth_summary,
    latency_summary,
)
from grasp_core.perception.flowpose_pipeline import (  # noqa: E402
    DEFAULT_DINO_CKPT_CANDIDATES,
    DEFAULT_DINO_REPO_CANDIDATES,
    DEFAULT_FLOW_MODEL_PATH,
    DEFAULT_SCALE_MODEL_PATH,
    FlowPoseRunner,
    optional_existing_path,
    resolve_existing_path,
    run_flowpose_job,
    run_sam3_job,
)
from grasp_core.perception.realsense_sam3 import (  # noqa: E402
    DEFAULT_BBOX_CONTAINMENT_THRESHOLD,
    DEFAULT_CONTAINMENT_MIN_AREA_RATIO,
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_DEDUP_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    CameraIntrinsics,
    CaptureBundle,
    RealSenseD435,
    Sam3Runner,
    resolve_checkpoint_path,
)


SCENARIOS = ("sam3", "flowpose", "resident", "pipeline")


@dataclass(frozen=True)
class Fixture:
    bundle: CaptureBundle
    mask: np.ndarray
    source: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deployment benchmark for the production SAM3/FlowPose stack"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=(*SCENARIOS, "all"),
        default=["all"],
        help="Scenarios run in separate subprocesses (default: all).",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--sample-interval-ms", type=float, default=100.0)
    parser.add_argument("--idle-settle-sec", type=float, default=0.5)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--prompt", default="pen")
    parser.add_argument("--color", type=Path, help="Recorded BGR/RGB image readable by OpenCV.")
    parser.add_argument("--depth", type=Path, help="Aligned uint16 depth PNG or NPY.")
    parser.add_argument("--mask", type=Path, help="Optional binary/object-id PNG or NPY for standalone FlowPose.")
    parser.add_argument("--metadata", type=Path, help="Optional JSON containing depth_scale and color_intrinsics.")
    parser.add_argument("--camera", action="store_true", help="Capture one aligned RealSense frame per scenario.")
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--fx", type=float)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--sam3-checkpoint", default=str(PROJECT_ROOT / "perception/models/sam3.pt"))
    parser.add_argument("--sam3-root", default=None)
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--flow-model", default=str(DEFAULT_FLOW_MODEL_PATH))
    parser.add_argument("--scale-model", default=str(DEFAULT_SCALE_MODEL_PATH))
    parser.add_argument("--dino-repo", default=None)
    parser.add_argument("--dino-checkpoint", default=None)
    parser.add_argument("--target-vram-mib", type=float, default=8192.0)
    parser.add_argument("--target-ram-mib", type=float, default=16384.0)
    parser.add_argument("--ram-headroom-mib", type=float, default=2048.0)
    parser.add_argument("--vram-headroom-mib", type=float, default=512.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-scenario", choices=SCENARIOS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def _read_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path) if path.suffix.lower() == ".npy" else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f"cannot decode {path}")
    return array


def _intrinsics(args: argparse.Namespace, metadata: dict[str, Any], width: int, height: int) -> CameraIntrinsics:
    raw = metadata.get("color_intrinsics", metadata.get("intrinsics", {}))
    fx = args.fx if args.fx is not None else raw.get("fx")
    fy = args.fy if args.fy is not None else raw.get("fy")
    cx = args.cx if args.cx is not None else raw.get("cx", width / 2.0)
    cy = args.cy if args.cy is not None else raw.get("cy", height / 2.0)
    if fx is None or fy is None:
        # A conservative pinhole approximation, clearly recorded in the report.
        fx = fy = float(max(width, height))
    return CameraIntrinsics(float(fx), float(fy), float(cx), float(cy))


def load_fixture(args: argparse.Namespace, scenario: str) -> Fixture:
    if args.camera:
        camera = RealSenseD435(args.camera_serial, args.camera_width, args.camera_height, args.camera_fps)
        try:
            camera.open()
            bundle = camera.read_latest(timeout_ms=5000)
            if bundle is None:
                raise RuntimeError("RealSense returned no aligned frame")
        finally:
            camera.close()
        source = {"kind": "realsense", "serial": args.camera_serial or "default"}
    else:
        if args.color is None:
            raise ValueError("--color is required unless --camera is used")
        color = cv2.imread(str(args.color), cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError(f"cannot decode color image {args.color}")
        height, width = color.shape[:2]
        metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else {}
        if args.depth is None:
            if scenario != "sam3":
                raise ValueError("--depth is required for FlowPose/resident/pipeline scenarios")
            depth = np.full((height, width), 600, dtype=np.uint16)
            synthetic_depth = True
        else:
            depth = _read_array(args.depth)
            if depth.ndim == 3:
                depth = depth[..., 0]
            if depth.shape != (height, width):
                raise ValueError(f"depth shape {depth.shape} does not match color {(height, width)}")
            depth = depth.astype(np.uint16, copy=False)
            synthetic_depth = False
        bundle = CaptureBundle(
            color_image=color,
            depth_image=depth,
            depth_scale=float(metadata.get("depth_scale", args.depth_scale)),
            frame_id=int(metadata.get("frame_id", 0)),
            device_timestamp_ms=float(metadata.get("device_timestamp_ms", 0.0)),
            host_receive_timestamp_ns=int(metadata.get("host_receive_timestamp_ns", 0)),
            intrinsics=_intrinsics(args, metadata, width, height),
        )
        source = {
            "kind": "recorded",
            "color": str(args.color.resolve()),
            "depth": str(args.depth.resolve()) if args.depth else None,
            "metadata": str(args.metadata.resolve()) if args.metadata else None,
            "synthetic_depth": synthetic_depth,
        }

    height, width = bundle.color_image.shape[:2]
    if args.mask:
        mask = _read_array(args.mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (height, width):
            raise ValueError(f"mask shape {mask.shape} does not match color {(height, width)}")
        unique_nonzero = [value for value in np.unique(mask) if value != 0]
        if not unique_nonzero:
            raise ValueError(f"mask {args.mask} contains no non-background object ids")
        # Conventional binary PNGs use 255; labeled masks use 1..254 while
        # FlowPose itself reserves 255 as invalid in the packed id image.
        if unique_nonzero == [255]:
            mask = (mask > 0)[None, ...]
            instance_ids = [255]
        else:
            instance_ids = [value for value in unique_nonzero if value != 255]
            if not instance_ids:
                raise ValueError(f"mask {args.mask} contains only invalid id 255")
            mask = np.stack([mask == value for value in instance_ids], axis=0)
        source["mask"] = str(args.mask.resolve())
        source["standalone_mask_source"] = "file"
        source["standalone_mask_instances"] = len(instance_ids)
    else:
        # Only used by standalone FlowPose/residency. Full pipeline always uses SAM3 masks.
        mask = np.zeros((1, height, width), dtype=np.bool_)
        mask[0, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = True
        source["standalone_mask_source"] = "synthetic_center_rectangle"
        source["standalone_mask_instances"] = 1
    source["pipeline_mask_source"] = "sam3_output"
    return Fixture(bundle=bundle, mask=mask, source=source)


def sam_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checkpoint_path": resolve_checkpoint_path(args.sam3_checkpoint),
        "score_threshold": args.score_threshold,
        "dedup_iou_threshold": DEFAULT_DEDUP_IOU_THRESHOLD,
        "suppress_contained_masks": True,
        "containment_threshold": DEFAULT_CONTAINMENT_THRESHOLD,
        "bbox_containment_threshold": DEFAULT_BBOX_CONTAINMENT_THRESHOLD,
        "containment_min_area_ratio": DEFAULT_CONTAINMENT_MIN_AREA_RATIO,
        "device": "cuda",
        "resolution": args.sam3_resolution,
        "sam3_root": args.sam3_root,
    }


def flow_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "flow_model_path": resolve_existing_path(args.flow_model),
        "scale_model_path": resolve_existing_path(args.scale_model),
        "dino_repo_path": optional_existing_path(args.dino_repo, DEFAULT_DINO_REPO_CANDIDATES),
        "dino_ckpt_path": optional_existing_path(args.dino_checkpoint, DEFAULT_DINO_CKPT_CANDIDATES),
        "device": "cuda",
    }


def _sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _torch_memory(torch: Any) -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"allocated_mib": None, "reserved_mib": None, "max_allocated_mib": None, "max_reserved_mib": None}
    divisor = 1024 * 1024
    return {
        "allocated_mib": torch.cuda.memory_allocated() / divisor,
        "reserved_mib": torch.cuda.memory_reserved() / divisor,
        "max_allocated_mib": torch.cuda.max_memory_allocated() / divisor,
        "max_reserved_mib": torch.cuda.max_memory_reserved() / divisor,
    }


def _pipeline_args() -> SimpleNamespace:
    return SimpleNamespace(sam3_roi_filter=False, sam3_roi_xyxy=None, use_box_z_symmetry_grasp_policy=False)


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    scenario = args.worker_scenario
    assert scenario is not None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this deployment benchmark requires an NVIDIA GPU")
    torch.cuda.set_device(args.gpu_index)
    fixture = load_fixture(args, scenario)
    sampler = ResourceSampler(args.sample_interval_ms / 1000.0, args.gpu_index)
    nvidia = sampler.nvidia
    result: dict[str, Any] = {
        "schema_version": 1,
        "scenario": scenario,
        "status": "running",
        "iterations_requested": args.iterations,
        "warmup_iterations": args.warmup,
        "fixture": fixture.source,
        "hardware": {
            "gpu": nvidia.identity(),
            "cpu": platform.processor() or platform.machine(),
            "logical_cpu_count": psutil.cpu_count(),
            "system_ram_total_mib": psutil.virtual_memory().total / (1024 * 1024),
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
    }
    sampler.start()
    result["resource_points"] = {"before_load": sampler.checkpoint()}
    sam_runner: Sam3Runner | None = None
    flow_runner: FlowPoseRunner | None = None
    try:
        init_started = time.perf_counter()
        if scenario in {"sam3", "resident", "pipeline"}:
            sam_runner = Sam3Runner(**sam_kwargs(args))
        if scenario in {"flowpose", "resident", "pipeline"}:
            flow_runner = FlowPoseRunner(**flow_kwargs(args))
        _sync(torch)
        result["initialization_ms"] = (time.perf_counter() - init_started) * 1000.0
        result["resource_points"]["after_load"] = sampler.checkpoint()
        time.sleep(max(0.0, args.idle_settle_sec))
        result["resource_points"]["loaded_idle"] = sampler.checkpoint()

        labels = [args.prompt + "_0"]
        scores = np.array([1.0], dtype=np.float32)
        sam_cache = {"runner": sam_runner}
        flow_cache = {"runner": flow_runner}
        metadata = {"frame_id": fixture.bundle.frame_id, "depth_scale": fixture.bundle.depth_scale, "color_intrinsics": asdict(fixture.bundle.intrinsics)}
        placeholder = PROJECT_ROOT / "benchmark_fixture_meta.json"

        def sam_only() -> int:
            assert sam_runner is not None
            return len(sam_runner.infer(fixture.bundle.color_image, args.prompt)["scores"])

        def flow_only() -> int:
            assert flow_runner is not None
            output = flow_runner.infer_instances(
                fixture.bundle.color_image,
                fixture.bundle.depth_image,
                fixture.bundle.depth_scale,
                fixture.bundle.intrinsics,
                fixture.mask,
                labels,
                scores,
            )
            return len(output["objects"])

        def resident() -> int:
            assert sam_runner is not None
            count = len(sam_runner.infer(fixture.bundle.color_image, args.prompt)["scores"])
            return count + flow_only()

        def pipeline() -> int:
            sam_result = run_sam3_job(
                sam_cache, sam_kwargs(args), fixture.bundle, args.prompt, placeholder,
                metadata, _pipeline_args(), False, False,
            )
            if sam_result.object_count == 0:
                raise RuntimeError(f"SAM3 found no objects for prompt {args.prompt!r}; full pipeline workload is invalid")
            flow_result = run_flowpose_job(
                flow_cache, flow_kwargs(args), sam_result, _pipeline_args(),
                np.eye(4, dtype=np.float64), False, False,
            )
            if not flow_result.objects:
                raise RuntimeError("FlowPose produced no objects from SAM3 masks")
            return len(flow_result.objects)

        invoke: Callable[[], int] = {
            "sam3": sam_only,
            "flowpose": flow_only,
            "resident": resident,
            "pipeline": pipeline,
        }[scenario]

        for _ in range(args.warmup):
            invoke()
            _sync(torch)
        torch.cuda.reset_peak_memory_stats()
        result["resource_points"]["before_measured_inference"] = sampler.checkpoint()

        latencies: list[float] = []
        rss_values: list[float] = []
        allocated_values: list[float] = []
        reserved_values: list[float] = []
        object_counts: list[int] = []
        process = psutil.Process()
        for _ in range(args.iterations):
            _sync(torch)
            started = time.perf_counter()
            object_counts.append(invoke())
            _sync(torch)
            latencies.append((time.perf_counter() - started) * 1000.0)
            rss_values.append(process.memory_info().rss / (1024 * 1024))
            allocated_values.append(torch.cuda.memory_allocated() / (1024 * 1024))
            reserved_values.append(torch.cuda.memory_reserved() / (1024 * 1024))

        result["resource_points"]["after_inference"] = sampler.checkpoint()
        result["latency"] = latency_summary(latencies)
        result["object_count"] = {
            "min": min(object_counts) if object_counts else None,
            "max": max(object_counts) if object_counts else None,
        }
        result["memory_growth"] = {
            "process_rss": growth_summary(rss_values, 128.0),
            "torch_allocated": growth_summary(allocated_values, 64.0),
            "torch_reserved": growth_summary(reserved_values, 64.0),
        }
        result["torch_cuda_memory"] = _torch_memory(torch)
        result["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="oom", error=str(exc), traceback=traceback.format_exc())
    except Exception as exc:
        result.update(status="error", error=str(exc), traceback=traceback.format_exc())
    finally:
        sampler.stop()
        result["resources"] = sampler.summary()
        result["completed_at"] = datetime.now().astimezone().isoformat()
    return result


def deployment_assessment(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "ok"]
    errors = [result["scenario"] for result in results if result.get("status") != "ok"]
    process_peaks = [
        result.get("resources", {}).get("process_gpu_mib", {}).get("max")
        for result in successful
    ]
    process_peaks = [value for value in process_peaks if value is not None]
    torch_peaks = [result.get("torch_cuda_memory", {}).get("max_reserved_mib") for result in successful]
    torch_peaks = [value for value in torch_peaks if value is not None]
    measured_peak = max(process_peaks or torch_peaks or [0.0])
    rss_peaks = [
        result.get("resources", {}).get("process_rss_mib", {}).get("max")
        for result in successful
    ]
    rss_peaks = [value for value in rss_peaks if value is not None]
    measured_rss_peak = max(rss_peaks or [0.0])
    leaks = [
        result["scenario"]
        for result in successful
        if any(item.get("suspected_leak") for item in result.get("memory_growth", {}).values())
    ]
    fits = measured_peak + args.vram_headroom_mib <= args.target_vram_mib
    fits_ram = measured_rss_peak + args.ram_headroom_mib <= args.target_ram_mib
    current_names = [str(result.get("hardware", {}).get("gpu", {}).get("name", "")) for result in successful]
    ran_on_target = bool(current_names) and all("RTX 5060" in name for name in current_names)
    if errors:
        verdict = "fail"
    elif leaks or not fits or not fits_ram:
        verdict = "fail"
    elif ran_on_target:
        verdict = "pass"
    else:
        verdict = "capacity_candidate_only"
    return {
        "verdict": verdict,
        "ran_on_rtx_5060": ran_on_target,
        "target_vram_mib": args.target_vram_mib,
        "reserved_headroom_mib": args.vram_headroom_mib,
        "measured_process_or_torch_peak_mib": measured_peak,
        "fits_target_vram_budget": fits,
        "target_ram_mib": args.target_ram_mib,
        "reserved_ram_headroom_mib": args.ram_headroom_mib,
        "measured_process_rss_peak_mib": measured_rss_peak,
        "fits_target_ram_budget": fits_ram,
        "failed_scenarios": errors,
        "leak_suspicions": leaks,
        "note": "A run on another GPU can estimate capacity, but only a run on the target RTX 5060 certifies latency, CUDA-extension compatibility, and power/thermal behavior.",
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    assessment = payload["assessment"]
    lines = [
        "# Deployment Benchmark Report",
        "",
        f"- Verdict: **{assessment['verdict']}**",
        f"- Target budget: {assessment['target_vram_mib']:.0f} MiB VRAM + {assessment['reserved_headroom_mib']:.0f} MiB headroom",
        f"- Measured process/Torch peak: {assessment['measured_process_or_torch_peak_mib']:.1f} MiB",
        f"- Target RAM budget: {assessment['target_ram_mib']:.0f} MiB with {assessment['reserved_ram_headroom_mib']:.0f} MiB headroom",
        f"- Measured process RSS peak: {assessment['measured_process_rss_peak_mib']:.1f} MiB",
        f"- Ran on RTX 5060: {assessment['ran_on_rtx_5060']}",
        "",
        "| Scenario | Status | Init ms | Mean ms | P50 ms | P95 ms | Max ms | Process GPU peak MiB | Machine GPU peak MiB | RSS peak MiB | Leak |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        latency = result.get("latency", {})
        resources = result.get("resources", {})
        leak = any(item.get("suspected_leak") for item in result.get("memory_growth", {}).values())
        fmt = lambda value: "-" if value is None else f"{value:.1f}"
        lines.append(
            "| {scenario} | {status} | {init} | {mean} | {p50} | {p95} | {maximum} | {pgpu} | {gpu} | {rss} | {leak} |".format(
                scenario=result["scenario"], status=result["status"], init=fmt(result.get("initialization_ms")),
                mean=fmt(latency.get("average_ms")), p50=fmt(latency.get("p50_ms")),
                p95=fmt(latency.get("p95_ms")), maximum=fmt(latency.get("max_ms")),
                pgpu=fmt(resources.get("process_gpu_mib", {}).get("max")),
                gpu=fmt(resources.get("gpu_used_mib", {}).get("max")),
                rss=fmt(resources.get("process_rss_mib", {}).get("max")), leak=leak,
            )
        )
    lines.extend(["", assessment["note"], "", "Full machine-readable details, checkpoints, utilization samples, and growth slopes are in `report.json`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def orchestrate(args: argparse.Namespace) -> int:
    scenarios = list(SCENARIOS) if "all" in args.scenarios else list(dict.fromkeys(args.scenarios))
    output_dir = args.output_dir or PROJECT_ROOT / "benchmark_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for scenario in scenarios:
        result_path = output_dir / f"{scenario}.json"
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker-scenario", scenario, "--worker-result", str(result_path)]
        print(f"[benchmark] starting {scenario}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        if not result_path.is_file():
            results.append({"scenario": scenario, "status": "worker_failed", "returncode": completed.returncode})
        else:
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv,
        "assessment": deployment_assessment(results, args),
        "results": results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "report.md", payload)
    print(f"[benchmark] report: {report_path}", flush=True)
    print(f"[benchmark] verdict: {payload['assessment']['verdict']}", flush=True)
    return 0 if payload["assessment"]["verdict"] in {"pass", "capacity_candidate_only"} else 2


def main() -> int:
    args = build_parser().parse_args()
    if args.iterations < 1 or args.warmup < 0:
        raise SystemExit("--iterations must be >= 1 and --warmup must be >= 0")
    if args.worker_scenario:
        assert args.worker_result is not None
        try:
            result = run_worker(args)
        except Exception as exc:
            result = {"schema_version": 1, "scenario": args.worker_scenario, "status": "error", "error": str(exc), "traceback": traceback.format_exc()}
        args.worker_result.parent.mkdir(parents=True, exist_ok=True)
        args.worker_result.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0 if result.get("status") == "ok" else 2
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
