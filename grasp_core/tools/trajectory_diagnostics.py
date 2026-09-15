#!/usr/bin/env python3
"""CSV export and plotting helpers for published motion trajectories."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grasp_core.execution.config import DEFAULT_TRAJECTORY_PLOT_DIR
from grasp_core.config.trajectory_config import (
    should_save_joint_trajectory_csv,
    should_visualize_grasp_path,
)
from grasp_core.core.math.pose import (
    checked_position,
    normalize_object_type,
    normalize_quaternion,
)
from grasp_core.core.types.robot_target_pose import TargetObjectPose
from grasp_core.planning.trajectory.constraints import effective_trajectory_step_limits

PublishedTrajectory = dict[str, object]


@dataclass(frozen=True)
class GraspPathArtifacts:
    """Saved grasp path CSV and plot files."""

    csv_path: Path
    plot_path: Path | None

def save_request_ik_grasp_path_artifacts(
    publisher: RequestIkTargetPublisher,
    target: TargetObjectPose,
    hand: str,
    args: argparse.Namespace,
) -> GraspPathArtifacts | None:
    if not should_visualize_grasp_path(args):
        print("[grasp_path] visualization disabled; skip CSV/plot", flush=True)
        return None

    trajectory = publisher.last_published_trajectory()
    if trajectory is None:
        print(
            "[grasp_path] no recorded trajectory; skip CSV/plot "
            "(check request_ik publisher startup log and press S after FlowPose)",
            flush=True,
        )
        return None

    plot_data = prepare_request_ik_trajectory_plot_data(trajectory)
    if plot_data is None:
        print("[grasp_path] recorded trajectory is empty; skip CSV/plot", flush=True)
        return None

    raw_positions, sample_positions, sample_orientations, segment_steps = plot_data
    csv_path = request_ik_grasp_path_csv_path(
        target,
        hand,
        sample_count=len(sample_positions),
        args=args,
    )
    write_request_ik_grasp_path_csv(
        csv_path,
        raw_positions=raw_positions,
        sample_positions=sample_positions,
        sample_orientations=sample_orientations,
        segment_steps=segment_steps,
    )

    plot_path = save_request_ik_grasp_path_plot_from_csv(
        csv_path,
        target=target,
        hand=hand,
        args=args,
    )

    print(
        "[grasp_path] saved grasp path CSV/plot "
        f"csv={csv_path} plot={plot_path} "
        f"samples={len(sample_positions)} raw_points={len(raw_positions)} "
        f"segment_steps={segment_steps if isinstance(segment_steps, list) else []}",
        flush=True,
    )
    return GraspPathArtifacts(csv_path=csv_path, plot_path=plot_path)


def save_request_ik_grasp_path_plot(
    publisher: RequestIkTargetPublisher,
    target: TargetObjectPose,
    hand: str,
    args: argparse.Namespace,
) -> Path | None:
    artifacts = save_request_ik_grasp_path_artifacts(publisher, target, hand, args)
    return artifacts.plot_path if artifacts is not None else None


def prepare_request_ik_trajectory_plot_data(
    trajectory: PublishedTrajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object] | None:
    raw_waypoints = trajectory.get("raw_waypoints")
    samples = trajectory.get("samples")
    segment_steps = trajectory.get("segment_steps")
    if not isinstance(raw_waypoints, list) or not isinstance(samples, list):
        return None
    if not raw_waypoints or not samples:
        return None

    raw_positions = np.asarray(
        [checked_position(item[0]) for item in raw_waypoints], dtype=np.float64
    )
    sample_positions = np.asarray(
        [checked_position(item[0]) for item in samples], dtype=np.float64
    )
    sample_orientations = np.asarray(
        [normalize_quaternion(item[1]) for item in samples], dtype=np.float64
    )
    if raw_positions.ndim != 2 or sample_positions.ndim != 2:
        return None

    return raw_positions, sample_positions, sample_orientations, segment_steps


def request_ik_trajectory_plot_path(
    target: TargetObjectPose,
    hand: str,
    *,
    sample_count: int,
    args: argparse.Namespace,
) -> Path:
    output_dir = Path(
        getattr(args, "target_trajectory_plot_dir", DEFAULT_TRAJECTORY_PLOT_DIR)
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{timestamp}_{normalize_object_type(target.label)}_{normalize_object_type(target.frame_id)}_"
        f"{hand}_grasp_path_{sample_count}pts.png"
    )
    return output_dir / filename


def request_ik_grasp_path_csv_path(
    target: TargetObjectPose,
    hand: str,
    *,
    sample_count: int,
    args: argparse.Namespace,
) -> Path:
    output_dir = Path(
        getattr(args, "target_trajectory_plot_dir", DEFAULT_TRAJECTORY_PLOT_DIR)
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{timestamp}_{normalize_object_type(target.label)}_{normalize_object_type(target.frame_id)}_"
        f"{hand}_grasp_path_{sample_count}pts.csv"
    )
    return output_dir / filename


def write_request_ik_grasp_path_csv(
    csv_path: Path,
    *,
    raw_positions: np.ndarray,
    sample_positions: np.ndarray,
    sample_orientations: np.ndarray,
    segment_steps: object,
) -> None:
    raw_indices = raw_target_sample_indices(segment_steps, len(sample_positions))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "kind",
                "index",
                "sample_index",
                "segment_index",
                "x_m",
                "y_m",
                "z_m",
                "qx",
                "qy",
                "qz",
                "qw",
            ]
        )
        for index, position in enumerate(raw_positions):
            sample_index = raw_indices[index] if index < len(raw_indices) else ""
            writer.writerow(
                [
                    "raw",
                    index,
                    sample_index,
                    max(index - 1, 0),
                    *[f"{float(value):.9f}" for value in position[:3]],
                    "",
                    "",
                    "",
                    "",
                ]
            )
        sample_segment_indices = sample_segment_index_by_steps(
            segment_steps,
            len(sample_positions),
        )
        for index, (position, orientation) in enumerate(
            zip(sample_positions, sample_orientations, strict=False)
        ):
            writer.writerow(
                [
                    "sample",
                    index,
                    index,
                    int(sample_segment_indices[index]),
                    *[f"{float(value):.9f}" for value in position[:3]],
                    *[f"{float(value):.9f}" for value in orientation[:4]],
                ]
            )


def load_request_ik_grasp_path_csv(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, object] | None:
    raw_positions: list[list[float]] = []
    sample_positions: list[list[float]] = []
    raw_sample_indices: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                position = [
                    float(row["x_m"]),
                    float(row["y_m"]),
                    float(row["z_m"]),
                ]
            except (KeyError, TypeError, ValueError):
                continue
            kind = str(row.get("kind", "")).strip().lower()
            if kind == "raw":
                raw_positions.append(position)
                try:
                    raw_sample_indices.append(float(row.get("sample_index", "")))
                except (TypeError, ValueError):
                    raw_sample_indices.append(float("nan"))
            elif kind == "sample":
                sample_positions.append(position)

    if not raw_positions or not sample_positions:
        return None

    return (
        np.asarray(raw_positions, dtype=np.float64),
        np.asarray(sample_positions, dtype=np.float64),
        np.asarray(raw_sample_indices, dtype=np.float64),
    )


def save_request_ik_grasp_path_plot_from_csv(
    csv_path: Path,
    *,
    target: TargetObjectPose,
    hand: str,
    args: argparse.Namespace,
) -> Path | None:
    csv_data = load_request_ik_grasp_path_csv(csv_path)
    if csv_data is None:
        print(f"[grasp_path] CSV has no plottable path rows: {csv_path}", flush=True)
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[grasp_path] matplotlib unavailable; CSV saved, skip plot: {exc}", flush=True)
        return None

    raw_positions, sample_positions, raw_sample_indices = csv_data
    output_path = csv_path.with_suffix(".png")
    render_request_ik_trajectory_plot(
        plt,
        output_path=output_path,
        target=target,
        hand=hand,
        raw_positions=raw_positions,
        sample_positions=sample_positions,
        raw_sample_indices=raw_sample_indices,
        args=args,
    )
    return output_path


def render_request_ik_trajectory_plot(
    plt,
    *,
    output_path: Path,
    target: TargetObjectPose,
    hand: str,
    raw_positions: np.ndarray,
    sample_positions: np.ndarray,
    raw_sample_indices: object,
    args: argparse.Namespace,
) -> None:
    fig = plt.figure(figsize=(13, 7.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_xyz = fig.add_subplot(1, 2, 2)

    ax3d.plot(
        raw_positions[:, 0],
        raw_positions[:, 1],
        raw_positions[:, 2],
        "o--",
        color="#222222",
        linewidth=1.2,
        markersize=5,
        label="raw targets",
    )
    ax3d.scatter(
        sample_positions[:, 0],
        sample_positions[:, 1],
        sample_positions[:, 2],
        s=13,
        color="#1f77b4",
        alpha=0.82,
        label="published interpolation points",
    )
    ax3d.plot(
        sample_positions[:, 0],
        sample_positions[:, 1],
        sample_positions[:, 2],
        "-",
        color="#1f77b4",
        linewidth=0.8,
        alpha=0.35,
    )
    ax3d.scatter(
        [raw_positions[0, 0]],
        [raw_positions[0, 1]],
        [raw_positions[0, 2]],
        s=75,
        color="#2ca02c",
        label="start",
    )
    ax3d.scatter(
        [target.base_xyz[0]],
        [target.base_xyz[1]],
        [target.base_xyz[2]],
        marker="x",
        s=90,
        color="#9467bd",
        label="object origin",
    )
    for index, position in enumerate(raw_positions):
        label = "start" if index == 0 else f"P{index}"
        ax3d.text(position[0], position[1], position[2], f"  {label}", fontsize=8)
    ax3d.set_xlabel("X base_link (m)")
    ax3d.set_ylabel("Y base_link (m)")
    ax3d.set_zlabel("Z base_link (m)")
    ax3d.set_title("Grasp Path in base_link")
    ax3d.legend(loc="best", fontsize=8)
    set_3d_axes_equal(
        ax3d,
        np.vstack([raw_positions, sample_positions, target.base_xyz.reshape(1, 3)]),
    )

    sample_indices = np.arange(len(sample_positions), dtype=np.int64)
    ax_xyz.plot(
        sample_indices, sample_positions[:, 0], color="#1f77b4", label="interp x"
    )
    ax_xyz.plot(
        sample_indices, sample_positions[:, 1], color="#ff7f0e", label="interp y"
    )
    ax_xyz.plot(
        sample_indices, sample_positions[:, 2], color="#2ca02c", label="interp z"
    )
    ax_xyz.scatter(
        sample_indices, sample_positions[:, 0], s=8, color="#1f77b4", alpha=0.45
    )
    ax_xyz.scatter(
        sample_indices, sample_positions[:, 1], s=8, color="#ff7f0e", alpha=0.45
    )
    ax_xyz.scatter(
        sample_indices, sample_positions[:, 2], s=8, color="#2ca02c", alpha=0.45
    )
    raw_sample_indices = np.asarray(raw_sample_indices, dtype=np.float64)
    if (
        len(raw_sample_indices) == len(raw_positions)
        and np.all(np.isfinite(raw_sample_indices))
    ):
        ax_xyz.plot(
            raw_sample_indices,
            raw_positions[:, 0],
            "o--",
            color="#1f77b4",
            alpha=0.35,
            label="raw x",
        )
        ax_xyz.plot(
            raw_sample_indices,
            raw_positions[:, 1],
            "o--",
            color="#ff7f0e",
            alpha=0.35,
            label="raw y",
        )
        ax_xyz.plot(
            raw_sample_indices,
            raw_positions[:, 2],
            "o--",
            color="#2ca02c",
            alpha=0.35,
            label="raw z",
        )
        for index, sample_index in enumerate(raw_sample_indices):
            ax_xyz.axvline(sample_index, color="#777777", linewidth=0.7, alpha=0.22)
            label = "start" if index == 0 else f"P{index}"
            ax_xyz.text(
                sample_index,
                ax_xyz.get_ylim()[1],
                label,
                va="top",
                ha="center",
                fontsize=8,
            )
    ax_xyz.set_xlabel("Interpolation sample index")
    ax_xyz.set_ylabel("Position (m)")
    ax_xyz.set_title("Published Grasp Path Samples")
    ax_xyz.grid(True, alpha=0.25)
    ax_xyz.legend(loc="best", ncol=2, fontsize=8)

    effective_step_m, effective_step_deg = effective_trajectory_step_limits(args)
    fig.suptitle(
        f"{target.label} / {target.frame_id} / {hand}: "
        f"{len(sample_positions)} grasp path sample(s)\n"
        f"effective_step<={effective_step_m:.4f}m/"
        f"{effective_step_deg:.2f}deg, "
        f"speed<={float(args.target_trajectory_speed_mps):.3f}m/s/"
        f"{float(args.target_trajectory_angular_speed_dps):.1f}deg/s, "
        f"min_steps={int(args.target_trajectory_min_steps)}, "
        f"publish={float(args.target_publish_rate_hz):.1f}Hz",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def raw_target_sample_indices(segment_steps: object, sample_count: int) -> np.ndarray:
    if not isinstance(segment_steps, list) or not segment_steps:
        return np.linspace(0, max(int(sample_count) - 1, 0), 2, dtype=np.float64)
    indices = [0.0]
    current = 0.0
    for value in segment_steps:
        try:
            current += max(float(value), 0.0)
        except (TypeError, ValueError):
            continue
        indices.append(
            min(max(current - 1.0, 0.0), max(float(sample_count) - 1.0, 0.0))
        )
    return np.asarray(indices, dtype=np.float64)


def sample_segment_index_by_steps(segment_steps: object, sample_count: int) -> np.ndarray:
    if not isinstance(segment_steps, list) or not segment_steps:
        return np.zeros(max(int(sample_count), 0), dtype=np.int64)

    result: list[int] = []
    for segment_index, steps in enumerate(segment_steps):
        try:
            count = max(int(steps), 0)
        except (TypeError, ValueError):
            count = 0
        result.extend([segment_index] * count)

    sample_count = max(int(sample_count), 0)
    if len(result) < sample_count:
        fill_value = len(segment_steps) - 1
        result.extend([fill_value] * (sample_count - len(result)))
    return np.asarray(result[:sample_count], dtype=np.int64)


def set_3d_axes_equal(ax, points: np.ndarray) -> None:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        return
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    centers = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs - mins)) * 0.5, 0.05)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
