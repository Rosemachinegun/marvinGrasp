#!/usr/bin/env python3
"""Save ROS trajectory frames to CSV with one timestamp per frame."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from grasp_core.core.math.pose import checked_position, normalize_quaternion


DEFAULT_JOINT_TRAJECTORY_CSV_DIR = (
    PROJECT_ROOT / "captures" / "request_ik_joint_trajectories"
)


def make_joint_trajectory_csv_path(
    output_dir: str | Path,
    *,
    hand: str | None = None,
    joint_name: str | None = None,
    point_count: int = 0,
) -> Path:
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parts = [timestamp]
    if hand:
        parts.append(_clean_filename_part(hand))
    if joint_name:
        parts.append(_clean_filename_part(joint_name))
    parts.extend(["joint_trajectory", f"{int(point_count)}pts"])
    return directory / ("_".join(parts) + ".csv")


def write_pose_samples_joint_trajectory_csv(
    csv_path: str | Path,
    samples: Iterable[tuple[np.ndarray, tuple[float, float, float, float]]],
    *,
    hand: str,
    joint_name: str,
    sample_period_sec: float,
    frame_id: str,
    source_topic: str | None = None,
) -> Path:
    """Write MultiDOF-style target samples as a timestamped joint trajectory CSV."""

    rows = []
    period_sec = max(float(sample_period_sec), 1e-4)
    for index, (position_xyz, orientation_xyzw) in enumerate(samples, start=1):
        position = checked_position(position_xyz)
        qx, qy, qz, qw = normalize_quaternion(orientation_xyzw)
        rows.append(
            {
                "frame_index": index - 1,
                "time_from_start_s": f"{index * period_sec:.9f}",
                "hand": hand,
                "joint_name": joint_name,
                "frame_id": frame_id,
                "source_topic": source_topic or "",
                "x_m": f"{float(position[0]):.9f}",
                "y_m": f"{float(position[1]):.9f}",
                "z_m": f"{float(position[2]):.9f}",
                "qx": f"{qx:.9f}",
                "qy": f"{qy:.9f}",
                "qz": f"{qz:.9f}",
                "qw": f"{qw:.9f}",
            }
        )

    path = Path(csv_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "time_from_start_s",
                "hand",
                "joint_name",
                "frame_id",
                "source_topic",
                "x_m",
                "y_m",
                "z_m",
                "qx",
                "qy",
                "qz",
                "qw",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_timestamped_pose_samples_joint_trajectory_csv(
    csv_path: str | Path,
    rows: Iterable[dict[str, object]],
) -> Path:
    """Write already timestamped pose/joint rows to CSV."""

    path = Path(csv_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "time_from_start_s",
                "hand",
                "joint_name",
                "frame_id",
                "source_topic",
                "x_m",
                "y_m",
                "z_m",
                "qx",
                "qy",
                "qz",
                "qw",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_joint_trajectory_msg_csv(csv_path: str | Path, msg, *, source_topic: str) -> Path:
    """Write a trajectory_msgs/JointTrajectory message to one row per point."""

    joint_names = [str(name) for name in getattr(msg, "joint_names", [])]
    rows = []
    for frame_index, point in enumerate(getattr(msg, "points", [])):
        row = {
            "frame_index": frame_index,
            "time_from_start_s": f"{_duration_to_seconds(point.time_from_start):.9f}",
            "source_topic": source_topic,
        }
        for prefix, values in (
            ("position", getattr(point, "positions", [])),
            ("velocity", getattr(point, "velocities", [])),
            ("acceleration", getattr(point, "accelerations", [])),
            ("effort", getattr(point, "effort", [])),
        ):
            for index, value in enumerate(values):
                name = joint_names[index] if index < len(joint_names) else f"joint_{index}"
                row[f"{name}_{prefix}"] = f"{float(value):.9f}"
        rows.append(row)

    fieldnames = ["frame_index", "time_from_start_s", "source_topic"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path = Path(csv_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _duration_to_seconds(duration) -> float:
    return float(getattr(duration, "sec", 0)) + float(
        getattr(duration, "nanosec", 0)
    ) / 1_000_000_000.0


def _clean_filename_part(value: str) -> str:
    text = str(value).strip().replace("/", "_")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Subscribe to a JointTrajectory topic and save each message as CSV."
    )
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_JOINT_TRAJECTORY_CSV_DIR))
    parser.add_argument("--once", action="store_true", help="Exit after the first CSV.")
    args = parser.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from trajectory_msgs.msg import JointTrajectory
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 Python packages are not importable. Source your ROS2 workspace first."
        ) from exc

    rclpy.init(args=None)
    node = Node("joint_trajectory_csv_recorder")
    output_dir = Path(args.output_dir).expanduser()

    def on_message(msg) -> None:
        path = make_joint_trajectory_csv_path(
            output_dir,
            joint_name="joint_state",
            point_count=len(getattr(msg, "points", [])),
        )
        write_joint_trajectory_msg_csv(path, msg, source_topic=args.topic)
        node.get_logger().info(f"saved joint trajectory CSV: {path}")
        if args.once:
            rclpy.shutdown()

    node.create_subscription(JointTrajectory, args.topic, on_message, 10)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
