"""Optional recording helpers for published Cartesian trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from grasp_core.core.math.pose import PoseWaypoint, checked_position, normalize_quaternion
from grasp_core.tools.trajectory_csv import (
    make_joint_trajectory_csv_path,
    write_timestamped_pose_samples_joint_trajectory_csv,
)

PublishedTrajectory = dict[str, object]


class TrajectoryRecorder:
    """Collect optional diagnostic samples and export merged joint CSV data."""

    def __init__(
        self,
        *,
        frame_id: str,
        enabled: bool,
        save_joint_csv: bool,
        joint_csv_dir: str | Path,
        joint_name_for_hand: Callable[[str], str],
        source_topic_for_hand: Callable[[str], str | None],
        log_info: Callable[[str], None],
    ) -> None:
        self.frame_id = frame_id
        self.enabled = bool(enabled)
        self.save_joint_csv = bool(save_joint_csv)
        self.joint_csv_dir = Path(joint_csv_dir).expanduser()
        self._joint_name_for_hand = joint_name_for_hand
        self._source_topic_for_hand = source_topic_for_hand
        self._log_info = log_info
        self._last_trajectory: PublishedTrajectory | None = None
        self._csv_active = False
        self._csv_rows: list[dict[str, object]] = []
        self._csv_time_s = 0.0
        self._csv_default_hand: str | None = None

    def begin_csv(self, hand: str | None = None) -> None:
        if not self.save_joint_csv:
            return
        self._csv_active = True
        self._csv_rows = []
        self._csv_time_s = 0.0
        self._csv_default_hand = hand

    def finish_csv(self) -> Path | None:
        if not self._csv_active:
            return None
        self._csv_active = False
        rows = self._csv_rows
        self._csv_rows = []
        self._csv_time_s = 0.0
        if not self.save_joint_csv or not rows:
            return None
        hand = str(rows[0].get("hand") or self._csv_default_hand or "")
        joint_name = str(rows[0].get("joint_name") or self._joint_name_for_hand(hand))
        csv_path = make_joint_trajectory_csv_path(
            self.joint_csv_dir,
            hand=hand,
            joint_name=joint_name,
            point_count=len(rows),
        )
        saved_path = write_timestamped_pose_samples_joint_trajectory_csv(csv_path, rows)
        self._log_info(f"saved merged joint trajectory CSV: {saved_path} points={len(rows)}")
        return saved_path

    def save_samples(
        self,
        hand: str,
        samples: list[PoseWaypoint],
        sample_period_sec: float,
    ) -> None:
        if not self.save_joint_csv or not samples:
            return
        if not self._csv_active:
            self.begin_csv(hand)
        joint_name = self._joint_name_for_hand(hand)
        source_topic = self._source_topic_for_hand(hand) or ""
        period_sec = max(float(sample_period_sec), 1e-4)
        for position_xyz, orientation_xyzw in samples:
            self._csv_time_s += period_sec
            position = checked_position(position_xyz)
            qx, qy, qz, qw = normalize_quaternion(orientation_xyzw)
            self._csv_rows.append({
                "frame_index": len(self._csv_rows),
                "time_from_start_s": f"{self._csv_time_s:.9f}",
                "hand": hand,
                "joint_name": joint_name,
                "frame_id": self.frame_id,
                "source_topic": source_topic,
                "x_m": f"{float(position[0]):.9f}",
                "y_m": f"{float(position[1]):.9f}",
                "z_m": f"{float(position[2]):.9f}",
                "qx": f"{qx:.9f}",
                "qy": f"{qy:.9f}",
                "qz": f"{qz:.9f}",
                "qw": f"{qw:.9f}",
            })

    def start_trajectory(
        self,
        hand: str,
        raw_waypoints: list[PoseWaypoint],
        segment_steps: list[int],
        publish_rate_hz: float,
    ) -> None:
        if not self.enabled:
            self._last_trajectory = None
            return
        self._last_trajectory = {
            "hand": hand,
            "raw_waypoints": raw_waypoints,
            "segment_steps": segment_steps,
            "samples": [],
            "publish_rate_hz": publish_rate_hz,
        }

    def append_sample(
        self,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
    ) -> None:
        if self._last_trajectory is None:
            return
        samples = self._last_trajectory.get("samples")
        if isinstance(samples, list):
            samples.append((checked_position(position_xyz).copy(), normalize_quaternion(orientation_xyzw)))

    def last_trajectory(self) -> PublishedTrajectory | None:
        return self._last_trajectory
