#!/usr/bin/env python3
"""通信执行层：面向 request_ik_tester 发布单点、路径和 home 目标。

轨迹插值来自 motion 层，ROS2 PoseStamped 发送来自 ros2_pose_publisher。
本模块保留轨迹记录/绘图，方便调试实际发布出去的 waypoint。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from grasp_core.communication.ros2_pose_publisher import Ros2PoseTargetPublisher
from grasp_core.motion.trajectory import (
    effective_trajectory_step_limits,
    plan_pose_path,
    plan_pose_target,
)
from grasp_core.core.robot_target_pose import TargetObjectPose
from grasp_core.config.request_ik_config import (
    DEFAULT_JOINT_TRAJECTORY_CSV_DIR,
    DEFAULT_TRAJECTORY_PLOT_DIR,
)
from grasp_core.communication.joint_trajectory_csv import (
    make_joint_trajectory_csv_path,
    write_timestamped_pose_samples_joint_trajectory_csv,
)
from grasp_core.core.pose_math import (
    PoseWaypoint,
    checked_position,
    ik_downward_tilt_deg_for_hand,
    ik_downward_tilt_y_deg_for_hand,
    ik_wrist_orientation_quat,
    normalize_object_type,
    normalize_quaternion,
    quaternion_angle_rad,
)

PublishedTrajectory = dict[str, object]
WaypointCallback = Callable[
    ["RequestIkTargetPublisher", str, np.ndarray, tuple[float, float, float, float]],
    int,
]


@dataclass(frozen=True)
class GraspPathArtifacts:
    """Saved grasp path CSV and plot files."""

    csv_path: Path
    plot_path: Path | None


class RequestIkTargetPublisher:
    """Publishes request_ik_tester target poses or timestamped Cartesian trajectories."""

    def __init__(
        self,
        *,
        left_topic: str,
        right_topic: str,
        frame_id: str,
        publish_rate_hz: float,
        publish_sec: float,
        record_trajectories: bool,
        record_joint_trajectory_csv: bool = False,
        joint_trajectory_csv_dir: str | Path = DEFAULT_JOINT_TRAJECTORY_CSV_DIR,
        command_mode: str = "auto",
        left_trajectory_topic: str | None = None,
        right_trajectory_topic: str | None = None,
        left_trajectory_joint_name: str = "left_tcp",
        right_trajectory_joint_name: str = "right_tcp",
    ) -> None:
        self.frame_id = frame_id
        self.publish_rate_hz = max(float(publish_rate_hz), 0.1)
        self.publish_sec = max(float(publish_sec), 0.0)
        self.record_trajectories = bool(record_trajectories)
        self.record_joint_trajectory_csv = bool(record_joint_trajectory_csv)
        self.joint_trajectory_csv_dir = Path(joint_trajectory_csv_dir).expanduser()
        self.left_topic = left_topic
        self.right_topic = right_topic
        self.command_mode = str(command_mode)
        self.left_trajectory_topic = left_trajectory_topic
        self.right_trajectory_topic = right_trajectory_topic
        self.left_trajectory_joint_name = str(left_trajectory_joint_name)
        self.right_trajectory_joint_name = str(right_trajectory_joint_name)
        self._last_targets: dict[
            str, tuple[np.ndarray, tuple[float, float, float, float]]
        ] = {}
        self._last_published_trajectory: PublishedTrajectory | None = None
        self._joint_trajectory_csv_batch_active = False
        self._joint_trajectory_csv_rows: list[dict[str, object]] = []
        self._joint_trajectory_csv_time_s = 0.0
        self._joint_trajectory_csv_default_hand: str | None = None
        self._stop_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._stop_generation = 0

        self.client = Ros2PoseTargetPublisher(
            left_topic=left_topic,
            right_topic=right_topic,
            left_trajectory_topic=left_trajectory_topic,
            right_trajectory_topic=right_trajectory_topic,
            frame_id=frame_id,
        )
        self.node = self.client.node

    def close(self) -> None:
        self.finish_joint_trajectory_csv_recording()
        self.client.close()

    def request_stop(self) -> int:
        with self._stop_lock:
            self._stop_generation += 1
            generation = self._stop_generation
            self._stop_event.set()
        self.node.get_logger().warning("request_ik target publishing stop requested")
        return generation

    def clear_stop(self, *, expected_generation: int | None = None) -> bool:
        with self._stop_lock:
            if (expected_generation is not None
                    and expected_generation != self._stop_generation):
                return False
            self._stop_event.clear()
            return True

    def stop_generation(self) -> int:
        with self._stop_lock:
            return self._stop_generation

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def begin_joint_trajectory_csv_recording(self, hand: str | None = None) -> None:
        if not self.record_joint_trajectory_csv:
            return
        self._joint_trajectory_csv_batch_active = True
        self._joint_trajectory_csv_rows = []
        self._joint_trajectory_csv_time_s = 0.0
        self._joint_trajectory_csv_default_hand = hand

    def finish_joint_trajectory_csv_recording(self) -> Path | None:
        if not self._joint_trajectory_csv_batch_active:
            return None
        self._joint_trajectory_csv_batch_active = False
        rows = self._joint_trajectory_csv_rows
        self._joint_trajectory_csv_rows = []
        self._joint_trajectory_csv_time_s = 0.0
        if not self.record_joint_trajectory_csv or not rows:
            return None
        hand = str(rows[0].get("hand") or self._joint_trajectory_csv_default_hand or "")
        joint_name = str(rows[0].get("joint_name") or self._trajectory_joint_name(hand))
        csv_path = make_joint_trajectory_csv_path(
            self.joint_trajectory_csv_dir,
            hand=hand,
            joint_name=joint_name,
            point_count=len(rows),
        )
        saved_path = write_timestamped_pose_samples_joint_trajectory_csv(csv_path, rows)
        self.node.get_logger().info(
            f"saved merged joint trajectory CSV: {saved_path} points={len(rows)}"
        )
        return saved_path

    def publish_target(
        self,
        hand: str,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
        *,
        hold_sec: float | None = None,
    ) -> int:
        topic = self.client.topic_for_hand(hand)
        period_sec = 1.0 / self.publish_rate_hz
        target_hold_sec = self.publish_sec
        if hold_sec is not None:
            target_hold_sec = max(float(hold_sec), 0.0)
        deadline = time.monotonic() + target_hold_sec
        count = 0
        position = checked_position(position_xyz)
        orientation = normalize_quaternion(orientation_xyzw)
        self._start_trajectory_record(hand, [(position, orientation)], [])
        self._append_trajectory_sample(position, orientation)

        while (
            self.client.ok()
            and not self.stop_requested()
            and time.monotonic() < deadline
        ):
            self.client.publish_pose(hand, position, orientation)
            count += 1
            time.sleep(period_sec)

        if count == 0 and self.client.ok() and not self.stop_requested():
            self.client.publish_pose(hand, position, orientation)
            count = 1

        self._save_joint_trajectory_csv(
            hand,
            [(position, orientation)],
            period_sec,
        )
        self.node.get_logger().info(
            f"published {hand} target on {topic}: "
            f"x={position_xyz[0]:.3f}, y={position_xyz[1]:.3f}, z={position_xyz[2]:.3f}, "
            f"count={count}"
        )
        self._remember_target(hand, position, orientation)
        return count

    def publish_smooth_target(
        self,
        hand: str,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
        *,
        start_position_xyz: np.ndarray | None = None,
        start_orientation_xyzw: tuple[float, float, float, float] | None = None,
        max_step_m: float = 0.01,
        max_step_deg: float = 3.0,
        min_steps: int = 1,
        final_hold_sec: float | None = None,
        terminal_slowdown: bool = False,
        include_start: bool = False,
        startup_slowdown: bool = False,
    ) -> int:
        topic = self.client.topic_for_hand(hand)
        end_position = checked_position(position_xyz)
        end_orientation = normalize_quaternion(orientation_xyzw)
        start = self._last_targets.get(hand)
        if start_position_xyz is not None or start_orientation_xyzw is not None:
            fallback_position, fallback_orientation = (
                start if start is not None else (end_position, end_orientation)
            )
            start_position = checked_position(
                start_position_xyz
                if start_position_xyz is not None
                else fallback_position
            )
            start_orientation = normalize_quaternion(
                start_orientation_xyzw
                if start_orientation_xyzw is not None
                else fallback_orientation
            )
        elif start is not None:
            start_position, start_orientation = start
        else:
            start_position = checked_position(end_position)
            start_orientation = normalize_quaternion(end_orientation)

        distance_m = float(np.linalg.norm(end_position - start_position))
        angle_rad = quaternion_angle_rad(start_orientation, end_orientation)
        trajectory = plan_pose_target(
            start_position,
            start_orientation,
            end_position,
            end_orientation,
            max_step_m=max_step_m,
            max_step_deg=max_step_deg,
            min_steps=min_steps,
        )
        if include_start:
            # Establish the measured IK pose as the first command, before any
            # interpolated motion. Only post-interruption HOME opts into this.
            trajectory.samples.insert(0, (start_position.copy(), start_orientation))
            trajectory.segment_steps[0] += 1
        steps = sum(trajectory.segment_steps)
        period_sec = 1.0 / self.publish_rate_hz
        count = 0
        self._start_trajectory_record(
            hand,
            trajectory.raw_waypoints,
            trajectory.segment_steps,
        )
        if self._uses_trajectory_command(hand):
            count = self._publish_trajectory_samples_and_wait(
                hand,
                trajectory.samples,
                final_position=end_position,
                final_orientation=end_orientation,
                final_hold_sec=final_hold_sec,
                terminal_slowdown=terminal_slowdown,
                startup_slowdown=startup_slowdown,
            )
            self._remember_target(hand, end_position, end_orientation)
            self.node.get_logger().info(
                f"published trajectory {hand} target on {self.client.trajectory_topic_for_hand(hand)}: "
                f"x={end_position[0]:.3f}, y={end_position[1]:.3f}, z={end_position[2]:.3f}, "
                f"waypoints={steps}, count={count}, distance={distance_m:.3f}m, "
                f"angle={np.rad2deg(angle_rad):.1f}deg"
            )
            return count

        sample_periods_sec = trajectory_sample_periods(
            len(trajectory.samples),
            period_sec,
            startup_slowdown=startup_slowdown,
            terminal_slowdown=terminal_slowdown,
        )
        for (position, orientation), sample_period_sec in zip(
            trajectory.samples,
            sample_periods_sec,
        ):
            if not self.client.ok() or self.stop_requested():
                break
            self._append_trajectory_sample(position, orientation)
            self.client.publish_pose(hand, position, orientation)
            count += 1
            time.sleep(sample_period_sec)

        target_final_hold_sec = self.publish_sec
        if final_hold_sec is not None:
            target_final_hold_sec = max(float(final_hold_sec), 0.0)
        hold_deadline = time.monotonic() + target_final_hold_sec
        while (
            self.client.ok()
            and not self.stop_requested()
            and time.monotonic() < hold_deadline
        ):
            self.client.publish_pose(hand, end_position, end_orientation)
            count += 1
            time.sleep(period_sec)

        if count == 0 and self.client.ok() and not self.stop_requested():
            self.client.publish_pose(hand, end_position, end_orientation)
            count = 1

        self._save_joint_trajectory_csv(hand, trajectory.samples, period_sec)
        self._remember_target(hand, end_position, end_orientation)
        self.node.get_logger().info(
            f"published smooth {hand} target on {topic}: "
            f"x={end_position[0]:.3f}, y={end_position[1]:.3f}, z={end_position[2]:.3f}, "
            f"waypoints={steps}, count={count}, distance={distance_m:.3f}m, "
            f"angle={np.rad2deg(angle_rad):.1f}deg"
        )
        return count

    def hold_target(
        self,
        hand: str,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
        duration_sec: float,
    ) -> int:
        duration_sec = max(float(duration_sec), 0.0)
        if duration_sec <= 0.0 or self.stop_requested():
            return 0
        position = checked_position(position_xyz)
        orientation = normalize_quaternion(orientation_xyzw)
        if self._uses_trajectory_command(hand):
            samples = [(position, orientation)]
            count = self.client.publish_pose_trajectory(
                hand,
                samples,
                joint_name=self._trajectory_joint_name(hand),
                sample_period_sec=duration_sec,
            )
            self._save_joint_trajectory_csv(hand, samples, duration_sec)
            time.sleep(duration_sec)
            return count
        return self.client.hold_pose(
            hand,
            position,
            orientation,
            duration_sec=duration_sec,
            publish_rate_hz=self.publish_rate_hz,
        )

    def publish_smooth_path(
        self,
        hand: str,
        waypoints: list[PoseWaypoint],
        *,
        start_position_xyz: np.ndarray | None = None,
        start_orientation_xyzw: tuple[float, float, float, float] | None = None,
        max_step_m: float = 0.01,
        max_step_deg: float = 3.0,
        min_steps: int = 1,
        on_after_waypoint: dict[int, WaypointCallback] | None = None,
        final_hold_sec: float | None = None,
        terminal_slowdown: bool = False,
    ) -> int:
        if not waypoints:
            return 0

        topic = self.client.topic_for_hand(hand)
        period_sec = 1.0 / self.publish_rate_hz
        start = self._last_targets.get(hand)
        first_position, first_orientation = waypoints[0]
        if start_position_xyz is not None or start_orientation_xyzw is not None:
            fallback_position, fallback_orientation = (
                start if start is not None else (first_position, first_orientation)
            )
            current_position = checked_position(
                start_position_xyz
                if start_position_xyz is not None
                else fallback_position
            )
            current_orientation = normalize_quaternion(
                start_orientation_xyzw
                if start_orientation_xyzw is not None
                else fallback_orientation
            )
        elif start is not None:
            current_position, current_orientation = start
        else:
            current_position = checked_position(first_position)
            current_orientation = normalize_quaternion(first_orientation)

        total_count = 0
        completed_path = True
        trajectory = plan_pose_path(
            current_position,
            current_orientation,
            waypoints,
            max_step_m=max_step_m,
            max_step_deg=max_step_deg,
            min_steps=min_steps,
        )
        self._start_trajectory_record(
            hand,
            trajectory.raw_waypoints,
            trajectory.segment_steps,
        )
        if self._uses_trajectory_command(hand):
            sample_offset = 0
            group_start = 0
            for waypoint_index, steps in enumerate(trajectory.segment_steps):
                sample_offset += steps
                is_callback_stop = (
                    on_after_waypoint is not None
                    and waypoint_index in on_after_waypoint
                )
                is_last_segment = waypoint_index == len(trajectory.segment_steps) - 1
                if not is_callback_stop and not is_last_segment:
                    continue

                group_samples = trajectory.samples[group_start:sample_offset]
                self.node.get_logger().info(
                    f"publishing trajectory group {hand}: "
                    f"end_waypoint={waypoint_index} samples={len(group_samples)} "
                    f"callback={is_callback_stop} final={is_last_segment}"
                )
                total_count += self._publish_trajectory_samples_and_wait(
                    hand,
                    group_samples,
                    final_position=trajectory.raw_waypoints[waypoint_index + 1][0],
                    final_orientation=trajectory.raw_waypoints[waypoint_index + 1][1],
                    hold_final=is_last_segment,
                    final_hold_sec=final_hold_sec,
                    terminal_slowdown=terminal_slowdown and is_last_segment,
                )
                if self.stop_requested():
                    completed_path = False
                    break
                current_position, current_orientation = trajectory.raw_waypoints[
                    waypoint_index + 1
                ]
                group_start = sample_offset
                if (
                    not self.stop_requested()
                    and is_callback_stop
                    and on_after_waypoint is not None
                ):
                    total_count += on_after_waypoint[waypoint_index](
                        self,
                        hand,
                        current_position,
                        current_orientation,
                    )

            self._remember_target(hand, current_position, current_orientation)
            self.node.get_logger().info(
                f"published trajectory {hand} pick path on {self.client.trajectory_topic_for_hand(hand)}: "
                f"waypoints={len(waypoints)}, interp_steps={sum(trajectory.segment_steps)}, count={total_count}, "
                f"final=({current_position[0]:.3f}, {current_position[1]:.3f}, {current_position[2]:.3f})"
            )
            return total_count

        sample_offset = 0
        sample_periods_sec = terminal_sample_periods(
            len(trajectory.samples),
            period_sec,
            enabled=terminal_slowdown,
        )
        global_sample_index = 0
        for waypoint_index, steps in enumerate(trajectory.segment_steps):
            segment_samples = trajectory.samples[sample_offset : sample_offset + steps]
            sample_offset += steps
            for position, orientation in segment_samples:
                if not self.client.ok() or self.stop_requested():
                    completed_path = False
                    break
                self._append_trajectory_sample(position, orientation)
                self.client.publish_pose(hand, position, orientation)
                total_count += 1
                time.sleep(sample_periods_sec[global_sample_index])
                global_sample_index += 1
            if not completed_path:
                break
            current_position, current_orientation = trajectory.raw_waypoints[
                waypoint_index + 1
            ]
            if (
                not self.stop_requested()
                and on_after_waypoint is not None
                and waypoint_index in on_after_waypoint
            ):
                total_count += on_after_waypoint[waypoint_index](
                    self,
                    hand,
                    current_position,
                    current_orientation,
                )

        path_final_hold_sec = self.publish_sec
        if final_hold_sec is not None:
            path_final_hold_sec = max(float(final_hold_sec), 0.0)
        hold_deadline = time.monotonic() + path_final_hold_sec
        while (
            self.client.ok()
            and not self.stop_requested()
            and time.monotonic() < hold_deadline
        ):
            self.client.publish_pose(hand, current_position, current_orientation)
            total_count += 1
            time.sleep(period_sec)

        if total_count == 0 and self.client.ok() and not self.stop_requested():
            self.client.publish_pose(hand, current_position, current_orientation)
            total_count = 1

        self._save_joint_trajectory_csv(hand, trajectory.samples, period_sec)
        self._remember_target(hand, current_position, current_orientation)
        self.node.get_logger().info(
            f"published smooth {hand} pick path on {topic}: "
            f"waypoints={len(waypoints)}, interp_steps={sum(trajectory.segment_steps)}, count={total_count}, "
            f"final=({current_position[0]:.3f}, {current_position[1]:.3f}, {current_position[2]:.3f})"
        )
        return total_count

    def _uses_trajectory_command(self, hand: str) -> bool:
        mode = self.command_mode
        if mode == "pose_stream":
            return False
        if mode == "cartesian_trajectory":
            return self.client.trajectory_topic_for_hand(hand) is not None
        if mode == "auto":
            return self.client.trajectory_subscriber_count(hand) > 0
        return False

    def uses_trajectory_command(self, hand: str) -> bool:
        return self._uses_trajectory_command(hand)

    def _trajectory_joint_name(self, hand: str) -> str:
        return (
            self.left_trajectory_joint_name
            if hand == "left"
            else self.right_trajectory_joint_name
        )

    def _publish_trajectory_samples_and_wait(
        self,
        hand: str,
        samples: list[PoseWaypoint],
        *,
        final_position: np.ndarray,
        final_orientation: tuple[float, float, float, float],
        hold_final: bool = True,
        final_hold_sec: float | None = None,
        terminal_slowdown: bool = False,
        startup_slowdown: bool = False,
    ) -> int:
        if not samples:
            return 0
        if self.stop_requested():
            return 0
        period_sec = 1.0 / self.publish_rate_hz
        for position, orientation in samples:
            self._append_trajectory_sample(position, orientation)
        sample_periods_sec = trajectory_sample_periods(
            len(samples),
            period_sec,
            startup_slowdown=startup_slowdown,
            terminal_slowdown=terminal_slowdown,
        )
        count = self.client.publish_pose_trajectory(
            hand,
            samples,
            joint_name=self._trajectory_joint_name(hand),
            sample_period_sec=period_sec,
            sample_periods_sec=sample_periods_sec,
        )
        self._save_joint_trajectory_csv(hand, samples, period_sec)
        time.sleep(sum(sample_periods_sec))
        if self.stop_requested():
            return count
        trajectory_final_hold_sec = self.publish_sec
        if final_hold_sec is not None:
            trajectory_final_hold_sec = max(float(final_hold_sec), 0.0)
        if hold_final and trajectory_final_hold_sec > 0.0:
            count += self.hold_target(
                hand,
                final_position,
                final_orientation,
                trajectory_final_hold_sec,
            )
        return count

    def _save_joint_trajectory_csv(
        self,
        hand: str,
        samples: list[PoseWaypoint],
        sample_period_sec: float,
    ) -> Path | None:
        if not self.record_joint_trajectory_csv or not samples:
            return None
        if not self._joint_trajectory_csv_batch_active:
            self.begin_joint_trajectory_csv_recording(hand)
        self._append_joint_trajectory_csv_rows(hand, samples, sample_period_sec)
        return None

    def _append_joint_trajectory_csv_rows(
        self,
        hand: str,
        samples: list[PoseWaypoint],
        sample_period_sec: float,
    ) -> None:
        joint_name = self._trajectory_joint_name(hand)
        source_topic = self.client.trajectory_topic_for_hand(hand) or ""
        period_sec = max(float(sample_period_sec), 1e-4)
        for position_xyz, orientation_xyzw in samples:
            self._joint_trajectory_csv_time_s += period_sec
            position = checked_position(position_xyz)
            qx, qy, qz, qw = normalize_quaternion(orientation_xyzw)
            self._joint_trajectory_csv_rows.append(
                {
                    "frame_index": len(self._joint_trajectory_csv_rows),
                    "time_from_start_s": f"{self._joint_trajectory_csv_time_s:.9f}",
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
                }
            )

    def _remember_target(
        self,
        hand: str,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
    ) -> None:
        self._last_targets[hand] = (
            checked_position(position_xyz).copy(),
            normalize_quaternion(orientation_xyzw),
        )

    def remembered_target(
        self,
        hand: str,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
        target = self._last_targets.get(hand)
        if target is None:
            return None
        position, orientation = target
        return position.copy(), orientation

    def synchronize_measured_target(
        self,
        hand: str,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
    ) -> None:
        """Replace command history with a freshly measured IK-model pose."""
        self._remember_target(hand, position_xyz, orientation_xyzw)
        self.node.get_logger().info(
            f"synchronized {hand} remembered target from measured FK"
        )

    def last_published_trajectory(self) -> PublishedTrajectory | None:
        return self._last_published_trajectory

    def _start_trajectory_record(
        self,
        hand: str,
        raw_waypoints: list[PoseWaypoint],
        segment_steps: list[int],
    ) -> None:
        if not self.record_trajectories:
            self._last_published_trajectory = None
            return
        self._last_published_trajectory = {
            "hand": hand,
            "raw_waypoints": raw_waypoints,
            "segment_steps": segment_steps,
            "samples": [],
            "publish_rate_hz": self.publish_rate_hz,
        }

    def _append_trajectory_sample(
        self,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
    ) -> None:
        if self._last_published_trajectory is None:
            return
        samples = self._last_published_trajectory.get("samples")
        if isinstance(samples, list):
            samples.append(
                (
                    checked_position(position_xyz).copy(),
                    normalize_quaternion(orientation_xyzw),
                )
            )

def build_ik_target_publisher(
    args: argparse.Namespace,
) -> RequestIkTargetPublisher | None:
    try:
        publisher = RequestIkTargetPublisher(
            left_topic=args.left_target_topic,
            right_topic=args.right_target_topic,
            frame_id=args.ik_frame_id,
            publish_rate_hz=args.target_publish_rate_hz,
            publish_sec=args.target_publish_sec,
            record_trajectories=should_visualize_grasp_path(args),
            record_joint_trajectory_csv=should_save_joint_trajectory_csv(args),
            joint_trajectory_csv_dir=args.joint_trajectory_csv_dir,
            command_mode=args.target_command_mode,
            left_trajectory_topic=args.left_trajectory_topic,
            right_trajectory_topic=args.right_trajectory_topic,
            left_trajectory_joint_name=args.left_trajectory_joint_name,
            right_trajectory_joint_name=args.right_trajectory_joint_name,
        )
    except RuntimeError as exc:
        print(f"[request_ik_tester] disabled: {exc}", flush=True)
        return None
    print(
        "[request_ik_tester] press S to publish targets: "
        f"left={args.left_target_topic}, right={args.right_target_topic}, "
        f"traj_left={args.left_trajectory_topic}, traj_right={args.right_trajectory_topic}, "
        f"mode={args.target_command_mode}, "
        f"frame_id={args.ik_frame_id}, "
        f"smooth={bool(args.target_smooth_trajectory)}, "
        f"step<={args.target_trajectory_step_m:.3f}m/"
        f"{args.target_trajectory_step_deg:.1f}deg, "
        f"speed<={args.target_trajectory_speed_mps:.3f}m/s/"
        f"{args.target_trajectory_angular_speed_dps:.1f}deg/s, "
        f"min_steps={int(args.target_trajectory_min_steps)}, "
        f"visualize_grasp_path={should_visualize_grasp_path(args)}, "
        f"save_joint_trajectory_csv={should_save_joint_trajectory_csv(args)}, "
        f"hold={args.target_publish_sec:.2f}s@{args.target_publish_rate_hz:.1f}Hz",
        flush=True,
    )
    return publisher


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


def save_request_ik_trajectory_plot(
    publisher: RequestIkTargetPublisher,
    target: TargetObjectPose,
    hand: str,
    args: argparse.Namespace,
) -> Path | None:
    """Backward-compatible wrapper for older call sites."""

    return save_request_ik_grasp_path_plot(publisher, target, hand, args)


def should_visualize_grasp_path(args: argparse.Namespace) -> bool:
    """Single switch for grasp path CSV/PNG recording and rendering."""

    value = getattr(args, "visualize_grasp_path", None)
    if value is None:
        value = getattr(args, "target_trajectory_plot", True)
    return bool(value)


def should_save_request_ik_trajectory_plot(args: argparse.Namespace) -> bool:
    """Backward-compatible wrapper; use should_visualize_grasp_path()."""

    return should_visualize_grasp_path(args)


def should_save_joint_trajectory_csv(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "save_joint_trajectory_csv", False))


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


def publish_home_request_ik_target(
    publisher: RequestIkTargetPublisher | None,
    hand: str,
    home_xyz: tuple[float, float, float],
    args: argparse.Namespace,
    *,
    final_hold_sec: float | None = None,
    start_pose: PoseWaypoint | None = None,
) -> str:
    if publisher is None:
        status = "request_ik_tester publisher unavailable; check ROS2 sourcing"
        print(f"[request_ik_tester] {status}", flush=True)
        return status

    position = np.asarray(home_xyz, dtype=np.float64)
    orientation = ik_wrist_orientation_quat(args, hand=hand)
    # Directly publish the single home target (no multi-waypoint home path)
    home_waypoints = [(position, orientation)]
    count = publish_request_ik_target(
        publisher,
        hand,
        position,
        orientation,
        args,
        start_position_xyz=start_pose[0] if start_pose is not None else None,
        start_orientation_xyzw=start_pose[1] if start_pose is not None else None,
        final_hold_sec=final_hold_sec,
        terminal_slowdown=True,
        include_start=start_pose is not None,
        startup_slowdown=start_pose is not None,
    )
    if start_pose is not None and (count <= 0 or publisher.stop_requested()):
        raise RuntimeError("HOME interrupted before completion; measured sync still required")

    topic = args.left_target_topic if hand == "left" else args.right_target_topic
    qx, qy, qz, qw = orientation
    tilt_deg = ik_downward_tilt_deg_for_hand(args, hand)
    tilt_y_deg = ik_downward_tilt_y_deg_for_hand(args, hand)
    status = (
        f"Published {hand} home target "
        f"xyz=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f})m"
    )
    print(
        "[request_ik_tester] sent home: "
        f"hand={hand} topic={topic} frame={args.ik_frame_id} count={count} "
        f"waypoints={len(home_waypoints)} "
        f"position=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) m "
        f"orientation_xyzw=({qx:.5f}, {qy:.5f}, {qz:.5f}, {qw:.5f}) "
        f"tilt={tilt_deg:.2f}deg/{args.ik_downward_tilt_axis}+y={tilt_y_deg:.2f}deg/"
        f"{args.ik_downward_tilt_frame}",
        flush=True,
    )
    return status


def publish_request_ik_target(
    publisher: RequestIkTargetPublisher,
    hand: str,
    position: np.ndarray,
    orientation: tuple[float, float, float, float],
    args: argparse.Namespace,
    *,
    start_position_xyz: np.ndarray | None = None,
    start_orientation_xyzw: tuple[float, float, float, float] | None = None,
    final_hold_sec: float | None = None,
    terminal_slowdown: bool = False,
    include_start: bool = False,
    startup_slowdown: bool = False,
) -> int:
    if not bool(getattr(args, "target_smooth_trajectory", True)):
        return publisher.publish_target(
            hand,
            position,
            orientation,
            hold_sec=final_hold_sec,
        )
    max_step_m, max_step_deg = effective_trajectory_step_limits(args)
    return publisher.publish_smooth_target(
        hand,
        position,
        orientation,
        start_position_xyz=start_position_xyz,
        start_orientation_xyzw=start_orientation_xyzw,
        max_step_m=max_step_m,
        max_step_deg=max_step_deg,
        min_steps=int(getattr(args, "target_trajectory_min_steps", 1)),
        final_hold_sec=final_hold_sec,
        terminal_slowdown=terminal_slowdown,
        include_start=include_start,
        startup_slowdown=startup_slowdown,
    )


def publish_request_ik_path(
    publisher: RequestIkTargetPublisher,
    hand: str,
    waypoints: list[PoseWaypoint],
    args: argparse.Namespace,
    *,
    start_position_xyz: np.ndarray | None = None,
    start_orientation_xyzw: tuple[float, float, float, float] | None = None,
    on_after_waypoint: dict[int, WaypointCallback] | None = None,
    final_hold_sec: float | None = None,
    terminal_slowdown: bool = False,
    min_steps: int | None = None,
) -> int:
    if not bool(getattr(args, "target_smooth_trajectory", True)):
        count = 0
        for waypoint_index, (position, orientation) in enumerate(waypoints):
            if publisher.stop_requested():
                break
            if (
                final_hold_sec is not None
                and waypoint_index == len(waypoints) - 1
                and float(final_hold_sec) != publisher.publish_sec
            ):
                original_publish_sec = publisher.publish_sec
                publisher.publish_sec = max(float(final_hold_sec), 0.0)
                try:
                    count += publisher.publish_target(hand, position, orientation)
                finally:
                    publisher.publish_sec = original_publish_sec
            else:
                count += publisher.publish_target(hand, position, orientation)
            if (
                not publisher.stop_requested()
                and on_after_waypoint is not None
                and waypoint_index in on_after_waypoint
            ):
                count += on_after_waypoint[waypoint_index](
                    publisher,
                    hand,
                    checked_position(position),
                    normalize_quaternion(orientation),
                )
        return count
    max_step_m, max_step_deg = effective_trajectory_step_limits(args)
    return publisher.publish_smooth_path(
        hand,
        waypoints,
        start_position_xyz=start_position_xyz,
        start_orientation_xyzw=start_orientation_xyzw,
        max_step_m=max_step_m,
        max_step_deg=max_step_deg,
        min_steps=(
            int(getattr(args, "target_trajectory_min_steps", 1))
            if min_steps is None
            else max(int(min_steps), 1)
        ),
        on_after_waypoint=on_after_waypoint,
        final_hold_sec=final_hold_sec,
        terminal_slowdown=terminal_slowdown,
    )


def terminal_sample_periods(
    sample_count: int,
    base_period_sec: float,
    *,
    enabled: bool,
    tail_count: int = 8,
    max_scale: float = 3.0,
) -> list[float]:
    count = max(int(sample_count), 0)
    base_period = max(float(base_period_sec), 1e-4)
    periods = [base_period] * count
    if not enabled or count <= 1:
        return periods

    tail = min(max(int(tail_count), 1), count)
    max_scale = max(float(max_scale), 1.0)
    for tail_index in range(tail):
        ratio = float(tail_index + 1) / float(tail)
        scale = 1.0 + (max_scale - 1.0) * ratio * ratio
        periods[count - tail + tail_index] = base_period * scale
    return periods


def trajectory_sample_periods(
    sample_count: int,
    base_period_sec: float,
    *,
    startup_slowdown: bool = False,
    terminal_slowdown: bool = False,
    startup_hold_sec: float = 0.25,
    startup_ramp_count: int = 16,
    startup_max_scale: float = 4.0,
) -> list[float]:
    """Build timing with a stationary handover and gentle initial acceleration.

    Sample zero is the fresh measured pose for post-interruption HOME. Holding
    it briefly lets request_ik settle its target before Cartesian motion begins.
    The following periods then decrease smoothly to the normal period. Spatial
    interpolation, speed limits and quaternion shortest-path remain unchanged.
    """
    periods = terminal_sample_periods(
        sample_count,
        base_period_sec,
        enabled=terminal_slowdown,
    )
    if not startup_slowdown or not periods:
        return periods

    base_period = max(float(base_period_sec), 1e-4)
    periods[0] = max(float(startup_hold_sec), base_period)
    ramp = min(max(int(startup_ramp_count), 0), max(len(periods) - 1, 0))
    max_scale = max(float(startup_max_scale), 1.0)
    for ramp_index in range(ramp):
        ratio = float(ramp_index + 1) / float(max(ramp, 1))
        scale = 1.0 + (max_scale - 1.0) * (1.0 - ratio) ** 2
        sample_index = ramp_index + 1
        periods[sample_index] = max(periods[sample_index], base_period * scale)
    return periods


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
