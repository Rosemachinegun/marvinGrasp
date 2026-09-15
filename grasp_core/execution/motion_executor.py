#!/usr/bin/env python3
"""Execution-layer motion commands: calculate and publish Cartesian targets.

轨迹插值来自 action.motion，ROS2 下发来自 communication.request_ik_transport。
可选轨迹记录由 tools.trajectory_recorder 负责。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import threading
import time
from typing import Callable

import numpy as np

from grasp_core.communication.request_ik_transport import Ros2PoseTargetPublisher
from grasp_core.planning.trajectory.constraints import effective_trajectory_step_limits
from grasp_core.planning.trajectory.planner import plan_trajectory
from grasp_core.planning.trajectory.timing import trajectory_linear_velocities
from grasp_core.planning.trajectory.timing import (
    terminal_sample_periods,
    trajectory_sample_periods,
)
from grasp_core.execution.config import DEFAULT_JOINT_TRAJECTORY_CSV_DIR
from grasp_core.config.trajectory_config import (
    should_save_joint_trajectory_csv,
    should_visualize_grasp_path,
)
from grasp_core.core.math.pose import (
    PoseWaypoint,
    checked_position,
    ik_home_wrist_orientation_quat,
    normalize_quaternion,
    quaternion_angle_rad,
)

from grasp_core.tools.trajectory_recorder import TrajectoryRecorder

PublishedTrajectory = dict[str, object]
WaypointCallback = Callable[
    ["RequestIkTargetPublisher", str, np.ndarray, tuple[float, float, float, float]],
    int,
]


class RequestIkTargetPublisher:
    """Publishes request_ik_tester targets as a PoseStamped stream."""

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
    ) -> None:
        self.frame_id = frame_id
        self.publish_rate_hz = max(float(publish_rate_hz), 0.1)
        self.publish_sec = max(float(publish_sec), 0.0)
        self.record_trajectories = bool(record_trajectories)
        self.record_joint_trajectory_csv = bool(record_joint_trajectory_csv)
        self.joint_trajectory_csv_dir = Path(joint_trajectory_csv_dir).expanduser()
        self.left_topic = left_topic
        self.right_topic = right_topic
        self._last_targets: dict[
            str, tuple[np.ndarray, tuple[float, float, float, float]]
        ] = {}
        self._stop_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._stop_generation = 0

        self.client = Ros2PoseTargetPublisher(
            left_topic=left_topic,
            right_topic=right_topic,
            frame_id=frame_id,
        )
        self.node = self.client.node
        self.recorder = TrajectoryRecorder(
            frame_id=self.frame_id,
            enabled=self.record_trajectories,
            save_joint_csv=self.record_joint_trajectory_csv,
            joint_csv_dir=self.joint_trajectory_csv_dir,
            joint_name_for_hand=lambda hand: f"{hand}_tcp",
            source_topic_for_hand=self.client.topic_for_hand,
            log_info=self.node.get_logger().info,
        )

    def close(self) -> None:
        self.finish_joint_trajectory_csv_recording()
        self.client.close()

    def _get_recorder(self) -> TrajectoryRecorder:
        """Create a recorder lazily for lightweight test publishers."""
        recorder = getattr(self, "recorder", None)
        if recorder is None:
            recorder = TrajectoryRecorder(
                frame_id=str(getattr(self, "frame_id", "base_link")),
                enabled=bool(getattr(self, "record_trajectories", False)),
                save_joint_csv=bool(getattr(self, "record_joint_trajectory_csv", False)),
                joint_csv_dir=getattr(
                    self, "joint_trajectory_csv_dir", DEFAULT_JOINT_TRAJECTORY_CSV_DIR
                ),
                joint_name_for_hand=lambda hand: f"{hand}_tcp",
                source_topic_for_hand=self.client.topic_for_hand,
                log_info=self.node.get_logger().info,
            )
            self.recorder = recorder
        return recorder

    def request_stop(self, reason: str | None = None) -> int:
        with self._stop_lock:
            self._stop_generation += 1
            generation = self._stop_generation
            self._stop_event.set()
        suffix = f" reason={reason}" if reason else ""
        self.node.get_logger().warning(
            f"request_ik target publishing stop requested{suffix}"
        )
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
        self._get_recorder().begin_csv(hand)

    def finish_joint_trajectory_csv_recording(self) -> Path | None:
        return self._get_recorder().finish_csv()

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
        startup_slowdown: bool = False,
    ) -> int:
        if not waypoints:
            return 0

        topic = self.client.topic_for_hand(hand)
        period_sec = 1.0 / self.publish_rate_hz
        if (start_position_xyz is None) != (start_orientation_xyzw is None):
            raise ValueError(
                "trajectory start position and orientation must be provided together"
            )
        if start_position_xyz is None:
            current_position, current_orientation = read_current_tool_pose(self, hand)
        else:
            current_position = checked_position(start_position_xyz)
            current_orientation = normalize_quaternion(start_orientation_xyzw)

        total_count = 0
        completed_path = True
        last_published_pose: PoseWaypoint | None = None
        trajectory = plan_trajectory(
            current_position,
            current_orientation,
            waypoints,
            max_step_m=max_step_m,
            max_step_deg=max_step_deg,
            min_steps=min_steps,
        )
        place_timing = getattr(self, "_place_timing", None)
        if place_timing is not None:
            place_timing["plan_done"] = time.monotonic()
            print(
                "\033[94m"
                f"[time] place planner detail waypoints={len(trajectory.raw_waypoints)} "
                f"samples={len(trajectory.samples)} "
                f"max_segment_steps={max(trajectory.segment_steps, default=0)}\033[0m",
                flush=True,
            )
        start_position_gap_m = float(
            np.linalg.norm(trajectory.samples[0][0] - current_position)
        )
        start_angle_gap_deg = float(
            np.rad2deg(quaternion_angle_rad(trajectory.samples[0][1], current_orientation))
        )
        recorder = self._get_recorder()
        place_timing = getattr(self, "_place_timing", None)
        recorder.start_trajectory(
            hand,
            trajectory.raw_waypoints,
            trajectory.segment_steps,
            self.publish_rate_hz,
        )
        first_publish_index = 0
        planned_samples = trajectory.samples[first_publish_index:]
        sample_periods_sec = trajectory_sample_periods(
            len(planned_samples),
            period_sec,
            startup_slowdown=startup_slowdown,
            terminal_slowdown=terminal_slowdown,
        )
        sent_samples: list[PoseWaypoint] = []
        sample_offset = 0
        trajectory_sample_index = 0
        next_publish_at: float | None = None
        for waypoint_index, steps in enumerate(trajectory.segment_steps):
            segment_samples = trajectory.samples[sample_offset : sample_offset + steps]
            sample_offset += steps
            for position, orientation in segment_samples:
                if trajectory_sample_index < first_publish_index:
                    trajectory_sample_index += 1
                    continue
                if not self.client.ok() or self.stop_requested():
                    completed_path = False
                    break
                if next_publish_at is not None:
                    next_publish_at += sample_periods_sec[len(sent_samples) - 1]
                    time.sleep(0.005)
                    if not self.client.ok() or self.stop_requested():
                        completed_path = False
                        break
                recorder.append_sample(position, orientation)
                if place_timing is not None and place_timing["first_pose"] is None:
                    place_timing["first_pose"] = time.monotonic()
                self.client.publish_pose(hand, position, orientation)
                last_published_pose = (position.copy(), orientation)
                sent_samples.append(last_published_pose)
                total_count += 1
                if next_publish_at is None:
                    next_publish_at = time.monotonic()
                trajectory_sample_index += 1
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
                if waypoint_index + 1 < len(trajectory.raw_waypoints) - 1:
                    measured_position, measured_orientation = read_current_tool_pose(
                        self, hand,
                    )
                    planned_position, planned_orientation = trajectory.raw_waypoints[
                        waypoint_index + 1
                    ]
                    position_gap_m = float(
                        np.linalg.norm(measured_position - planned_position)
                    )
                    angle_gap_deg = float(
                        np.rad2deg(
                            quaternion_angle_rad(
                                measured_orientation, planned_orientation,
                            )
                        )
                    )
                    self.node.get_logger().info(
                        f"trajectory boundary {hand}: "
                        f"position_gap={position_gap_m * 1000.0:.2f}mm "
                        f"angle_gap={angle_gap_deg:.2f}deg"
                    )
                    if position_gap_m > 0.002 or angle_gap_deg > 1.0:
                        current_position, current_orientation = (
                            measured_position,
                            measured_orientation,
                        )
                        completed_path = False
                        self.node.get_logger().warning(
                            f"trajectory {hand} stopped: boundary start gap exceeds "
                            f"limit ({position_gap_m * 1000.0:.2f}mm/"
                            f"{angle_gap_deg:.2f}deg)"
                        )
                        break
                # A callback can block for gripper activity. Start the next
                # segment from a fresh clock instead of trying to catch up.
                next_publish_at = None

        path_final_hold_sec = self.publish_sec
        if final_hold_sec is not None:
            path_final_hold_sec = max(float(final_hold_sec), 0.0)
        total_count += self.hold_target(
            hand,
            current_position,
            current_orientation,
            path_final_hold_sec,
        )

        if total_count == 0 and self.client.ok() and not self.stop_requested():
            self.client.publish_pose(hand, current_position, current_orientation)
            last_published_pose = (current_position.copy(), current_orientation)
            sent_samples.append(last_published_pose)
            total_count = 1

        self._get_recorder().save_samples(hand, sent_samples, period_sec)
        if last_published_pose is not None:
            self._remember_target(hand, *last_published_pose)
        self.node.get_logger().info(
            f"published smooth {hand} pick path on {topic}: "
            f"waypoints={len(waypoints)}, interp_steps={sum(trajectory.segment_steps)}, "
            f"published={len(sent_samples)}, count={total_count}, completed={completed_path}, "
            f"start_gap={start_position_gap_m * 1000.0:.2f}mm/"
            f"{start_angle_gap_deg:.2f}deg, "
            f"last={None if last_published_pose is None else last_published_pose[0].tolist()}"
        )
        return total_count

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
        return self._get_recorder().last_trajectory()

def build_ik_target_publisher(
    args: argparse.Namespace,
) -> RequestIkTargetPublisher | None:
    setattr(args, "request_ik_publisher_unavailable_reason", None)
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
        )
    except RuntimeError as exc:
        setattr(args, "request_ik_publisher_unavailable_reason", str(exc))
        print(f"[request_ik_tester] disabled: {exc}", flush=True)
        return None
    print(
        "[request_ik_tester] press S to publish targets: "
        f"left={args.left_target_topic}, right={args.right_target_topic}, "
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


def request_ik_publisher_unavailable_status(args: argparse.Namespace) -> str:
    status = "request_ik_tester publisher unavailable; check ROS2 sourcing"
    reason = getattr(args, "request_ik_publisher_unavailable_reason", None)
    if reason:
        status = f"{status}: {reason}"
    return status


def publish_home(
    publisher: RequestIkTargetPublisher | None,
    hand: str,
    home_xyz: tuple[float, float, float],
    args: argparse.Namespace,
    *,
    start_pose: PoseWaypoint | None = None,
    final_hold_sec: float | None = 0.0,
    orientation_xyzw: tuple[float, float, float, float] | None = None,
) -> str:
    if publisher is None:
        status = request_ik_publisher_unavailable_status(args)
        print(f"[request_ik_tester] {status}", flush=True)
        return status

    position = np.asarray(home_xyz, dtype=np.float64)
    orientation = (
        normalize_quaternion(orientation_xyzw)
        if orientation_xyzw is not None
        else ik_home_wrist_orientation_quat(args, hand=hand)
    )
    count = publish_path(
        publisher,
        hand,
        [(position, orientation)],
        args,
        start_position_xyz=None if start_pose is None else start_pose[0],
        start_orientation_xyzw=None if start_pose is None else start_pose[1],
        final_hold_sec=final_hold_sec,
    )
    if start_pose is not None and (count <= 0 or publisher.stop_requested()):
        raise RuntimeError("HOME interrupted before completion; measured sync still required")

    topic = args.left_target_topic if hand == "left" else args.right_target_topic
    qx, qy, qz, qw = orientation
    hand_name = "left" if str(hand).strip().lower() == "left" else "right"
    tilt_z_deg = float(getattr(args, f"home_tilt_z_{hand_name}_deg", 0.0))
    tilt_y_deg = float(getattr(args, f"home_tilt_y_{hand_name}_deg", 0.0))
    status = (
        f"Published {hand} home target "
        f"xyz=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f})m"
    )
    print(
        "[request_ik_tester] sent home: "
        f"hand={hand} topic={topic} frame={args.ik_frame_id} count={count} "
        "waypoints=1 "
        f"position=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) m "
        f"orientation_xyzw=({qx:.5f}, {qy:.5f}, {qz:.5f}, {qw:.5f}) "
        f"home_tilt=z={tilt_z_deg:.2f}deg/y={tilt_y_deg:.2f}deg/"
        f"{args.ik_downward_tilt_frame}",
        flush=True,
    )
    return status


def publish_path(
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
    startup_slowdown: bool = False,
) -> int:
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
        startup_slowdown=startup_slowdown,
    )


def read_current_tool_pose(
    publisher: RequestIkTargetPublisher,
    hand: str,
    *,
    timeout_sec: float = 5.0,
) -> PoseWaypoint:
    """Read the newest valid hardware pose before planning a new motion."""

    stop_requested = getattr(publisher, "stop_requested", None)
    measured_reader = getattr(
        getattr(publisher, "client", None),
        "wait_for_current_tool_pose",
        None,
    )
    if not callable(stop_requested) or not callable(measured_reader):
        remembered = publisher.remembered_target(hand)
        if remembered is None:
            raise RuntimeError(f"current pose reader unavailable for {hand}")
        return remembered
    if stop_requested():
        raise RuntimeError(f"motion start cancelled for {hand}")
    started_at = time.monotonic()
    pose = measured_reader(
        hand,
        timeout_sec=max(float(timeout_sec), 0.1),
        cancelled=stop_requested,
    )
    elapsed_sec = time.monotonic() - started_at
    print(
        "\033[91m"
        f"read_current_tool_pose hand={hand} "
        f"elapsed={elapsed_sec:.4f}s source=measured"
        "\033[0m",
        flush=True,
    )
    return pose
