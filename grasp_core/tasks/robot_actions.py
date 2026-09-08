#!/usr/bin/env python3
"""任务层：对外提供抓取、回 home、夹爪命令和抓取失败解析的统一服务。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from grasp_core.tasks.grasp_request_ik import publish_latest_request_ik_target
from grasp_core.tasks.ribbon_policy import (
    assume_grasp_success,
    skip_grasp_drop_detection,
)
from grasp_core.tasks.put import execute_fixed_put_after_grasp
from grasp_core.tasks.grasp_drop_detection import GraspDropMonitor, read_grasp_baseline
from grasp_core.communication.gripper_signal import send_gripper_signal
from grasp_core.core.pose_math import (
    PickTemplateWaypoint,
    ik_wrist_orientation_quat,
    quaternion_angle_rad,
)
from grasp_core.communication.request_ik_publisher import (
    RequestIkTargetPublisher,
    publish_home_request_ik_target,
)
from grasp_core.core.robot_target_pose import TargetObjectPose


GRIP_MIN_LIMIT_TOKENS = ("GRASP_FAILED_MIN_LIMIT", "GRIP_FAILED_MIN_LIMIT")
FAILURE_RECOVERY_XYZ = {
    "left": (0.28, 0.23, 0.79),
    "right": (0.28, -0.23, 0.79),
}


@dataclass(frozen=True)
class RobotActionResult:
    """Normalized status from task-level robot actions."""

    status: str
    failed_min_limit: bool = False
    failed_hand: str | None = None
    grasp_confirmed: bool = False
    grasp_hand: str | None = None
    object_label: str | None = None
    ok: bool = False


class RobotActionService:
    """Coordinates grasp/home/gripper commands for one robot setup."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        ik_publisher: RequestIkTargetPublisher | None,
        pick_templates: dict[str, dict[str, list[PickTemplateWaypoint]]],
    ) -> None:
        self.args = args
        self.ik_publisher = ik_publisher
        self.pick_templates = pick_templates

    def publish_grasp(self, targets: list[TargetObjectPose]) -> RobotActionResult:
        selected_target = selected_ik_target(targets, self.args)
        status = publish_latest_request_ik_target(
            self.ik_publisher,
            targets,
            self.pick_templates,
            self.args,
        )
        assumed_success = (
            selected_target is not None
            and assume_grasp_success(selected_target.label)
        )
        non_contact_failure = grip_failed_min_limit(status)
        accepted = not action_failed(status) or (
            assumed_success and non_contact_failure
        )
        return RobotActionResult(
            status=status,
            failed_min_limit=not assumed_success and non_contact_failure,
            failed_hand=grip_failure_hand(status, str(self.args.ik_hand)),
            grasp_confirmed=(assumed_success and accepted) or grip_confirmed(status),
            grasp_hand=(
                grip_success_hand(status)
                or (
                    grip_failure_hand(status, str(self.args.ik_hand))
                    if assumed_success
                    else None
                )
            ),
            object_label=selected_target.label if selected_target is not None else None,
            ok=accepted,
        )

    def publish_put(
        self,
        *,
        grasp_confirmed: bool,
        hand: str | None,
        object_label: str | None = None,
    ) -> RobotActionResult:
        put_hand = hand or ("left" if self.args.ik_hand == "left" else "right")
        drop_detection_enabled = (
            bool(getattr(self.args, "grip_drop_detection", True))
            and not skip_grasp_drop_detection(object_label)
        )
        baseline = (
            read_grasp_baseline(self.args, put_hand)
            if drop_detection_enabled
            else None
        )
        monitor = None
        if baseline is not None and self.ik_publisher is not None:
            monitor = GraspDropMonitor(
                self.args, put_hand, baseline, self.ik_publisher
            )
            monitor.start()
        try:
            result = execute_fixed_put_after_grasp(
                self.ik_publisher,
                put_hand,
                self.args,
                grasp_confirmed=grasp_confirmed,
                object_type=object_label,
                keep_put_pose=bool(getattr(self.args, "put_keep_pose", True)),
            )
        finally:
            if monitor is not None:
                monitor.close()
        status = result.status
        if monitor is not None and monitor.dropped:
            status = (
                f"GRASP_DROPPED hand={put_hand} baseline={baseline} "
                f"pos={monitor.detected_position} target={monitor.threshold}"
            )
        return RobotActionResult(
            status=status,
            grasp_confirmed=grasp_confirmed,
            grasp_hand=put_hand,
            object_label=object_label,
            ok=result.ok and not (monitor is not None and monitor.dropped),
        )

    def publish_home(
        self, hand: str, *, fresh_measured_start: bool = False,
        resume_stop_generation: int | None = None,
        target_xyz: tuple[float, float, float] | None = None,
    ) -> str:
        home_xyz = target_xyz or (
            self.args.left_home_xyz if hand == "left" else self.args.right_home_xyz
        )
        home_position = np.asarray(home_xyz, dtype=np.float64)
        home_orientation = ik_wrist_orientation_quat(self.args, hand=hand)
        start = None
        if resume_stop_generation is not None:
            fresh_measured_start = True
        if fresh_measured_start:
            if not bool(getattr(self.args, "target_smooth_trajectory", True)):
                raise RuntimeError("post-pause HOME requires target_smooth_trajectory")
            if self.ik_publisher is None:
                raise RuntimeError("measured pose publisher unavailable")
            # Automatic recovery keeps publishing stopped throughout settling.
            # A later S invalidates this recovery, even before HOME has started.
            cancelled = self.ik_publisher.stop_requested
            if resume_stop_generation is not None:
                cancelled = lambda: (
                    self.ik_publisher.stop_generation() != resume_stop_generation
                )
            start = self.ik_publisher.client.wait_for_settled_tool_pose(
                hand, cancelled=cancelled,
            )
            position_error_m = float(np.linalg.norm(start[0] - home_position))
            angle_error_deg = float(np.rad2deg(
                quaternion_angle_rad(start[1], home_orientation)
            ))
            position_tolerance_m = max(float(getattr(
                self.args, "interrupted_home_position_tolerance_m", 0.006,
            )), 0.0)
            angle_tolerance_deg = max(float(getattr(
                self.args, "interrupted_home_angle_tolerance_deg", 2.0,
            )), 0.0)
            if resume_stop_generation is not None and not self.ik_publisher.clear_stop(
                expected_generation=resume_stop_generation,
            ):
                raise RuntimeError("HOME cancelled by a newer stop request")
            if self.ik_publisher.stop_requested():
                raise RuntimeError("HOME interrupted by S")
            if (position_error_m <= position_tolerance_m
                    and angle_error_deg <= angle_tolerance_deg):
                self.ik_publisher.synchronize_measured_target(hand, *start)
                status = (
                    f"{hand} already at HOME; no target published "
                    f"(error={position_error_m * 1000.0:.2f}mm/"
                    f"{angle_error_deg:.2f}deg)"
                )
                print(f"[request_ik_tester] {status}", flush=True)
                return status
        return publish_home_request_ik_target(
            self.ik_publisher,
            hand,
            home_xyz,
            self.args,
            start_pose=start,
        )

    def publish_failure_recovery(
        self, hand: str, *, fresh_measured_start: bool = False,
        resume_stop_generation: int | None = None,
    ) -> str:
        """Move a failed grasping arm to its dedicated recovery point."""
        return self.publish_home(
            hand,
            fresh_measured_start=fresh_measured_start,
            resume_stop_generation=resume_stop_generation,
            target_xyz=FAILURE_RECOVERY_XYZ[hand],
        )

    def send_gripper(self, command: str, hand: str | None = None) -> str:
        return send_gripper_signal(command, self.args, hand=hand)


def grip_failed_min_limit(status: object) -> bool:
    """Return True when the gripper reports its minimum-limit failure."""

    text = str(status)
    return any(token in text for token in GRIP_MIN_LIMIT_TOKENS)


def grip_confirmed(status: object) -> bool:
    """Return True only for gripper-confirmed successful grip results."""

    text = str(status)
    return (
        "grasp_confirmed=True" in text
        or ("grip done exit_code=0" in text and not action_failed(text))
    )


def grip_success_hand(status: object) -> str | None:
    text = str(status).lower()
    if "grasp_confirmed=true hand=left" in text or "left gripper grip" in text:
        return "left"
    if "grasp_confirmed=true hand=right" in text or "right gripper grip" in text:
        return "right"
    return None


def action_failed(status: object) -> bool:
    text = str(status)
    return (
        "ERR " in text
        or "failed exit_code=" in text
        or "Failed to send" in text
        or "Invalid " in text
        or grip_failed_min_limit(text)
    )


def grip_failure_hand(status: object, default_hand: str) -> str:
    """Extract the failed hand from a status string, with a safe fallback."""

    text = str(status).lower()
    if "hand=left" in text:
        return "left"
    if "hand=right" in text:
        return "right"
    return "left" if default_hand == "left" else "right"


def selected_ik_target(
    targets: list[TargetObjectPose],
    args: argparse.Namespace,
) -> TargetObjectPose | None:
    if not targets:
        return None
    index = min(max(int(args.ik_target_index), 0), len(targets) - 1)
    return targets[index]
