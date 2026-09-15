#!/usr/bin/env python3
"""任务层：对外提供抓取、回 home、夹爪命令和抓取失败解析的统一服务。"""

from __future__ import annotations

import argparse

from grasp_core.execution.skills.grasp import GraspSkill, execute_grasp
from grasp_core.planning.grasp.policies.ribbon import (
    assume_grasp_success,
    skip_grasp_drop_detection,
)
from grasp_core.execution.skills.place import (
    PlaceSkill,
    execute_fixed_place_after_grasp,
)
from grasp_core.execution.drop_monitor import GraspDropMonitor, read_grasp_baseline
from grasp_core.communication.gripper_signal import send_gripper_signal
from grasp_core.core.math.pose import PickTemplateWaypoint
from grasp_core.execution.motion_executor import (
    RequestIkTargetPublisher,
)
from grasp_core.execution.skills.home import HomeAction
from grasp_core.core.types.robot_target_pose import TargetObjectPose
from grasp_core.execution.result import RobotActionResult
from grasp_core.execution.config import GRIP_MIN_LIMIT_TOKENS

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
        self.home = HomeAction(args=args, ik_publisher=ik_publisher)
        self.grasp = GraspSkill(
            args=args,
            publisher=ik_publisher,
            executor=execute_grasp,
        )
        self.place = PlaceSkill(
            args=args,
            publisher=ik_publisher,
            executor=execute_fixed_place_after_grasp,
        )

    def publish_grasp(self, targets: list[TargetObjectPose]) -> RobotActionResult:
        selected_target = selected_ik_target(targets, self.args)
        status = self.grasp.execute(targets, self.pick_templates)
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

    def publish_place(
        self,
        *,
        grasp_confirmed: bool,
        hand: str | None,
        object_label: str | None = None,
        target_count: int | None = None,
    ) -> RobotActionResult:
        place_hand = hand or ("left" if self.args.ik_hand == "left" else "right")
        drop_detection_enabled = (
            bool(getattr(self.args, "grip_drop_detection", True))
            and not skip_grasp_drop_detection(object_label)
        )
        baseline = (
            read_grasp_baseline(self.args, place_hand)
            if drop_detection_enabled
            else None
        )
        monitor = None
        if baseline is not None and self.ik_publisher is not None:
            monitor = GraspDropMonitor(
                self.args, place_hand, baseline, self.ik_publisher
            )
            monitor.start()
        try:
            result = self.place.execute(
                place_hand,
                grasp_confirmed=grasp_confirmed,
                object_type=object_label,
                target_count=target_count,
            )
        finally:
            if monitor is not None:
                monitor.close()
        status = result.status
        if monitor is not None and monitor.dropped:
            status = (
                f"GRASP_DROPPED hand={place_hand} baseline={baseline} "
                f"pos={monitor.detected_position} target={monitor.threshold}"
            )
        return RobotActionResult(
            status=status,
            grasp_confirmed=grasp_confirmed,
            grasp_hand=place_hand,
            object_label=object_label,
            ok=result.ok and not (monitor is not None and monitor.dropped),
        )

    def publish_home(
        self, hand: str, *, fresh_measured_start: bool = False,
        resume_stop_generation: int | None = None,
    ) -> str:
        return self.home.execute(
            hand,
            fresh_measured_start=fresh_measured_start,
            resume_stop_generation=resume_stop_generation,
        )

    def publish_target_wait_point(self, hand: str) -> str:
        """Move one hand to the randomized target waiting point."""
        return self.home.move_to_target_wait_point(hand)

    def publish_failure_recovery(
        self, hand: str, *, fresh_measured_start: bool = False,
        resume_stop_generation: int | None = None,
    ) -> str:
        """Move a failed grasping arm to its dedicated recovery point."""
        return self.home.recover(
            hand,
            fresh_measured_start=fresh_measured_start,
            resume_stop_generation=resume_stop_generation,
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
