#!/usr/bin/env python3
"""任务层：把最新感知目标转换成 request_ik 抓取路径并触发夹爪动作。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from copy import copy
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from grasp_core.core.types.robot_target_pose import TargetObjectPose, matrix_to_quaternion
from grasp_core.communication.gripper_signal import (
    gripper_receiver_args,
    send_gripper_signal,
)
from grasp_core.planning.grasp.grasp_pose import make_gripper_target_pose
from grasp_core.planning.grasp.policies.ribbon import assume_grasp_success
from grasp_core.planning.trajectory.bezier import smooth_bezier_arc_waypoints
from grasp_core.core.math.pose import (
    PickTemplateWaypoint,
    PoseWaypoint,
    checked_position,
    format_quat,
    format_xyz,
    ik_downward_tilt_deg_for_hand,
    ik_downward_tilt_y_deg_for_hand,
    ik_wrist_orientation_quat,
    log_grasp_pose_plan,
    normalize_quaternion,
    pose_from_position_quaternion,
    select_ik_hand,
    slerp_quaternion,
)
from grasp_core.core.math.easing import smootherstep
from grasp_core.execution.motion_executor import (
    RequestIkTargetPublisher,
    read_current_tool_pose,
    publish_path,
    request_ik_publisher_unavailable_status,
)
from grasp_core.execution.skill_runtime import SkillRuntime
from grasp_core.execution.skills.home import HOME_CONFIG
from grasp_core.execution.config import GRASP_CONFIG, GRIP_MIN_LIMIT_TOKENS, GraspConfig
from grasp_core.tools.trajectory_diagnostics import save_request_ik_grasp_path_artifacts
from grasp_core.planning.grasp.tool_pick_templates import (
    build_pick_template_waypoints,
    pick_template_for_target,
)
GripConfirmedCallback = Callable[[str, str], None]
@dataclass(frozen=True)
class GraspTarget:
    """Resolved grasp target and the selected planning mode."""

    target: TargetObjectPose
    hand: str
    position: np.ndarray
    orientation: tuple[float, float, float, float]
    template: np.ndarray | None
    fallback_reason: str | None
    used_pick_template: bool


@dataclass(frozen=True)
class GraspPlan:
    """Executable phases of one grasp action."""

    approach_waypoints: list[PoseWaypoint]
    final_approach_waypoints: list[PoseWaypoint]
    grip_position: np.ndarray
    grip_orientation: tuple[float, float, float, float]
    final_position: np.ndarray
    final_orientation: tuple[float, float, float, float]
    gripper_pose: np.ndarray
    template: np.ndarray | None
    fallback_reason: str | None
    used_pick_template: bool
    grip_waypoint_index: int | None
    grip_required: bool
    grip_state: float | None


class GraspPlanner:
    """Own target resolution and grasp trajectory construction."""

    def __init__(self, config: GraspConfig | None = None) -> None:
        self.config = config or GRASP_CONFIG

    def controlled_args(self, args: argparse.Namespace) -> argparse.Namespace:
        return self.config.controlled_args(args)

    def resolve_start_pose(
        self,
        publisher: RequestIkTargetPublisher,
        hand: str,
        args: argparse.Namespace,
    ) -> tuple[np.ndarray, tuple[float, float, float, float], str]:
        args = self.controlled_args(args)
        measured_reader = getattr(
            getattr(publisher, "client", None),
            "wait_for_settled_tool_pose",
            None,
        )
        if callable(measured_reader):
            measured_position, measured_orientation = read_current_tool_pose(
                publisher, hand,
            )
            start_source = "measured_tool_pose"
        else:
            remembered_start = publisher.remembered_target(hand)
            if remembered_start is None:
                return (
                    np.asarray(HOME_CONFIG.position(hand), dtype=np.float64),
                    ik_wrist_orientation_quat(args, hand=hand),
                    "home",
                )
            measured_position, measured_orientation = remembered_start
            start_source = "last_published_target"
        return (
            checked_position(measured_position).copy(),
            normalize_quaternion(measured_orientation),
            start_source,
        )

    def build_template_target(
        self,
        target: TargetObjectPose,
        relative_pick_waypoints: list[PickTemplateWaypoint],
        args: argparse.Namespace,
        hand: str,
    ) -> list[PickTemplateWaypoint]:
        args = self.controlled_args(args)
        return build_pick_template_waypoints(
            target,
            relative_pick_waypoints,
            args,
            hand=hand,
        )

    def strip_gripper_states(
        self,
        waypoints: list[PickTemplateWaypoint],
    ) -> list[PoseWaypoint]:
        """Convert template points to pose-only waypoints."""
        return [
            (position, orientation)
            for position, orientation, _gripper_state in waypoints
        ]

    def build_computed_target(
        self,
        target: TargetObjectPose,
        args: argparse.Namespace,
        hand: str,
    ) -> tuple[np.ndarray, np.ndarray | None, str | None]:
        args = self.controlled_args(args)
        return make_gripper_target_pose(target, args, hand=hand)

    def build_waypoints(
        self,
        start_position: np.ndarray,
        start_orientation: tuple[float, float, float, float],
        end_position: np.ndarray,
        end_orientation: tuple[float, float, float, float],
        args: argparse.Namespace,
    ) -> list[PoseWaypoint]:
        args = self.controlled_args(args)
        return smooth_bezier_arc_waypoints(
            start_position,
            start_orientation,
            end_position,
            end_orientation,
            args,
            lift_arc=False,
        )

    def build_final_approach_waypoints(
        self,
        start_position: np.ndarray,
        start_orientation: tuple[float, float, float, float],
        end_position: np.ndarray,
        end_orientation: tuple[float, float, float, float],
    ) -> list[PoseWaypoint]:
        """Build the dense, eased final approach used before gripping."""
        sample_count = max(int(self.config.final_approach_samples), 1)
        slowdown_ratio = float(
            np.clip(self.config.final_approach_slowdown_ratio, 0.0, 1.0)
        )
        slowdown_start = 1.0 - slowdown_ratio
        waypoints: list[PoseWaypoint] = []
        for index in range(1, sample_count + 1):
            progress = index / sample_count
            if slowdown_ratio <= 0.0 or progress <= slowdown_start:
                alpha = progress
            else:
                slowdown_progress = (progress - slowdown_start) / slowdown_ratio
                alpha = slowdown_start + slowdown_ratio * smootherstep(
                    slowdown_progress
                )
            waypoints.append(
                (
                    start_position + (end_position - start_position) * alpha,
                    slerp_quaternion(start_orientation, end_orientation, alpha),
                )
            )
        return waypoints

    def build_template_plan(
        self,
        target: TargetObjectPose,
        relative_pick_waypoints: list[PickTemplateWaypoint],
        start_position: np.ndarray,
        start_orientation: tuple[float, float, float, float],
        args: argparse.Namespace,
        hand: str,
    ) -> GraspPlan:
        pick_waypoints = self.build_template_target(
            target, relative_pick_waypoints, args, hand
        )
        pose_waypoints = self.strip_gripper_states(pick_waypoints)
        grip_index = self.grip_waypoint_index(pick_waypoints)
        grip_state = (
            float(pick_waypoints[grip_index][2])
            if grip_index is not None
            else None
        )
        if grip_index is None:
            final_position, final_orientation = pose_waypoints[-1]
            approach_waypoints = self.build_waypoints(
                start_position,
                start_orientation,
                final_position,
                final_orientation,
                args,
            )
            grip_position, grip_orientation = final_position, final_orientation
            final_approach_waypoints: list[PoseWaypoint] = []
        else:
            grip_position, grip_orientation = pose_waypoints[grip_index]
            if grip_index > 0:
                pregrasp_position, pregrasp_orientation = pose_waypoints[grip_index - 1]
                approach_waypoints = self.build_waypoints(
                    start_position,
                    start_orientation,
                    pregrasp_position,
                    pregrasp_orientation,
                    args,
                )
                final_approach_waypoints = self.build_final_approach_waypoints(
                    pregrasp_position,
                    pregrasp_orientation,
                    grip_position,
                    grip_orientation,
                )
            else:
                approach_waypoints = self.build_waypoints(
                    start_position,
                    start_orientation,
                    grip_position,
                    grip_orientation,
                    args,
                )
                final_approach_waypoints = []
            final_position, final_orientation = grip_position, grip_orientation
        return GraspPlan(
            approach_waypoints=approach_waypoints,
            final_approach_waypoints=final_approach_waypoints,
            grip_position=checked_position(grip_position).copy(),
            grip_orientation=normalize_quaternion(grip_orientation),
            final_position=checked_position(final_position).copy(),
            final_orientation=normalize_quaternion(final_orientation),
            gripper_pose=pose_from_position_quaternion(final_position, final_orientation),
            template=None,
            fallback_reason=None,
            used_pick_template=True,
            grip_waypoint_index=grip_index,
            grip_required=grip_index is not None,
            grip_state=grip_state,
        )

    def build_computed_plan(
        self,
        target: TargetObjectPose,
        start_position: np.ndarray,
        start_orientation: tuple[float, float, float, float],
        args: argparse.Namespace,
        hand: str,
    ) -> GraspPlan:
        planning_args = self.controlled_args(args)
        pregrasp_pose, template, fallback_reason = self.build_computed_target(
            target, planning_args, hand
        )
        grasp_args = copy(planning_args)
        grasp_args.ik_target_stage = "grasp"
        gripper_pose, _grasp_template, _grasp_fallback_reason = (
            self.build_computed_target(target, grasp_args, hand)
        )
        pregrasp_position = pregrasp_pose[:3, 3].copy()
        pregrasp_orientation = matrix_to_quaternion(pregrasp_pose)
        position = gripper_pose[:3, 3].copy()
        orientation = matrix_to_quaternion(gripper_pose)
        final_approach_waypoints: list[PoseWaypoint] = []
        if planning_args.ik_target_stage == "pregrasp":
            final_approach_waypoints = self.build_final_approach_waypoints(
                pregrasp_position,
                pregrasp_orientation,
                position,
                orientation,
            )
        return GraspPlan(
            approach_waypoints=self.build_waypoints(
                start_position,
                start_orientation,
                pregrasp_position,
                pregrasp_orientation,
                planning_args,
            ),
            final_approach_waypoints=final_approach_waypoints,
            grip_position=position.copy(),
            grip_orientation=orientation,
            final_position=position.copy(),
            final_orientation=orientation,
            gripper_pose=gripper_pose,
            template=template,
            fallback_reason=fallback_reason,
            used_pick_template=False,
            grip_waypoint_index=None,
            grip_required=True,
            grip_state=None,
        )

    def grip_waypoint_index(
        self,
        waypoints: list[PickTemplateWaypoint],
    ) -> int | None:
        if not waypoints:
            return None
        candidates = [
            index
            for index, (_position, _orientation, gripper_state) in enumerate(waypoints)
            if float(gripper_state) >= 0.5
        ]
        if candidates:
            return min(candidates, key=lambda index: float(waypoints[index][0][2]))
        fallback_index = min(
            range(len(waypoints)),
            key=lambda index: float(waypoints[index][0][2]),
        )
        position = checked_position(waypoints[fallback_index][0])
        print(
            "[tool_template] WARNING no gripper_state>=0.5 in pick template; "
            "defaulting grip trigger to lowest waypoint "
            f"index={fallback_index} "
            f"xyz=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})",
            flush=True,
        )
        return fallback_index


GRASP_PLANNER = GraspPlanner()


class GripFailedMinLimit(RuntimeError):
    """Raised when the gripper closes to its minimum limit without grasping."""


class GripCommandFailed(RuntimeError):
    """Raised when a gripper command fails before completing normally."""


def execute_grasp(
    publisher: RequestIkTargetPublisher | None,
    targets: list[TargetObjectPose],
    pick_templates: dict[str, dict[str, list[PickTemplateWaypoint]]],
    args: argparse.Namespace,
) -> str:
    if publisher is None:
        status = request_ik_publisher_unavailable_status(args)
        print(f"[request_ik_tester] {status}", flush=True)
        return status
    if not targets:
        status = "No FlowPose target to publish: press F and wait for result"
        print(f"[request_ik_tester] {status}", flush=True)
        return status

    index = min(max(int(args.ik_target_index), 0), len(targets) - 1)
    target = targets[index]
    hand = select_ik_hand(target.base_xyz, args.ik_hand)
    accept_without_contact_check = assume_grasp_success(target.label)
    if accept_without_contact_check:
        print(
            "[grip] ribbon policy active: skipping grasp contact/min-limit "
            f"check for label={target.label!r}",
            flush=True,
        )
    grip_result: dict[str, Any] = {"confirmed": False, "hand": None, "status": ""}
    planner = GRASP_PLANNER
    # Grasp motion values are owned by GraspConfig.  Keep external args only
    # for runtime dependencies such as ROS topics, target selection and paths.
    args = planner.controlled_args(args)
    start_position, start_orientation, start_source = planner.resolve_start_pose(
        publisher,
        hand,
        args,
    )
    tilt_deg = ik_downward_tilt_deg_for_hand(args, hand)
    tilt_y_deg = ik_downward_tilt_y_deg_for_hand(args, hand)
    publisher.begin_joint_trajectory_csv_recording(hand)

    relative_pick_waypoints = pick_template_for_target(target, hand, pick_templates)
    print(
        "[request_ik_tester] grasp request "
        f"label={target.label!r} hand={hand} "
        f"start={start_source} "
        f"tool_template={'hit' if relative_pick_waypoints is not None else 'miss'} "
        f"ik_target_stage={args.ik_target_stage}",
        flush=True,
    )
    grasp_path_artifacts = None
    plan: GraspPlan | None = None
    if relative_pick_waypoints is not None:
        try:
            print(
                "[tool_template] using YAML xyz only; YAML quaternions ignored, "
                "fixed orientation + downward tilt applied "
                f"base_quat={format_quat(start_orientation)} "
                f"tilt={tilt_deg:.2f}deg/"
                f"{args.ik_downward_tilt_axis}+y={tilt_y_deg:.2f}deg/"
                f"{args.ik_downward_tilt_frame}",
                flush=True,
            )
            plan = planner.build_template_plan(
                target,
                relative_pick_waypoints,
                start_position,
                start_orientation,
                args,
                hand,
            )
            if plan.grip_waypoint_index is not None:
                print(
                    "[tool_template] selected grip waypoint "
                    f"index={plan.grip_waypoint_index} "
                    f"xyz=({plan.grip_position[0]:.4f}, "
                    f"{plan.grip_position[1]:.4f}, {plan.grip_position[2]:.4f}) "
                    f"gripper_state={plan.grip_state:.1f}",
                    flush=True,
                )
        except GripFailedMinLimit as exc:
            publisher.finish_joint_trajectory_csv_recording()
            status = f"GRIP_FAILED_MIN_LIMIT hand={hand}: {exc}"
            print(f"[tool_template] {status}", flush=True)
            return status
        except GripCommandFailed as exc:
            publisher.finish_joint_trajectory_csv_recording()
            status = f"GRIP_COMMAND_FAILED hand={hand}: {exc}"
            print(f"[tool_template] {status}", flush=True)
            return status
        except Exception as exc:  # noqa: BLE001
            fallback_reason = f"pick template failed: {exc}"
            print(
                f"[tool_template] {fallback_reason}; using computed target", flush=True
            )
    if plan is None:
        try:
            plan = planner.build_computed_plan(
                target,
                start_position,
                start_orientation,
                args,
                hand,
            )
            print(
                "[computed_grasp] target reached; sending grip command "
                f"hand={hand} xyz=({plan.grip_position[0]:.4f}, "
                f"{plan.grip_position[1]:.4f}, {plan.grip_position[2]:.4f})",
                flush=True,
            )
        except GripFailedMinLimit as exc:
            publisher.finish_joint_trajectory_csv_recording()
            status = f"GRIP_FAILED_MIN_LIMIT hand={hand}: {exc}"
            print(f"[computed_grasp] {status}", flush=True)
            return status

    assert plan is not None
    count = publish_path(
        publisher,
        hand,
        plan.approach_waypoints,
        args,
        start_position_xyz=start_position,
        start_orientation_xyzw=start_orientation,
        final_hold_sec=0.0,
        min_steps=1,
    )
    if plan.final_approach_waypoints:
        # ``publish_smooth_path`` normally chooses one sample per short
        # waypoint segment.  That would erase the extra geometric points
        # above, so force additional controller samples in this phase.  Use
        # the explicit final approach sample count as the density control.
        final_min_steps = max(int(GRASP_PLANNER.config.final_approach_samples), 1)
        count += publish_path(
            publisher,
            hand,
            plan.final_approach_waypoints,
            args,
            start_position_xyz=plan.approach_waypoints[-1][0],
            start_orientation_xyzw=plan.approach_waypoints[-1][1],
            final_hold_sec=0.0,
            min_steps=final_min_steps,
        )
    if plan.grip_required:
        try:
            count += execute_grip_at_pose(
                publisher,
                hand,
                plan.grip_position,
                plan.grip_orientation,
                args,
                assume_success=accept_without_contact_check,
                on_grip_confirmed=lambda grip_hand, grip_status: grip_result.update(
                    confirmed=True,
                    hand=grip_hand,
                    status=grip_status,
                ),
            )
        except GripFailedMinLimit as exc:
            publisher.finish_joint_trajectory_csv_recording()
            status = f"GRIP_FAILED_MIN_LIMIT hand={hand}: {exc}"
            print(f"[grasp] {status}", flush=True)
            return status
        except GripCommandFailed as exc:
            publisher.finish_joint_trajectory_csv_recording()
            status = f"GRIP_COMMAND_FAILED hand={hand}: {exc}"
            print(f"[grasp] {status}", flush=True)
            return status

    position = plan.final_position
    orientation = plan.final_orientation
    gripper_pose = plan.gripper_pose
    template = plan.template
    fallback_reason = plan.fallback_reason
    used_pick_template = plan.used_pick_template

    resolved_target = GraspTarget(
        target=target,
        hand=hand,
        position=checked_position(position).copy(),
        orientation=normalize_quaternion(orientation),
        template=template,
        fallback_reason=fallback_reason,
        used_pick_template=used_pick_template,
    )
    log_grasp_pose_plan(
        resolved_target.target,
        gripper_pose,
        resolved_target.template,
        resolved_target.fallback_reason,
    )
    qx, qy, qz, qw = orientation
    topic = args.left_target_topic if hand == "left" else args.right_target_topic
    status = (
        f"Published {hand} request_ik_tester target "
        f"xyz=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f})m"
        f"{' pick-template' if used_pick_template else ''}"
        f"{' fallback' if fallback_reason else ''}"
    )
    if bool(grip_result["confirmed"]):
        status += f" | grasp_confirmed=True hand={grip_result['hand']}"
    print(
        "[request_ik_tester] sent "
        f"{target.frame_id}_{args.ik_target_stage}: hand={hand} topic={topic} "
        f"frame={args.ik_frame_id} count={count} "
        f"position=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) m "
        f"orientation_xyzw=({qx:.5f}, {qy:.5f}, {qz:.5f}, {qw:.5f}) "
        f"pick_template={used_pick_template} "
        f"template_raw_pose={used_pick_template} "
        f"use_flowpose_rotation={bool(args.use_flowpose_grasp_rotation)} "
        f"tcp_offset={format_xyz(np.asarray(args.ik_grasp_tcp_offset_m, dtype=np.float64))} "
        f"tilt={tilt_deg:.2f}deg/"
        f"{args.ik_downward_tilt_axis}+y={tilt_y_deg:.2f}deg/"
        f"{args.ik_downward_tilt_frame}",
        flush=True,
    )
    artifacts = grasp_path_artifacts or save_request_ik_grasp_path_artifacts(
        publisher, target, hand, args
    )
    if artifacts is not None:
        status += f" | grasp_path_csv={artifacts.csv_path}"
        if artifacts.plot_path is not None:
            status += f" | grasp_path_plot={artifacts.plot_path}"
    joint_csv_path = publisher.finish_joint_trajectory_csv_recording()
    if joint_csv_path is not None:
        status += f" | joint_trajectory_csv={joint_csv_path}"
    return status


def execute_grip_at_pose(
    publisher: RequestIkTargetPublisher,
    hand: str,
    position: np.ndarray,
    orientation: tuple[float, float, float, float],
    args: argparse.Namespace,
    *,
    assume_success: bool = False,
    on_grip_confirmed: GripConfirmedCallback | None = None,
) -> int:
    grasp_config = GRASP_PLANNER.config
    settle_sec = max(float(grasp_config.grip_settle_sec), 0.0)
    pre_grip_hold_sec = settle_sec
    post_confirm_hold_sec = max(float(grasp_config.grip_post_confirm_hold_sec), 0.0)
    print(
        "[grip] holding target before close "
        f"hand={hand} pre_hold={pre_grip_hold_sec:.2f}s "
        f"post_confirm_hold={post_confirm_hold_sec:.2f}s",
        flush=True,
    )
    count = 0
    count += publisher.hold_target(hand, position, orientation, pre_grip_hold_sec)
    endpoints = gripper_receiver_args(args, hand=hand)
    endpoint_text = ", ".join(
        f"{label}:127.0.0.1:{endpoint_args.grip_signal_port}->{endpoint_args.gripper_server}"
        for label, endpoint_args in endpoints
    )
    print(f"[grip] sending close command hand={hand} endpoint={endpoint_text}", flush=True)
    grip_started_at = time.monotonic()
    print("\033[94m[time] grip_start\033[0m", flush=True)
    gripper_status = send_gripper_signal("grip", args, hand=hand)
    print(
        "\033[94m"
        f"[time] grip_start -> grip_return elapsed={time.monotonic() - grip_started_at:.4f}s"
        "\033[0m",
        flush=True,
    )
    print(f"[grip] close command result hand={hand}: {gripper_status}", flush=True)
    if (
        not assume_success
        and any(
            token in gripper_status
            for token in getattr(args, "grip_min_limit_tokens", GRIP_MIN_LIMIT_TOKENS)
        )
    ):
        raise GripFailedMinLimit(gripper_status)
    if "ERR " in gripper_status or "failed exit_code=" in gripper_status:
        raise GripCommandFailed(gripper_status)
    if on_grip_confirmed is not None:
        on_grip_confirmed(hand, gripper_status)
    return count + publisher.hold_target(
        hand,
        position,
        orientation,
        post_confirm_hold_sec,
    )


class GraspSkill(SkillRuntime):
    """Object-oriented entry point for the existing grasp workflow."""

    def __init__(self, *, args: argparse.Namespace, publisher, executor=None) -> None:
        super().__init__(args=args, publisher=publisher)
        self._executor = executor or execute_grasp

    def execute(
        self,
        targets: list[TargetObjectPose],
        pick_templates: dict[str, dict[str, list[PickTemplateWaypoint]]],
    ) -> str:
        return self._executor(self.publisher, targets, pick_templates, self.args)
