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
from typing import Callable

import numpy as np

from grasp_core.core.robot_target_pose import TargetObjectPose, matrix_to_quaternion
from grasp_core.communication.gripper_signal import (
    gripper_receiver_args,
    send_gripper_signal,
)
from grasp_core.planning.grasp_pose import make_gripper_target_pose
from grasp_core.tasks.ribbon_policy import assume_grasp_success
from grasp_core.tasks.put import smooth_bezier_arc_waypoints
from grasp_core.core.pose_math import (
    PickTemplateWaypoint,
    PoseWaypoint,
    checked_position,
    format_quat,
    format_xyz,
    home_position_for_hand,
    ik_downward_tilt_deg_for_hand,
    ik_downward_tilt_y_deg_for_hand,
    ik_wrist_orientation_quat,
    log_grasp_pose_plan,
    normalize_quaternion,
    pose_from_position_quaternion,
    select_ik_hand,
)
from grasp_core.config.request_ik_config import DEFAULT_GRIP_SETTLE_SEC
from grasp_core.communication.request_ik_publisher import (
    RequestIkTargetPublisher,
    save_request_ik_grasp_path_artifacts,
    publish_request_ik_path,
)
from grasp_core.planning.tool_pick_templates import (
    build_pick_template_waypoints,
    pick_template_for_target,
)
WaypointCallback = Callable[
    [RequestIkTargetPublisher, str, np.ndarray, tuple[float, float, float, float]],
    int,
]
GripConfirmedCallback = Callable[[str, str], None]
GRIP_MIN_LIMIT_TOKENS = ("GRASP_FAILED_MIN_LIMIT", "GRIP_FAILED_MIN_LIMIT")


class GripFailedMinLimit(RuntimeError):
    """Raised when the gripper closes to its minimum limit without grasping."""


class GripCommandFailed(RuntimeError):
    """Raised when a gripper command fails before completing normally."""


def resolve_grasp_start_pose(
    publisher: RequestIkTargetPublisher,
    hand: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    """Use the last commanded pose, falling back to Home on the first grasp."""
    remembered_start = publisher.remembered_target(hand)
    if remembered_start is None:
        return (
            home_position_for_hand(hand, args),
            ik_wrist_orientation_quat(args, hand=hand),
            "home",
        )

    remembered_position, remembered_orientation = remembered_start
    return (
        checked_position(remembered_position).copy(),
        normalize_quaternion(remembered_orientation),
        "last_published_target",
    )


def publish_latest_request_ik_target(
    publisher: RequestIkTargetPublisher | None,
    targets: list[TargetObjectPose],
    pick_templates: dict[str, dict[str, list[PickTemplateWaypoint]]],
    args: argparse.Namespace,
) -> str:
    if publisher is None:
        status = "request_ik_tester publisher unavailable; check ROS2 sourcing"
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
    start_position, start_orientation, start_source = resolve_grasp_start_pose(
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
    used_pick_template = False
    template: np.ndarray | None = None
    fallback_reason: str | None = None
    grasp_path_artifacts = None
    if relative_pick_waypoints is not None:
        try:
            pick_waypoints = build_pick_template_waypoints(
                target, relative_pick_waypoints, args, hand=hand
            )
            print(
                "[tool_template] using YAML xyz only; YAML quaternions ignored, "
                "fixed orientation + downward tilt applied "
                f"base_quat={format_quat(start_orientation)} "
                f"tilt={tilt_deg:.2f}deg/"
                f"{args.ik_downward_tilt_axis}+y={tilt_y_deg:.2f}deg/"
                f"{args.ik_downward_tilt_frame}",
                flush=True,
            )
            grip_waypoint_index = pick_grip_waypoint_index(pick_waypoints)
            pose_waypoints = strip_gripper_states(pick_waypoints)
            grip_callbacks = make_grip_waypoint_callbacks(
                grip_waypoint_index,
                pick_waypoints,
                args,
                assume_success=accept_without_contact_check,
                on_grip_confirmed=lambda grip_hand, grip_status: grip_result.update(
                    confirmed=True,
                    hand=grip_hand,
                    status=grip_status,
                ),
            )
            if grip_waypoint_index is not None:
                grip_position, _grip_orientation, grip_state = pick_waypoints[
                    grip_waypoint_index
                ]
                print(
                    "[tool_template] selected grip waypoint "
                    f"index={grip_waypoint_index} "
                    f"xyz=({grip_position[0]:.4f}, {grip_position[1]:.4f}, {grip_position[2]:.4f}) "
                    f"gripper_state={float(grip_state):.1f}",
                    flush=True,
                )
            if grip_waypoint_index is None:
                smooth_pick_waypoints = smooth_bezier_arc_waypoints(
                    start_position,
                    start_orientation,
                    pose_waypoints[-1][0],
                    pose_waypoints[-1][1],
                    args,
                    lift_arc=False,
                )
                count = publish_request_ik_path(
                    publisher,
                    hand,
                    smooth_pick_waypoints,
                    args,
                    start_position_xyz=start_position,
                    start_orientation_xyzw=start_orientation,
                    min_steps=1,
                    terminal_slowdown=True,
                )
            else:
                print(
                    "[tool_template] executing pick path in explicit phases: "
                    f"approach_until_grip_index={grip_waypoint_index}, "
                    f"total_waypoints={len(pose_waypoints)}",
                    flush=True,
                )
                grip_position, grip_orientation = pose_waypoints[grip_waypoint_index]
                smooth_approach_waypoints = smooth_bezier_arc_waypoints(
                    start_position,
                    start_orientation,
                    grip_position,
                    grip_orientation,
                    args,
                    lift_arc=False,
                )
                count = publish_request_ik_path(
                    publisher,
                    hand,
                    smooth_approach_waypoints,
                    args,
                    start_position_xyz=start_position,
                    start_orientation_xyzw=start_orientation,
                    final_hold_sec=0.0,
                    terminal_slowdown=True,
                    min_steps=1,
                )
                # Do not render/save trajectory diagnostics on the critical
                # target-arrival -> gripper-close path.  Matplotlib startup and
                # disk I/O can otherwise leave the arm visibly waiting at the
                # object.  Artifacts are saved below after grip/lift completes.
                count += grip_callbacks[grip_waypoint_index](
                    publisher,
                    hand,
                    grip_position,
                    grip_orientation,
                )
                remaining_waypoints = pose_waypoints[grip_waypoint_index + 1 :]
                if remaining_waypoints:
                    lift_hold_sec = max(
                        float(getattr(args, "grip_lift_hold_sec", 0.05)),
                        0.0,
                    )
                    count += publish_request_ik_path(
                        publisher,
                        hand,
                        remaining_waypoints,
                        args,
                        final_hold_sec=lift_hold_sec,
                    )
            position, orientation = pose_waypoints[-1]
            used_pick_template = True
            gripper_pose = pose_from_position_quaternion(position, orientation)
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
    if not used_pick_template:
        try:
            gripper_pose, template, grasp_fallback_reason = make_gripper_target_pose(
                target, args, hand=hand
            )
            fallback_reason = fallback_reason or grasp_fallback_reason
            position = gripper_pose[:3, 3].copy()
            orientation = matrix_to_quaternion(gripper_pose)
            smooth_approach_waypoints = smooth_bezier_arc_waypoints(
                start_position,
                start_orientation,
                position,
                orientation,
                args,
                lift_arc=False,
            )
            count = publish_request_ik_path(
                publisher,
                hand,
                smooth_approach_waypoints,
                args,
                start_position_xyz=start_position,
                start_orientation_xyzw=start_orientation,
                final_hold_sec=0.0,
                terminal_slowdown=True,
                min_steps=1,
            )
            print(
                "[computed_grasp] target reached; sending grip command "
                f"hand={hand} xyz=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})",
                flush=True,
            )
            count += execute_grip_at_pose(
                publisher,
                hand,
                position,
                orientation,
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
            print(f"[computed_grasp] {status}", flush=True)
            return status
        except GripCommandFailed as exc:
            publisher.finish_joint_trajectory_csv_recording()
            status = f"GRIP_COMMAND_FAILED hand={hand}: {exc}"
            print(f"[computed_grasp] {status}", flush=True)
            return status

    log_grasp_pose_plan(target, gripper_pose, template, fallback_reason)
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


def strip_gripper_states(waypoints: list[PickTemplateWaypoint]) -> list[PoseWaypoint]:
    return [(position, orientation) for position, orientation, _ in waypoints]


def pick_grip_waypoint_index(waypoints: list[PickTemplateWaypoint]) -> int | None:
    if not waypoints:
        return None
    candidates = [
        index
        for index, (_position, _orientation, gripper_state) in enumerate(waypoints)
        if float(gripper_state) >= 0.5
    ]
    if not candidates:
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
    return min(candidates, key=lambda index: float(waypoints[index][0][2]))


def make_grip_waypoint_callbacks(
    grip_waypoint_index: int | None,
    waypoints: list[PickTemplateWaypoint],
    args: argparse.Namespace,
    *,
    assume_success: bool = False,
    on_grip_confirmed: GripConfirmedCallback | None = None,
) -> dict[int, WaypointCallback]:
    if grip_waypoint_index is None:
        return {}
    grip_position = checked_position(waypoints[grip_waypoint_index][0])

    def callback(
        publisher: RequestIkTargetPublisher,
        hand: str,
        position: np.ndarray,
        orientation: tuple[float, float, float, float],
    ) -> int:
        print(
            "[tool_template] gripper_state=1.0; lowest pick waypoint reached, sending grip command like L key "
            f"xyz=({grip_position[0]:.4f}, {grip_position[1]:.4f}, {grip_position[2]:.4f})",
            flush=True,
        )
        return execute_grip_at_pose(
            publisher,
            hand,
            position,
            orientation,
            args,
            assume_success=assume_success,
            on_grip_confirmed=on_grip_confirmed,
        )

    return {grip_waypoint_index: callback}


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
    settle_sec = max(
        float(getattr(args, "grip_settle_sec", DEFAULT_GRIP_SETTLE_SEC)), 0.0
    )
    pre_grip_hold_sec = settle_sec
    post_confirm_hold_sec = max(
        float(getattr(args, "grip_post_confirm_hold_sec", 0.0)),
        0.0,
    )
    print(
        "[grip] holding target before close "
        f"hand={hand} pre_hold={pre_grip_hold_sec:.2f}s "
        f"post_confirm_hold={post_confirm_hold_sec:.2f}s",
        flush=True,
    )
    count = 0
    if publisher.uses_trajectory_command(hand):
        count += publisher.client.hold_pose(
            hand,
            checked_position(position),
            normalize_quaternion(orientation),
            duration_sec=pre_grip_hold_sec,
            publish_rate_hz=publisher.publish_rate_hz,
        )
    else:
        count += publisher.hold_target(hand, position, orientation, pre_grip_hold_sec)
    endpoints = gripper_receiver_args(args, hand=hand)
    endpoint_text = ", ".join(
        f"{label}:127.0.0.1:{endpoint_args.grip_signal_port}->{endpoint_args.gripper_server}"
        for label, endpoint_args in endpoints
    )
    print(f"[grip] sending close command hand={hand} endpoint={endpoint_text}", flush=True)
    gripper_status = send_gripper_signal("grip", args, hand=hand)
    print(f"[grip] close command result hand={hand}: {gripper_status}", flush=True)
    if (
        not assume_success
        and any(token in gripper_status for token in GRIP_MIN_LIMIT_TOKENS)
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
