#!/usr/bin/env python3
"""任务层：抓取成功后的固定区域放置动作。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from grasp_core.communication.gripper_signal import send_gripper_signal
from grasp_core.communication.request_ik_publisher import (
    RequestIkTargetPublisher,
    publish_home_request_ik_target,
    publish_request_ik_path,
)
from grasp_core.core.pose_math import (
    PoseWaypoint,
    checked_position,
    ik_wrist_orientation_quat,
    normalize_object_type,
    quaternion_to_rotation_matrix,
    rotation_matrix_from_zyx_euler_deg,
    slerp_quaternion,
)
from grasp_core.core.robot_target_pose import matrix_to_quaternion
from grasp_core.motion.trajectory import effective_trajectory_step_limits

FIXED_PUT_RIGHT_XYZ = (0.45, -0.34, 0.826)
FIXED_PUT_LEFT_XYZ = (0.45, 0.34, 0.826)
# FIXED_PUT_RIGHT_XYZ = (0.54, -0.30, 0.776)
FIXED_PUT_OBJECT_RIGHT_XYZ = {
    "yellow_cube": (0.40, -0.40, 0.86),
    "yellow_duck": (0.37, -0.33, 0.80),
    "blue_cube": (0.35, -0.35, 0.83),
}


@dataclass(frozen=True)
class FixedPutResult:
    ok: bool
    status: str


def humanlike_put_waypoints(
    publisher: RequestIkTargetPublisher,
    hand: str,
    target_position: np.ndarray,
    target_orientation: tuple[float, float, float, float],
    args: argparse.Namespace,
) -> list[PoseWaypoint]:
    """Sample one continuous cubic Bezier arc from grasp to put.

    The two control points are vertically above the endpoints, so the arm lifts
    out of the grasp, travels across one smooth arch, and approaches the put
    point downward.  Dense samples describe the curve to the motion layer; they
    are not dwell points and are published as one trajectory.
    """

    end_position = checked_position(target_position)
    remembered = publisher.remembered_target(hand)
    if remembered is None:
        return [(end_position.copy(), target_orientation)]

    start_position, start_orientation = remembered
    place_orientation = put_outward_z_axis_orientation(hand, target_orientation)
    return smooth_bezier_arc_waypoints(
        start_position,
        start_orientation,
        end_position,
        place_orientation,
        args,
    )


def smooth_bezier_arc_waypoints(
    start_position: np.ndarray,
    start_orientation: tuple[float, float, float, float],
    end_position: np.ndarray,
    end_orientation: tuple[float, float, float, float],
    args: argparse.Namespace,
    *,
    lift_arc: bool = True,
) -> list[PoseWaypoint]:
    """Return constant-distance samples of one smooth cubic Bezier path."""

    start_position = checked_position(start_position)
    end_position = checked_position(end_position)
    distance_m = float(np.linalg.norm(end_position - start_position))
    if distance_m < 1e-4:
        return [(end_position.copy(), end_orientation)]

    if lift_arc:
        max_endpoint_z = max(float(start_position[2]), float(end_position[2]))
        lift_m = min(max(0.22 * distance_m, 0.07), 0.16)
        safe_z_m = max(
            float(getattr(args, "home_safe_z_m", 0.90)),
            max_endpoint_z + 0.04,
        )
        arc_apex_z = max(max_endpoint_z + lift_m, safe_z_m)
        control_1 = start_position.copy()
        control_2 = end_position.copy()
        control_z = (
            arc_apex_z
            - 0.125 * (float(start_position[2]) + float(end_position[2]))
        ) / 0.75
        control_1[2] = control_z
        control_2[2] = control_z
    else:
        # Grasp approach: travel laterally while still high, then round into a
        # near-vertical final approach above the object. This avoids the unsafe
        # diagonal sweep that can push an object sideways, without rising above
        # the current Home/start height.
        delta = end_position - start_position
        cruise_z = max(float(start_position[2]), float(end_position[2]))
        control_1 = start_position.copy()
        control_1[:2] += delta[:2] / 3.0
        control_1[2] = cruise_z
        control_2 = end_position.copy()
        control_2[2] = cruise_z

    # Uniform Bezier parameter values do not produce uniform Cartesian speed.
    # Build a dense lookup table and invert cumulative arc length so consecutive
    # commands are approximately equidistant at the configured publish rate.
    max_step_m, _max_step_deg = effective_trajectory_step_limits(args)
    lookup_alpha = np.linspace(0.0, 1.0, 401)
    lookup_positions = np.asarray([
        cubic_bezier_position(
            start_position, control_1, control_2, end_position, alpha,
        )
        for alpha in lookup_alpha
    ])
    cumulative_length = np.concatenate((
        np.zeros(1, dtype=np.float64),
        np.cumsum(np.linalg.norm(np.diff(lookup_positions, axis=0), axis=1)),
    ))
    curve_length_m = float(cumulative_length[-1])
    sample_count = max(int(np.ceil(curve_length_m / max_step_m)), 2)
    sample_distances = np.linspace(0.0, curve_length_m, sample_count + 1)[1:]
    sample_alphas = np.interp(sample_distances, cumulative_length, lookup_alpha)
    waypoints: list[PoseWaypoint] = []
    for alpha_raw in sample_alphas:
        alpha = float(alpha_raw)
        curve_position = cubic_bezier_position(
            start_position,
            control_1,
            control_2,
            end_position,
            alpha,
        )
        curve_orientation = slerp_quaternion(
            start_orientation,
            end_orientation,
            alpha,
        )
        waypoints.append((curve_position, curve_orientation))
    return waypoints


def cubic_bezier_position(
    start: np.ndarray,
    control_1: np.ndarray,
    control_2: np.ndarray,
    end: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Evaluate a cubic Bezier position at ``alpha`` in [0, 1]."""

    t = float(np.clip(alpha, 0.0, 1.0))
    one_minus_t = 1.0 - t
    return (
        one_minus_t**3 * checked_position(start)
        + 3.0 * one_minus_t**2 * t * checked_position(control_1)
        + 3.0 * one_minus_t * t**2 * checked_position(control_2)
        + t**3 * checked_position(end)
    )


def put_outward_z_axis_orientation(
    hand: str,
    base_orientation: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    hand_sign = 1.0 if hand == "left" else -1.0
    local_z_yaw = rotation_matrix_from_zyx_euler_deg(yaw_deg=hand_sign * 20.0)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quaternion_to_rotation_matrix(base_orientation) @ local_z_yaw
    return matrix_to_quaternion(pose)


def fixed_put_xyz_for_hand(
    hand: str,
    object_type: str | None = None,
) -> tuple[float, float, float]:
    hand_name = normalize_hand(hand)
    if object_type:
        object_name = normalize_object_type(object_type)
        if object_name in FIXED_PUT_OBJECT_RIGHT_XYZ:
            return mirror_right_xyz_for_hand(
                FIXED_PUT_OBJECT_RIGHT_XYZ[object_name],
                hand_name,
            )

    if hand_name == "left":
        return FIXED_PUT_LEFT_XYZ
    return FIXED_PUT_RIGHT_XYZ


def normalize_hand(hand: str) -> str:
    return "left" if str(hand).strip().lower() == "left" else "right"


def mirror_right_xyz_for_hand(
    right_xyz: tuple[float, float, float],
    hand: str,
) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in right_xyz)
    if hand == "left":
        return x, abs(y), z
    return x, -abs(y), z


def publisher_stop_requested(publisher: RequestIkTargetPublisher) -> bool:
    stop_requested = getattr(publisher, "stop_requested", None)
    if callable(stop_requested):
        return bool(stop_requested())

    client = getattr(publisher, "client", None)
    ok = getattr(client, "ok", None)
    if callable(ok):
        return not bool(ok())

    return False


def execute_fixed_put_after_grasp(
    publisher: RequestIkTargetPublisher | None,
    hand: str,
    args: argparse.Namespace,
    *,
    grasp_confirmed: bool,
    object_type: str | None = None,
    keep_put_pose: bool = False,
) -> FixedPutResult:
    """Place and release, optionally keeping the released put pose.

    The caller must pass grasp_confirmed=True from a completed gripper grip result.
    Without that explicit confirmation this function refuses to publish any put target.
    When keep_put_pose=True, no Home target is published and the publisher's
    remembered target remains the put pose for the next grasp.
    """
    hand = normalize_hand(hand)
    if not bool(grasp_confirmed):
        status = f"{hand} put blocked: gripper has not confirmed a successful grasp"
        print(f"[put] {status}", flush=True)
        return FixedPutResult(False, status)

    if publisher is None:
        status = "request_ik_tester publisher unavailable; check ROS2 sourcing"
        print(f"[put] {status}", flush=True)
        return FixedPutResult(False, status)

    if publisher_stop_requested(publisher):
        status = f"STOPPED by B before {hand} put target publishing"
        print(f"[put] {status}", flush=True)
        return FixedPutResult(False, status)

    put_target_hold_sec = max(float(getattr(args, "put_target_hold_sec", 0.05)), 0.0)
    put_home_hold_sec = max(float(getattr(args, "put_home_hold_sec", 0.05)), 0.0)
    position = np.asarray(fixed_put_xyz_for_hand(hand, object_type), dtype=np.float64)
    orientation = ik_wrist_orientation_quat(args, hand=hand)
    waypoints = humanlike_put_waypoints(publisher, hand, position, orientation, args)
    count = publish_request_ik_path(
        publisher,
        hand,
        waypoints,
        args,
        min_steps=1,
        final_hold_sec=put_target_hold_sec,
        terminal_slowdown=True,
    )
    if publisher_stop_requested(publisher):
        status = f"STOPPED by B during {hand} put target publishing"
        print(f"[put] {status}", flush=True)
        return FixedPutResult(False, status)

    print(
        "[put] fixed put target reached "
        f"hand={hand} object={normalize_object_type(object_type) if object_type else 'default'} "
        f"xyz=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f})m "
        f"waypoints={len(waypoints)} count={count} hold={put_target_hold_sec:.2f}s",
        flush=True,
    )

    release_status = send_gripper_signal("release", args, hand=hand)
    if "ERR " in release_status or "failed exit_code=" in release_status:
        status = f"{hand} put release failed: {release_status}"
        print(f"[put] {status}", flush=True)
        return FixedPutResult(False, status)
    else:
        status = f"{hand} put release ok"
        print(f"[put] {status}", flush=True)

    if bool(keep_put_pose):
        keep_status = f"{hand} kept at put pose; next grasp starts here"
        print(f"[put] {keep_status}", flush=True)
        return FixedPutResult(True, f"{status}; {keep_status}")

    if publisher_stop_requested(publisher):
        stop_status = f"STOPPED by B before {hand} home target publishing"
        print(f"[put] {stop_status}", flush=True)
        return FixedPutResult(False, f"{status}; {stop_status}")

    home_status = publish_home_request_ik_target(
        publisher,
        hand,
        position_for_home(hand, args),
        args,
        final_hold_sec=put_home_hold_sec,
    )
    return FixedPutResult(True, f"{status}; {home_status}")


def publish_fixed_put_after_grasp(
    publisher: RequestIkTargetPublisher | None,
    hand: str,
    args: argparse.Namespace,
    *,
    grasp_confirmed: bool = False,
    object_type: str | None = None,
    keep_put_pose: bool = False,
) -> bool:
    """Boolean one-call wrapper for fixed put.

    True means the put target was reached and release succeeded. With
    keep_put_pose=True, Home is deliberately skipped; otherwise Home must also succeed.
    """
    return execute_fixed_put_after_grasp(
        publisher,
        hand,
        args,
        grasp_confirmed=grasp_confirmed,
        object_type=object_type,
        keep_put_pose=keep_put_pose,
    ).ok


def position_for_home(hand: str, args: argparse.Namespace) -> tuple[float, float, float]:
    if normalize_hand(hand) == "left":
        return args.left_home_xyz
    return args.right_home_xyz
