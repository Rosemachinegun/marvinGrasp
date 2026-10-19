#!/usr/bin/env python3
"""Move a grasping tool to the voice-requested human handover pose."""

from __future__ import annotations

import argparse
import time
from threading import Event

import numpy as np

from grasp_core.communication.gripper_signal import send_gripper_signal
from grasp_core.core.math.hand import normalize_hand
from grasp_core.core.math.pose import PoseWaypoint
from grasp_core.execution.config import HOME_CONFIG
from grasp_core.execution.motion_executor import (
    RequestIkTargetPublisher,
    publish_home,
    publish_path,
    read_current_tool_pose,
    request_ik_publisher_unavailable_status,
)
from grasp_core.execution.result import SkillResult
from grasp_core.execution.stop_control import publisher_stop_requested


LEFT_HANDOVER_XYZ = (0.65, 0.253, 0.87)
POST_RELEASE_HOLD_SEC = 0.5
HANDOVER_CONFIRM_POLL_SEC = 0.1


def _release_failed(status: str) -> bool:
    return any(
        token in status
        for token in (
            "Invalid gripper command",
            "Failed to send",
            "Gripper command failed",
            "ERR ",
            "failed exit_code=",
        )
    )


def handover_xyz_for_hand(hand: str) -> tuple[float, float, float]:
    """Mirror the requested left-tool pose across the robot's Y=0 plane."""
    x, left_y, z = LEFT_HANDOVER_XYZ
    return (x, left_y, z) if normalize_hand(hand) == "left" else (x, -left_y, z)


def execute_handover_after_grasp(
    publisher: RequestIkTargetPublisher | None,
    hand: str,
    args: argparse.Namespace,
    *,
    grasp_confirmed: bool,
    release_requested: Event,
    ready_event: Event,
) -> SkillResult:
    """Move to handover, wait for operator confirmation, release, then HOME."""
    hand = normalize_hand(hand)
    if not bool(grasp_confirmed):
        status = f"{hand} handover blocked: grasp is not confirmed"
        return SkillResult(False, status)
    if publisher is None:
        return SkillResult(False, request_ik_publisher_unavailable_status(args))
    if publisher_stop_requested(publisher):
        return SkillResult(False, f"STOPPED before {hand} handover")

    start_position, start_orientation = read_current_tool_pose(publisher, hand)
    target_position = np.asarray(handover_xyz_for_hand(hand), dtype=np.float64)
    # Preserve the grasp orientation and take a small safe lift before moving
    # laterally. The final waypoint is the exact requested tool coordinate.
    safe_z = max(float(start_position[2]) + 0.05, float(target_position[2]) + 0.05)
    lift_position = np.asarray(start_position, dtype=np.float64).copy()
    lift_position[2] = safe_z
    approach_position = target_position.copy()
    approach_position[2] = safe_z
    waypoints: list[PoseWaypoint] = [
        (lift_position, start_orientation),
        (approach_position, start_orientation),
        (target_position, start_orientation),
    ]
    count = publish_path(
        publisher,
        hand,
        waypoints,
        args,
        start_position_xyz=start_position,
        start_orientation_xyzw=start_orientation,
        min_steps=1,
        final_hold_sec=0.0,
        terminal_slowdown=True,
        direct_bezier=True,
    )
    if publisher_stop_requested(publisher):
        return SkillResult(False, f"STOPPED during {hand} handover")

    ready_event.set()
    print(
        f"[handover] {hand} reached handover pose; press B to release",
        flush=True,
    )
    while not release_requested.wait(timeout=HANDOVER_CONFIRM_POLL_SEC):
        if publisher_stop_requested(publisher):
            return SkillResult(False, f"STOPPED while {hand} awaited B release")

    release_status = send_gripper_signal("release", args, hand=hand)
    if _release_failed(release_status):
        status = f"{hand} handover release failed: {release_status}"
        print(f"[handover] {status}", flush=True)
        return SkillResult(False, status)

    time.sleep(POST_RELEASE_HOLD_SEC)
    if publisher_stop_requested(publisher):
        status = f"{hand} handover released; STOPPED before HOME"
        print(f"[handover] {status}", flush=True)
        return SkillResult(False, status)

    home_status = publish_home(
        publisher,
        hand,
        HOME_CONFIG.position(hand),
        args,
        final_hold_sec=max(HOME_CONFIG.hold_sec, 0.0),
    )
    status = (
        f"{hand} handover complete; "
        f"xyz=({target_position[0]:.3f},{target_position[1]:.3f},"
        f"{target_position[2]:.3f})m release_confirmed_by=B; "
        f"release={release_status}; post_release_hold={POST_RELEASE_HOLD_SEC:.1f}s; "
        f"{home_status}; waypoints={len(waypoints)} count={count}"
    )
    print(f"[handover] {status}", flush=True)
    return SkillResult(True, status)
