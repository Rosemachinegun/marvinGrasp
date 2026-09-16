#!/usr/bin/env python3
"""任务层：抓取成功后的固定区域放置动作。"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
import numpy as np

from grasp_core.communication.gripper_signal import send_gripper_signal
from grasp_core.execution.motion_executor import (
    RequestIkTargetPublisher,
    read_current_tool_pose,
    publish_home,
    publish_path,
    request_ik_publisher_unavailable_status,
)
from grasp_core.execution.skill_runtime import SkillRuntime
from grasp_core.execution.skills.home import publish_target_wait_point
from grasp_core.execution.stop_control import publisher_stop_requested
from grasp_core.execution.config import HOME_CONFIG, PlaceConfig, PLACE_CONFIG
from grasp_core.execution.result import FixedPlaceResult
from grasp_core.core.math.hand import mirror_right_xyz_for_hand, normalize_hand
from grasp_core.core.math.pose import (
    PoseWaypoint,
    checked_position,
    ik_orientation_rotation,
    normalize_object_type,
    rotation_matrix_from_xyz_euler_rad,
    slerp_quaternion,
)
from grasp_core.core.types.robot_target_pose import matrix_to_quaternion
from grasp_core.planning.trajectory.interpolation import cubic_bezier_position
from grasp_core.planning.trajectory.bezier import (
    smooth_bezier_arc_waypoints,
)


# Compatibility export for older diagnostic/test callers.  Place itself uses
# the direct Bézier planner path and does not call this low-level helper.
@dataclass(frozen=True)
class PlaceTarget:
    """Resolved target used by one Place trajectory."""

    nominal_position: np.ndarray
    position: np.ndarray
    orientation: tuple[float, float, float, float]
    lift_height_m: float


class PlacePlanner:
    """Own Place target resolution and trajectory construction."""

    def __init__(self, config: PlaceConfig) -> None:
        self.config = config

    def resolve_target(
        self,
        args: argparse.Namespace,
        hand: str,
        target_position: np.ndarray | tuple[float, float, float],
    ) -> PlaceTarget:
        nominal_position = checked_position(target_position)
        target_jitter = np.asarray(
            tuple(
                random.uniform(-abs(float(limit)), abs(float(limit)))
                for limit in self.config.position_jitter_xyz_m
            ),
            dtype=np.float64,
        )
        end_position = checked_position(nominal_position + target_jitter)
        hand_name = normalize_hand(hand)
        orientation_xyz_rad = list(
            self.config.left_orientation_xyz_rad
            if hand_name == "left"
            else self.config.right_orientation_xyz_rad
        )
        orientation_xyz_rad[1] += random.uniform(
            -abs(float(self.config.pitch_jitter_rad)),
            abs(float(self.config.pitch_jitter_rad)),
        )
        orientation = self.orientation_for_hand(
            args,
            hand,
            orientation_xyz_rad=tuple(orientation_xyz_rad),
        )
        lift_height = max(
            0.0,
            self.config.lift_height_m
            + random.uniform(
                -abs(float(self.config.lift_height_jitter_m)),
                abs(float(self.config.lift_height_jitter_m)),
            ),
        )
        return PlaceTarget(
            nominal_position=nominal_position.copy(),
            position=end_position,
            orientation=orientation,
            lift_height_m=lift_height,
        )

    def orientation_for_hand(
        self,
        args: argparse.Namespace,
        hand: str,
        *,
        orientation_xyz_rad: tuple[float, float, float] | None = None,
    ) -> tuple[float, float, float, float]:
        hand_name = normalize_hand(hand)
        if orientation_xyz_rad is None:
            orientation_xyz_rad = (
                self.config.left_orientation_xyz_rad
                if hand_name == "left"
                else self.config.right_orientation_xyz_rad
            )
        place_rotation = rotation_matrix_from_xyz_euler_rad(orientation_xyz_rad)
        pose = np.eye(4, dtype=np.float64)
        base_rotation = ik_orientation_rotation(args)
        if getattr(args, "ik_downward_tilt_frame", "local") == "base":
            pose[:3, :3] = place_rotation @ base_rotation
        else:
            pose[:3, :3] = base_rotation @ place_rotation
        return matrix_to_quaternion(pose)

    def position_for_hand(
        self,
        hand: str,
        object_type: str | None = None,
    ) -> tuple[float, float, float]:
        hand_name = normalize_hand(hand)
        if object_type:
            object_name = normalize_object_type(object_type)
            object_targets = self.config.object_targets()
            if object_name in object_targets:
                return mirror_right_xyz_for_hand(
                    object_targets[object_name],
                    hand_name,
                )
        return self.config.left_xyz if hand_name == "left" else self.config.right_xyz

    def build_waypoints(
        self,
        publisher: RequestIkTargetPublisher,
        hand: str,
        target: PlaceTarget,
        args: argparse.Namespace,
        *,
        start_pose: PoseWaypoint | None = None,
    ) -> list[PoseWaypoint]:
        end_position = checked_position(target.position)
        if start_pose is None:
            start_pose = publisher.remembered_target(hand)
        if start_pose is None:
            return [(end_position.copy(), target.orientation)]

        start_position, start_orientation = start_pose
        delta = end_position - start_position
        horizontal_delta = delta[:2]
        horizontal_distance = float(np.linalg.norm(horizontal_delta))
        if horizontal_distance > 1e-6:
            approach_distance = min(
                max(float(self.config.approach_distance_m), 0.0),
                horizontal_distance * 0.35,
            )
            approach_position = end_position.copy()
            approach_position[:2] -= (
                horizontal_delta / horizontal_distance * approach_distance
            )
        else:
            approach_position = end_position.copy()

        # Keep only semantic waypoints.  Bézier control points are not
        # exposed as robot targets because doing so creates visible speed
        # changes at every control-point boundary.
        lift_position = start_position.copy()
        lift_position[2] = max(
            float(start_position[2]) + max(float(target.lift_height_m), 0.0),
            float(self.config.safe_z_m),
        )
        approach_position = approach_position.copy()
        approach_position[2] = lift_position[2]

        approach_orientation = slerp_quaternion(
            start_orientation,
            target.orientation,
            float(np.clip(self.config.approach_orientation_ratio, 0.0, 1.0)),
        )
        # Keep the path sparse.  The trajectory planner receives the complete
        # semantic chain and computes one global tangent/interpolation field;
        # Place does not pre-sample each segment into extra robot waypoints.
        return [
            (lift_position, start_orientation),
            (approach_position, approach_orientation),
            (end_position.copy(), target.orientation),
        ]


PLACE_PLANNER = PlacePlanner(PLACE_CONFIG)


# Legacy aliases retained for old diagnostics while the runtime uses
# ``PlacePlanner`` directly.
def fixed_place_xyz_for_hand(
    hand: str,
    object_type: str | None = None,
) -> tuple[float, float, float]:
    return PLACE_PLANNER.position_for_hand(hand, object_type)


def place_orientation_for_hand(
    args: argparse.Namespace,
    hand: str,
) -> tuple[float, float, float, float]:
    return PLACE_PLANNER.orientation_for_hand(args, hand)


def humanlike_place_waypoints(
    publisher: RequestIkTargetPublisher,
    hand: str,
    end_position: np.ndarray,
    end_orientation: tuple[float, float, float, float],
    args: argparse.Namespace,
) -> list[PoseWaypoint]:
    """Legacy dense direct-Bézier helper for pre-PlacePlanner callers."""
    start_position, start_orientation = publisher.remembered_target(hand)
    end_position = checked_position(end_position)
    lift_position = start_position.copy()
    lift_position[2] = max(
        float(start_position[2]) + PLACE_CONFIG.lift_height_m,
        PLACE_CONFIG.safe_z_m,
    )
    approach_position = end_position.copy()
    approach_position[2] = lift_position[2]
    controls = (start_position, lift_position, approach_position, end_position)
    samples: list[PoseWaypoint] = []
    for index in range(1, 41):
        alpha = index / 40.0
        samples.append(
            (
                cubic_bezier_position(*controls, alpha),
                slerp_quaternion(start_orientation, end_orientation, alpha),
            )
        )
    return samples


def execute_fixed_place_after_grasp(
    publisher: RequestIkTargetPublisher | None,
    hand: str,
    args: argparse.Namespace,
    *,
    grasp_confirmed: bool,
    object_type: str | None = None,
    keep_place_pose: bool | None = None,
    target_count: int | None = None,
) -> FixedPlaceResult:
    """Place and release, optionally keeping the released place pose.

    The caller must pass grasp_confirmed=True from a completed gripper grip result.
    Without that explicit confirmation this function refuses to publish a place target.
    When keep_place_pose=True, no Home target is published and the publisher's
    remembered target remains the place pose for the next grasp.
    """
    hand = normalize_hand(hand)
    if not bool(grasp_confirmed):
        status = f"{hand} place blocked: gripper has not confirmed a successful grasp"
        print(f"[place] {status}", flush=True)
        return FixedPlaceResult(False, status)

    if publisher is None:
        status = request_ik_publisher_unavailable_status(args)
        print(f"[place] {status}", flush=True)
        return FixedPlaceResult(False, status)

    if publisher_stop_requested(publisher):
        status = f"STOPPED by B before {hand} place target publishing"
        print(f"[place] {status}", flush=True)
        return FixedPlaceResult(False, status)

    config = PLACE_CONFIG
    place_target_hold_sec = max(config.target_hold_sec, 0.0)
    # Start the timing window before FK and waypoint construction so the
    # complete place startup path is visible in the diagnostic output.
    timing = {
        "place_start": time.monotonic(),
        "tcp_done": None,
        "waypoints_done": None,
        "plan_done": None,
        "first_pose": None,
    }
    setattr(publisher, "_place_timing", timing)
    nominal_position = np.asarray(
        PLACE_PLANNER.position_for_hand(hand, object_type),
        dtype=np.float64,
    )
    place_target = PLACE_PLANNER.resolve_target(
        args,
        hand,
        nominal_position,
    )
    start_pose = read_current_tool_pose(publisher, hand)
    timing["tcp_done"] = time.monotonic()
    waypoints = PLACE_PLANNER.build_waypoints(
        publisher,
        hand,
        place_target,
        args,
        start_pose=start_pose,
    )
    timing["waypoints_done"] = time.monotonic()
    count = publish_path(
        publisher,
        hand,
        waypoints,
        args,
        start_position_xyz=start_pose[0],
        start_orientation_xyzw=start_pose[1],
        min_steps=1,
        final_hold_sec=place_target_hold_sec,
        terminal_slowdown=False,
        final_slowdown_ratio=config.final_slowdown_ratio,
        direct_bezier=True,
    )
    if timing["first_pose"] is not None:
        for label, key in (
            ("publish_place_start -> tcp_pose", "tcp_done"),
            ("tcp_pose -> waypoints_ready", "waypoints_done"),
            ("waypoints_ready -> plan_done", "plan_done"),
        ):
            if timing.get(key) is not None:
                print(
                    "\033[94m"
                    f"[time] {label} elapsed="
                    f"{timing[key] - (timing['place_start'] if key == 'tcp_done' else timing['tcp_done'] if key == 'waypoints_done' else timing['waypoints_done']):.4f}s\033[0m",
                    flush=True,
                )
        print(
            "\033[94m"
            f"[time] publish_place_start -> first_place_pose elapsed="
            f"{timing['first_pose'] - timing['place_start']:.4f}s\033[0m",
            flush=True,
        )
        print(
            "\033[94m"
            f"[time] first_place_pose -> place_end elapsed="
            f"{time.monotonic() - timing['first_pose']:.4f}s\033[0m",
            flush=True,
        )
    setattr(publisher, "_place_timing", None)
    if publisher_stop_requested(publisher):
        status = f"STOPPED by B during {hand} place target publishing"
        print(f"[place] {status}", flush=True)
        return FixedPlaceResult(False, status)

    print(
        "[place] fixed place target reached "
        f"hand={hand} object={normalize_object_type(object_type) if object_type else 'default'} "
        f"xyz=({place_target.position[0]:.3f},{place_target.position[1]:.3f},"
        f"{place_target.position[2]:.3f})m "
        f"nominal_xyz=({place_target.nominal_position[0]:.3f},"
        f"{place_target.nominal_position[1]:.3f},"
        f"{place_target.nominal_position[2]:.3f})m "
        f"waypoints={len(waypoints)} count={count} hold={place_target_hold_sec:.2f}s",
        flush=True,
    )

    release_status = send_gripper_signal("release", args, hand=hand)
    if "ERR " in release_status or "failed exit_code=" in release_status:
        status = f"{hand} place release failed: {release_status}"
        print(f"[place] {status}", flush=True)
        return FixedPlaceResult(False, status)
    else:
        status = f"{hand} place release ok"
        print(f"[place] {status}", flush=True)

    if target_count is not None and target_count > 1:
        wait_status = publish_target_wait_point(publisher, hand, args)
        return FixedPlaceResult(
            True,
            f"{status}; remaining targets={target_count}; {wait_status}",
        )

    if target_count is not None:
        keep_place_pose = target_count > 1
    elif keep_place_pose is None:
        keep_place_pose = not config.home_after_release
    if bool(keep_place_pose):
        keep_status = f"{hand} kept at place pose; next grasp starts here"
        print(f"[place] {keep_status}", flush=True)
        return FixedPlaceResult(True, f"{status}; {keep_status}")

    if publisher_stop_requested(publisher):
        stop_status = f"STOPPED by B before {hand} home target publishing"
        print(f"[place] {stop_status}", flush=True)
        return FixedPlaceResult(False, f"{status}; {stop_status}")

    home_status = publish_home(
        publisher,
        hand,
        position_for_home(hand, args),
        args,
        final_hold_sec=max(config.home_hold_sec, 0.0),
    )
    return FixedPlaceResult(True, f"{status}; {home_status}")


class PlaceSkill(SkillRuntime):
    """Object-oriented entry point for the fixed place workflow."""

    def __init__(self, *, args: argparse.Namespace, publisher, executor=None) -> None:
        super().__init__(args=args, publisher=publisher)
        self._executor = executor or execute_fixed_place_after_grasp

    def execute(
        self,
        hand: str,
        *,
        grasp_confirmed: bool,
        object_type: str | None = None,
        keep_place_pose: bool | None = None,
        target_count: int | None = None,
    ) -> FixedPlaceResult:
        return self._executor(
            self.publisher,
            hand,
            self.args,
            grasp_confirmed=grasp_confirmed,
            object_type=object_type,
            keep_place_pose=keep_place_pose,
            target_count=target_count,
        )


def position_for_home(hand: str, args: argparse.Namespace) -> tuple[float, float, float]:
    return HOME_CONFIG.position(normalize_hand(hand))
