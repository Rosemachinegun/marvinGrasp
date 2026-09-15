"""HOME and failed-action recovery execution."""

from __future__ import annotations

import argparse
import random
import numpy as np

from grasp_core.core.math.pose import (
    PoseWaypoint,
    ik_home_wrist_orientation_quat,
    quaternion_to_rotation_matrix,
    quaternion_angle_rad,
    rotation_matrix_from_xyz_euler_rad,
)
from grasp_core.core.types.robot_target_pose import matrix_to_quaternion
from grasp_core.execution.motion_executor import RequestIkTargetPublisher, publish_home
from grasp_core.execution.skill_runtime import SkillRuntime
from grasp_core.execution.config import HOME_CONFIG


class HomeAction(SkillRuntime):
    """Coordinates normal HOME and measured-start recovery motions."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        ik_publisher: RequestIkTargetPublisher | None,
    ) -> None:
        super().__init__(args=args, publisher=ik_publisher)
        self.ik_publisher = self.publisher

    def execute(
        self,
        hand: str,
        *,
        fresh_measured_start: bool = False,
        resume_stop_generation: int | None = None,
        target_xyz: tuple[float, float, float] | None = None,
        recovery_random_target: bool = False,
        orientation_xyzw: tuple[float, float, float, float] | None = None,
    ) -> str:
        home_xyz = target_xyz or HOME_CONFIG.position(hand)
        home_position = np.asarray(home_xyz, dtype=np.float64)
        runtime_args = HOME_CONFIG.apply_runtime_args(self.args, hand)
        home_orientation = ik_home_wrist_orientation_quat(runtime_args, hand=hand)
        start: PoseWaypoint | None = None
        if resume_stop_generation is not None:
            fresh_measured_start = True
        if fresh_measured_start:
            if self.ik_publisher is None:
                raise RuntimeError("measured pose publisher unavailable")
            cancelled = self.ik_publisher.stop_requested
            if resume_stop_generation is not None:
                cancelled = lambda: (
                    self.ik_publisher.stop_generation() != resume_stop_generation
                )
            start = self.ik_publisher.client.wait_for_settled_tool_pose(
                hand, cancelled=cancelled,
            )
            if recovery_random_target and target_xyz is None:
                home_xyz = self._random_recovery_target(hand, start[0])
                home_position = np.asarray(home_xyz, dtype=np.float64)
                orientation_xyzw = self._random_z_rotation(start[1])
                home_orientation = orientation_xyzw
            position_error_m = float(np.linalg.norm(start[0] - home_position))
            angle_error_deg = float(np.rad2deg(
                quaternion_angle_rad(start[1], home_orientation)
            ))
            position_tolerance_m = max(float(getattr(
                self.args, "interrupted_home_position_tolerance_m",
                HOME_CONFIG.interrupted_position_tolerance_m,
            )), 0.0)
            angle_tolerance_deg = max(float(getattr(
                self.args, "interrupted_home_angle_tolerance_deg",
                HOME_CONFIG.interrupted_angle_tolerance_deg,
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
        return publish_home(
            self.ik_publisher,
            hand,
            home_xyz,
            runtime_args,
            final_hold_sec=HOME_CONFIG.hold_sec,
            start_pose=start,
            orientation_xyzw=orientation_xyzw,
        )

    @staticmethod
    def _random_recovery_target(
        hand: str,
        current_xyz: np.ndarray,
    ) -> tuple[float, float, float]:
        """Create a small human-like escape target from the measured pose.

        Recovery moves toward decreasing base-frame Y, with small X/Z
        variation.  The target is deliberately local to the failed grasp so
        recovery does not jump to a fixed HOME failure point.
        """
        del hand  # Kept in the signature for future hand-specific tuning.
        current = np.asarray(current_xyz, dtype=np.float64)
        y_offset = random.uniform(
            HOME_CONFIG.recovery_y_offset_min_m,
            HOME_CONFIG.recovery_y_offset_max_m,
        )
        target = current.copy()
        target[0] = max(
            current[0] - random.uniform(
                HOME_CONFIG.recovery_x_offset_min_m,
                HOME_CONFIG.recovery_x_offset_max_m,
            ),
            HOME_CONFIG.recovery_min_x_m,
        )
        target[1] -= y_offset
        target[2] += random.uniform(
            HOME_CONFIG.recovery_z_lift_min_m,
            HOME_CONFIG.recovery_z_lift_max_m,
        )
        return tuple(float(value) for value in target)

    @staticmethod
    def _random_z_rotation(
        orientation_xyzw: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Interpolate the current base-frame Z rotation toward zero."""
        rotation = quaternion_to_rotation_matrix(orientation_xyzw)
        yaw_rad = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        ratio = random.uniform(
            HOME_CONFIG.recovery_z_rotation_interpolation_min,
            HOME_CONFIG.recovery_z_rotation_interpolation_max,
        )
        zero_yaw_rotation = rotation_matrix_from_xyz_euler_rad(
            (0.0, 0.0, -yaw_rad * (1.0 - ratio)),
        )
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = zero_yaw_rotation @ rotation
        return matrix_to_quaternion(result)

    def recover(
        self,
        hand: str,
        *,
        fresh_measured_start: bool = False,
        resume_stop_generation: int | None = None,
    ) -> str:
        return self.execute(
            hand,
            # Recovery is defined relative to the actual failed-grasp pose;
            # never fall back to the normal or legacy fixed HOME point.
            fresh_measured_start=True,
            resume_stop_generation=resume_stop_generation,
            recovery_random_target=True,
        )

    def move_to_target_wait_point(self, hand: str) -> str:
        """Move a hand to a random waiting point near its HOME position."""
        home = np.asarray(HOME_CONFIG.position(hand), dtype=np.float64)
        target = home.copy()
        target[0] += random.uniform(
            HOME_CONFIG.target_wait_x_forward_min_m,
            HOME_CONFIG.target_wait_x_forward_max_m,
        )
        target[2] = home[2]
        home_orientation = ik_home_wrist_orientation_quat(self.args, hand=hand)
        y_rotation = random.uniform(
            HOME_CONFIG.target_wait_y_rotation_min_rad,
            HOME_CONFIG.target_wait_y_rotation_max_rad,
        )
        rotation = quaternion_to_rotation_matrix(home_orientation)
        rotation = rotation @ rotation_matrix_from_xyz_euler_rad(
            (0.0, y_rotation, 0.0),
        )
        orientation_pose = np.eye(4, dtype=np.float64)
        orientation_pose[:3, :3] = rotation
        return self.execute(
            hand,
            # Place has just completed for this hand.  Re-reading a settled
            # FK pose here adds an unnecessary blocking wait; publish_home's
            # normal current-pose read is sufficient for this short transition.
            fresh_measured_start=False,
            target_xyz=tuple(float(value) for value in target),
            orientation_xyzw=matrix_to_quaternion(orientation_pose),
        )


HomeSkill = HomeAction


def publish_target_wait_point(
    publisher: RequestIkTargetPublisher | None,
    hand: str,
    args: argparse.Namespace,
) -> str:
    """Publish the randomized waiting point used between remaining targets."""
    return HomeAction(args=args, ik_publisher=publisher).move_to_target_wait_point(hand)
