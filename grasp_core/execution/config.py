"""Unified configuration owned by the execution layer."""

from __future__ import annotations

import argparse
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grasp_core.config.resource_paths import (
    DEFAULT_JOINT_TRAJECTORY_CSV_DIR as RESOURCE_JOINT_TRAJECTORY_CSV_DIR,
    DEFAULT_TRAJECTORY_PLOT_DIR as RESOURCE_TRAJECTORY_PLOT_DIR,
)


GRIP_MIN_LIMIT_TOKENS = ("GRASP_FAILED_MIN_LIMIT", "GRIP_FAILED_MIN_LIMIT")


@dataclass(frozen=True)
class GraspConfig:
    grip_min_limit_tokens: tuple[str, ...] = GRIP_MIN_LIMIT_TOKENS
    force_object_z: bool = True
    forced_object_z_m: float = 0.66
    pregrasp_distance_m: float = 0.3
    ik_target_stage: str = "pregrasp"
    approach_axis: str = "z"
    approach_sign: float = -1.0
    use_flowpose_grasp_rotation: bool = False
    use_cube_z_symmetry_grasp_policy: bool = True
    ik_grasp_tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ik_pregrasp_extra_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ik_orientation_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    left_orientation_xyz_rad: tuple[float, float, float] = (0.0, 0.7853981634, -0.7853981634)
    right_orientation_xyz_rad: tuple[float, float, float] = (0.0, 0.7853981634, 0.7853981634)
    orientation_frame: str = "local"
    visualize_grasp_path: bool = False
    save_joint_trajectory_csv: bool = False
    show_raw_flowpose_window: bool = False
    target_publish_rate_hz: float = 75.0
    target_trajectory_step_m: float = 0.01
    target_trajectory_step_deg: float = 1.0
    target_trajectory_min_steps: int = 15
    target_trajectory_speed_mps: float = 0.15
    target_trajectory_angular_speed_dps: float = 35.0
    final_approach_samples: int = 12
    final_approach_slowdown_ratio: float = 0.05
    grip_settle_sec: float = 0.0
    grip_post_confirm_hold_sec: float = 0.0

    def controlled_args(self, args: argparse.Namespace) -> argparse.Namespace:
        controlled = copy(args)
        for field_name in (
            "force_object_z", "forced_object_z_m", "pregrasp_distance_m",
            "ik_target_stage", "approach_axis", "approach_sign",
            "use_flowpose_grasp_rotation", "use_cube_z_symmetry_grasp_policy",
            "ik_grasp_tcp_offset_m", "ik_pregrasp_extra_offset_m",
            "ik_orientation_quat", "orientation_frame", "visualize_grasp_path",
            "save_joint_trajectory_csv", "show_raw_flowpose_window",
            "target_publish_rate_hz", "target_trajectory_step_m",
            "target_trajectory_step_deg", "target_trajectory_min_steps",
            "target_trajectory_speed_mps", "target_trajectory_angular_speed_dps",
            "grip_settle_sec", "grip_post_confirm_hold_sec",
        ):
            setattr(controlled, field_name, getattr(self, field_name))
        controlled.grip_min_limit_tokens = self.grip_min_limit_tokens
        controlled.grasp_orientation_xyz_rad_left = self.left_orientation_xyz_rad
        controlled.grasp_orientation_xyz_rad_right = self.right_orientation_xyz_rad
        for hand_name, orientation_xyz_rad in (("left", self.left_orientation_xyz_rad), ("right", self.right_orientation_xyz_rad)):
            roll_rad, pitch_rad, yaw_rad = orientation_xyz_rad
            dominant_axis = max(
                (("x", abs(roll_rad), roll_rad), ("y", abs(pitch_rad), pitch_rad), ("z", abs(yaw_rad), yaw_rad)),
                key=lambda item: item[1],
            )
            axis_name, axis_rad = dominant_axis[0], dominant_axis[2]
            setattr(controlled, "ik_downward_tilt_axis", axis_name)
            setattr(controlled, "ik_downward_tilt_deg", float(np.rad2deg(axis_rad)))
            setattr(controlled, f"ik_downward_tilt_{hand_name}_deg", float(np.rad2deg(axis_rad)))
            setattr(controlled, "ik_downward_tilt_y_deg", float(np.rad2deg(pitch_rad)))
            setattr(controlled, f"ik_downward_tilt_y_{hand_name}_deg", float(np.rad2deg(pitch_rad)))
        controlled.ik_downward_tilt_frame = self.orientation_frame
        controlled.target_trajectory_plot = self.visualize_grasp_path
        return controlled


@dataclass(frozen=True)
class PlaceConfig:
    left_xyz: tuple[float, float, float] = (0.45, 0.5, 0.75)
    right_xyz: tuple[float, float, float] = (0.45, -0.5, 0.75)
    object_overrides: dict[str, tuple[float, float, float]] | None = None
    lift_height_m: float = 0.3
    safe_z_m: float = 0.95
    target_hold_sec: float = 0.0
    home_after_release: bool = False
    home_hold_sec: float = 0.05
    left_orientation_xyz_rad: tuple[float, float, float] = (0.0, 0.4, 0.0)
    right_orientation_xyz_rad: tuple[float, float, float] = (0.0, 0.4, 0.0)
    position_jitter_xyz_m: tuple[float, float, float] = (0.03, 0.03, 0.02)
    pitch_jitter_rad: float = 0.1
    lift_height_jitter_m: float = 0.1
    approach_distance_m: float = 0.025
    approach_orientation_ratio: float = 0.78
    final_approach_samples: int = 8
    lift_ramp_ratio: float = 0.4

    def object_targets(self) -> dict[str, tuple[float, float, float]]:
        return self.object_overrides or {
            "yellow_cube": (0.40, -0.40, 0.86),
            "yellow_duck": (0.37, -0.33, 0.80),
            "blue_cube": (0.35, -0.35, 0.83),
        }


@dataclass(frozen=True)
class HomeConfig:
    left_xyz: tuple[float, float, float] = (0.25, 0.25, 0.83)
    right_xyz: tuple[float, float, float] = (0.25, -0.25, 0.83)
    safe_z_m: float = 0.95
    side_clearance_y_m: float = 0.28
    tilt_z_left_deg: float = 0.0
    tilt_z_right_deg: float = 0.0
    tilt_y_left_deg: float = 0.0
    tilt_y_right_deg: float = 0.0
    interrupted_position_tolerance_m: float = 0.006
    interrupted_angle_tolerance_deg: float = 2.0
    hold_sec: float = 0.0
    recovery_y_offset_min_m: float = 0.02
    recovery_y_offset_max_m: float = 0.04
    recovery_x_offset_min_m: float = 0.1
    recovery_x_offset_max_m: float = 0.2
    recovery_min_x_m: float = 0.30
    recovery_z_lift_min_m: float = 0.05
    recovery_z_lift_max_m: float = 0.1
    recovery_z_rotation_interpolation_min: float = 0.0
    recovery_z_rotation_interpolation_max: float = 1.0
    target_wait_x_forward_min_m: float = 0.1
    target_wait_x_forward_max_m: float = 0.2
    target_wait_y_rotation_min_rad: float = 0.4
    target_wait_y_rotation_max_rad: float = 0.2

    def position(self, hand: str) -> tuple[float, float, float]:
        return self.left_xyz if hand == "left" else self.right_xyz

    def apply_runtime_args(self, args: argparse.Namespace, hand: str) -> argparse.Namespace:
        runtime_args = copy(args)
        runtime_args.home_safe_z_m = self.safe_z_m
        runtime_args.home_side_clearance_y_m = self.side_clearance_y_m
        runtime_args.home_tilt_z_left_deg = self.tilt_z_left_deg
        runtime_args.home_tilt_z_right_deg = self.tilt_z_right_deg
        runtime_args.home_tilt_y_left_deg = self.tilt_y_left_deg
        runtime_args.home_tilt_y_right_deg = self.tilt_y_right_deg
        return runtime_args


@dataclass(frozen=True)
class MotionConfig:
    publish_rate_hz: float = 75.0
    publish_sec: float = 0.5
    step_m: float = 0.01
    step_deg: float = 1.0
    min_steps: int = 15
    speed_mps: float = 0.15
    angular_speed_dps: float = 35.0
    plot_dir: Path = RESOURCE_TRAJECTORY_PLOT_DIR
    joint_trajectory_csv_dir: Path = RESOURCE_JOINT_TRAJECTORY_CSV_DIR


EXECUTION_CONFIG = {
    "grasp": GraspConfig(),
    "place": PlaceConfig(),
    "home": HomeConfig(),
    "motion": MotionConfig(),
}
GRASP_CONFIG = EXECUTION_CONFIG["grasp"]
PLACE_CONFIG = EXECUTION_CONFIG["place"]
HOME_CONFIG = EXECUTION_CONFIG["home"]
MOTION_CONFIG = EXECUTION_CONFIG["motion"]

# Compatibility constants.  New code should read MOTION_CONFIG directly.
DEFAULT_TARGET_PUBLISH_RATE_HZ = MOTION_CONFIG.publish_rate_hz
DEFAULT_TARGET_TRAJECTORY_STEP_M = MOTION_CONFIG.step_m
DEFAULT_TARGET_TRAJECTORY_STEP_DEG = MOTION_CONFIG.step_deg
DEFAULT_TARGET_TRAJECTORY_MIN_STEPS = MOTION_CONFIG.min_steps
DEFAULT_TARGET_TRAJECTORY_SPEED_MPS = MOTION_CONFIG.speed_mps
DEFAULT_TARGET_TRAJECTORY_ANGULAR_SPEED_DPS = MOTION_CONFIG.angular_speed_dps
DEFAULT_JOINT_TRAJECTORY_CSV_DIR = MOTION_CONFIG.joint_trajectory_csv_dir
DEFAULT_TRAJECTORY_PLOT_DIR = MOTION_CONFIG.plot_dir
