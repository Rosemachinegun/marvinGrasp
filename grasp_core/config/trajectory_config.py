"""Configuration policies specific to trajectory diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

from grasp_core.config.resource_paths import (
    DEFAULT_JOINT_TRAJECTORY_CSV_DIR as RESOURCE_JOINT_TRAJECTORY_CSV_DIR,
    DEFAULT_TRAJECTORY_PLOT_DIR as RESOURCE_TRAJECTORY_PLOT_DIR,
)


@dataclass(frozen=True)
class TrajectoryConfig:
    """Defaults shared by trajectory planning and trajectory execution."""

    publish_rate_hz: float = 75.0
    publish_sec: float = 0.5
    step_m: float = 0.01
    step_deg: float = 1.0
    min_steps: int = 15
    speed_mps: float = 0.15
    angular_speed_dps: float = 35.0
    safe_z_m: float = 0.95
    plot_dir: Path = RESOURCE_TRAJECTORY_PLOT_DIR
    joint_trajectory_csv_dir: Path = RESOURCE_JOINT_TRAJECTORY_CSV_DIR

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrajectoryConfig":
        """Create typed trajectory settings from the CLI compatibility object."""
        defaults = cls()
        values = {
            field_name: getattr(args, field_name, getattr(defaults, field_name))
            for field_name in (
                "publish_rate_hz",
                "publish_sec",
                "step_m",
                "step_deg",
                "min_steps",
                "speed_mps",
                "angular_speed_dps",
            )
        }
        values.update(
            {
                "publish_rate_hz": getattr(args, "target_publish_rate_hz", values["publish_rate_hz"]),
                "step_m": getattr(args, "target_trajectory_step_m", values["step_m"]),
                "step_deg": getattr(args, "target_trajectory_step_deg", values["step_deg"]),
                "min_steps": getattr(args, "target_trajectory_min_steps", values["min_steps"]),
                "speed_mps": getattr(args, "target_trajectory_speed_mps", values["speed_mps"]),
                "angular_speed_dps": getattr(args, "target_trajectory_angular_speed_dps", values["angular_speed_dps"]),
            }
        )
        return replace(defaults, **values)


TRAJECTORY_CONFIG = TrajectoryConfig()

# Compatibility names for the parser and older tools.
DEFAULT_TARGET_PUBLISH_RATE_HZ = TRAJECTORY_CONFIG.publish_rate_hz
DEFAULT_TARGET_TRAJECTORY_STEP_M = TRAJECTORY_CONFIG.step_m
DEFAULT_TARGET_TRAJECTORY_STEP_DEG = TRAJECTORY_CONFIG.step_deg
DEFAULT_TARGET_TRAJECTORY_MIN_STEPS = TRAJECTORY_CONFIG.min_steps
DEFAULT_TARGET_TRAJECTORY_SPEED_MPS = TRAJECTORY_CONFIG.speed_mps
DEFAULT_TARGET_TRAJECTORY_ANGULAR_SPEED_DPS = TRAJECTORY_CONFIG.angular_speed_dps
DEFAULT_JOINT_TRAJECTORY_CSV_DIR = TRAJECTORY_CONFIG.joint_trajectory_csv_dir
DEFAULT_TRAJECTORY_PLOT_DIR = TRAJECTORY_CONFIG.plot_dir


def should_visualize_grasp_path(args: argparse.Namespace) -> bool:
    """Return the switch controlling grasp path recording and plots."""
    value = getattr(args, "visualize_grasp_path", None)
    if value is None:
        value = getattr(args, "target_trajectory_plot", True)
    return bool(value)


def should_save_joint_trajectory_csv(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "save_joint_trajectory_csv", False))
