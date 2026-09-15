"""Configuration policies specific to trajectory diagnostics."""

from __future__ import annotations

import argparse


def should_visualize_grasp_path(args: argparse.Namespace) -> bool:
    """Return the switch controlling grasp path recording and plots."""
    value = getattr(args, "visualize_grasp_path", None)
    if value is None:
        value = getattr(args, "target_trajectory_plot", True)
    return bool(value)


def should_save_joint_trajectory_csv(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "save_joint_trajectory_csv", False))
