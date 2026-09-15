"""Configuration-derived Cartesian trajectory constraints."""

from __future__ import annotations

import argparse
import numpy as np

from grasp_core.core.math.pose import (
    PoseWaypoint,
)
from grasp_core.execution.config import (
    DEFAULT_TARGET_PUBLISH_RATE_HZ,
    DEFAULT_TARGET_TRAJECTORY_ANGULAR_SPEED_DPS,
    DEFAULT_TARGET_TRAJECTORY_SPEED_MPS,
    DEFAULT_TARGET_TRAJECTORY_STEP_DEG,
    DEFAULT_TARGET_TRAJECTORY_STEP_M,
)
from grasp_core.planning.trajectory.interpolation import (
    PreparedPosePath,
    _evaluate_prepared_pose,
    prepare_pose_path,
    startup_ramp_alpha,
)
from grasp_core.planning.trajectory.orientation import quaternion_step_rad


def effective_trajectory_step_limits(args: argparse.Namespace) -> tuple[float, float]:
    """Combine configured geometric step limits with speed/rate limits."""

    publish_rate_hz = max(
        float(getattr(args, "target_publish_rate_hz", DEFAULT_TARGET_PUBLISH_RATE_HZ)),
        0.1,
    )
    configured_step_m = max(
        float(getattr(args, "target_trajectory_step_m", DEFAULT_TARGET_TRAJECTORY_STEP_M)),
        1e-4,
    )
    configured_step_deg = max(
        float(getattr(args, "target_trajectory_step_deg", DEFAULT_TARGET_TRAJECTORY_STEP_DEG)),
        0.1,
    )
    speed_mps = max(
        float(getattr(args, "target_trajectory_speed_mps", DEFAULT_TARGET_TRAJECTORY_SPEED_MPS)),
        1e-4,
    )
    angular_speed_dps = max(
        float(
            getattr(
                args,
                "target_trajectory_angular_speed_dps",
                DEFAULT_TARGET_TRAJECTORY_ANGULAR_SPEED_DPS,
            )
        ),
        0.1,
    )
    return min(configured_step_m, speed_mps / publish_rate_hz), min(
        configured_step_deg,
        angular_speed_dps / publish_rate_hz,
    )


def refine_segment_steps_for_limits(
    raw_waypoints: list[PoseWaypoint],
    segment_steps: list[int],
    *,
    max_step_m: float,
    max_step_deg: float,
    max_iterations: int = 6,
    prepared: PreparedPosePath | None = None,
) -> list[int]:
    """Increase segment sample counts until the actual curve respects limits."""

    if len(raw_waypoints) < 2:
        return []

    path = prepared if prepared is not None else prepare_pose_path(raw_waypoints)
    positions = path.positions
    orientations = path.orientations
    refined_steps = [max(int(steps), 1) for steps in segment_steps]
    linear_limit = max(float(max_step_m), 1e-4)
    angular_limit = np.deg2rad(max(float(max_step_deg), 0.1))

    for _ in range(max_iterations):
        changed = False
        for segment_index, steps in enumerate(refined_steps):
            ramp_fraction = min(10.0 / float(max(steps, 1)), 1.0)
            max_linear_step = 0.0
            max_angular_step = 0.0
            previous_position = positions[segment_index]
            previous_orientation = orientations[segment_index]
            for step in range(1, steps + 1):
                alpha = float(step) / float(steps)
                if segment_index == 0:
                    alpha = startup_ramp_alpha(alpha, ramp_fraction)
                position, orientation = _evaluate_prepared_pose(
                    path, segment_index, alpha
                )
                max_linear_step = max(
                    max_linear_step,
                    float(np.linalg.norm(position - previous_position)),
                )
                max_angular_step = max(
                    max_angular_step,
                    quaternion_step_rad(previous_orientation, orientation),
                )
                previous_position = position
                previous_orientation = orientation

            ratio = max(
                max_linear_step / linear_limit,
                max_angular_step / angular_limit,
            )
            if ratio > 1.0 + 1e-6:
                refined_steps[segment_index] = max(
                    steps + 1,
                    int(np.ceil(float(steps) * ratio * 1.05)),
                )
                changed = True
        if not changed:
            break

    return refined_steps


__all__ = ["effective_trajectory_step_limits", "refine_segment_steps_for_limits"]
