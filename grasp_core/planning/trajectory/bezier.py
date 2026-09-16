"""Bezier path construction helpers."""

from __future__ import annotations

import numpy as np

from grasp_core.core.math.pose import PoseWaypoint, checked_position, slerp_quaternion
from grasp_core.core.math.easing import smootherstep
from grasp_core.config.trajectory_config import TRAJECTORY_CONFIG


def smooth_bezier_arc_waypoints(
    start_position: np.ndarray,
    start_orientation: tuple[float, float, float, float],
    end_position: np.ndarray,
    end_orientation: tuple[float, float, float, float],
    args=None,
    *,
    lift_arc: bool = True,
    ease_orientation: bool = False,
    lift_height_m: float = 0.12,
    safe_z_m: float | None = None,
    lift_ramp_ratio: float = 0.0,
) -> list[PoseWaypoint]:
    """Return geometric anchors for one smooth cubic-Bezier-like path.

    This function intentionally does not sample the curve. The final sampling
    is performed once by ``plan_trajectory`` after all task waypoints have
    been assembled.
    """

    start_position = checked_position(start_position)
    end_position = checked_position(end_position)
    distance_m = float(np.linalg.norm(end_position - start_position))
    if distance_m < 1e-4:
        return [(end_position.copy(), end_orientation)]

    if lift_arc:
        max_endpoint_z = max(float(start_position[2]), float(end_position[2]))
        lift_m = max(float(lift_height_m), 0.0)
        configured_safe_z = (
            float(TRAJECTORY_CONFIG.safe_z_m)
            if safe_z_m is None
            else float(safe_z_m)
        )
        arc_apex_z = max(max_endpoint_z + lift_m, configured_safe_z)
        ramp_ratio = float(np.clip(lift_ramp_ratio, 0.0, 0.5))
        delta = end_position - start_position
        control_1 = start_position + delta * ramp_ratio
        control_2 = end_position - delta * ramp_ratio
        control_1[2] = float(start_position[2]) + (
            arc_apex_z - float(start_position[2])
        ) * min(2.0 * ramp_ratio, 1.0)
        control_2[2] = float(end_position[2]) + (
            arc_apex_z - float(end_position[2])
        ) * min(2.0 * ramp_ratio, 1.0)
        anchor_alphas = (ramp_ratio, 1.0 - ramp_ratio, 1.0)
        anchor_positions = (control_1, control_2, end_position)
        return [
            (
                checked_position(position).copy(),
                slerp_quaternion(
                    start_orientation,
                    end_orientation,
                    smootherstep(float(alpha)) if ease_orientation else float(alpha),
                ),
            )
            for position, alpha in zip(anchor_positions, anchor_alphas, strict=True)
        ]
    else:
        orientation_alpha = smootherstep(1.0) if ease_orientation else 1.0
        return [
            (
                end_position.copy(),
                slerp_quaternion(
                    start_orientation,
                    end_orientation,
                    orientation_alpha,
                ),
            )
        ]


__all__ = ["smooth_bezier_arc_waypoints"]
