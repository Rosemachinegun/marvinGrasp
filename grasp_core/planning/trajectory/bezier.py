"""Bezier path construction helpers."""

from __future__ import annotations

import argparse
import numpy as np

from grasp_core.core.math.pose import PoseWaypoint, checked_position, slerp_quaternion
from grasp_core.planning.trajectory.constraints import effective_trajectory_step_limits
from grasp_core.planning.trajectory.interpolation import (
    cubic_bezier_position,
    smootherstep,
)


def smooth_bezier_arc_waypoints(
    start_position: np.ndarray,
    start_orientation: tuple[float, float, float, float],
    end_position: np.ndarray,
    end_orientation: tuple[float, float, float, float],
    args: argparse.Namespace,
    *,
    lift_arc: bool = True,
    ease_orientation: bool = False,
    lift_height_m: float = 0.12,
    safe_z_m: float | None = None,
    lift_ramp_ratio: float = 0.0,
) -> list[PoseWaypoint]:
    """Return constant-distance samples of one smooth cubic Bezier path.

    ``lift_ramp_ratio`` moves the lift control points away from the endpoints,
    so the path enters and leaves the elevated section progressively instead
    of starting with an abrupt vertical motion.
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
            float(getattr(args, "home_safe_z_m", 0.8))
            if safe_z_m is None
            else float(safe_z_m)
        )
        arc_apex_z = max(max_endpoint_z + lift_m, configured_safe_z)
        ramp_ratio = float(np.clip(lift_ramp_ratio, 0.0, 0.5))
        delta = end_position - start_position
        control_1 = start_position + delta * ramp_ratio
        control_2 = end_position - delta * ramp_ratio
        control_z = (
            arc_apex_z
            - 0.125 * (float(start_position[2]) + float(end_position[2]))
        ) / 0.75
        control_1[2] = float(start_position[2]) + (
            control_z - float(start_position[2])
        ) * ramp_ratio
        control_2[2] = float(end_position[2]) + (
            control_z - float(end_position[2])
        ) * ramp_ratio
    else:
        delta = end_position - start_position
        cruise_z = max(float(start_position[2]), float(end_position[2]))
        control_1 = start_position.copy()
        control_1[:2] += delta[:2] / 3.0
        control_1[2] = cruise_z
        control_2 = end_position.copy()
        control_2[2] = cruise_z

    max_step_m, _max_step_deg = effective_trajectory_step_limits(args)
    lookup_alpha = np.linspace(0.0, 1.0, 401)
    lookup_positions = np.asarray(
        [
            cubic_bezier_position(
                start_position, control_1, control_2, end_position, alpha
            )
            for alpha in lookup_alpha
        ]
    )
    cumulative_length = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(np.linalg.norm(np.diff(lookup_positions, axis=0), axis=1)),
        )
    )
    curve_length_m = float(cumulative_length[-1])
    # These are geometric waypoints, not controller publish samples.  Using
    # the 1 kHz controller step here can create thousands of tiny segments;
    # the motion planner will resample the resulting curve at the configured
    # limits afterward.  Keep enough points to preserve the arc shape while
    # bounding planning/refinement overhead.
    geometric_sample_count = int(np.ceil(curve_length_m / max_step_m))
    sample_count = max(min(geometric_sample_count, 128), 2)
    sample_distances = np.linspace(0.0, curve_length_m, sample_count + 1)[1:]
    sample_alphas = np.interp(sample_distances, cumulative_length, lookup_alpha)
    waypoints: list[PoseWaypoint] = []
    for alpha_raw in sample_alphas:
        alpha = float(alpha_raw)
        curve_position = cubic_bezier_position(
            start_position, control_1, control_2, end_position, alpha
        )
        orientation_alpha = smootherstep(alpha) if ease_orientation else alpha
        curve_orientation = slerp_quaternion(
            start_orientation, end_orientation, orientation_alpha
        )
        waypoints.append((curve_position, curve_orientation))
    return waypoints


__all__ = ["smooth_bezier_arc_waypoints"]
