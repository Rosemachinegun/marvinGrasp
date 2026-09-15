#!/usr/bin/env python3
"""运动层：唯一负责笛卡尔 waypoint 插值和轨迹步长计算的模块。

这里不发布 ROS、不控制夹爪、不读取配置文件，也不做抓取任务决策。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grasp_core.core.math.pose import (
    PoseWaypoint,
    checked_position,
    normalize_quaternion,
    path_segment_steps,
)
from grasp_core.planning.trajectory.bezier import smooth_bezier_arc_waypoints
from grasp_core.planning.trajectory.constraints import (
    effective_trajectory_step_limits,
    refine_segment_steps_for_limits,
)
from grasp_core.planning.trajectory.interpolation import (
    cubic_bezier_from_tangents,
    interpolate_pose_samples,
    prepare_pose_path,
    position_tangents,
)
from grasp_core.planning.trajectory.timing import trajectory_linear_velocities

# Legacy imports kept for callers that historically imported every helper from
# this module. New code should import each helper from its owning module.
__all__ = [
    "TrajectoryPlan",
    "plan_trajectory",
    "plan_pose_path",
    "plan_pose_target",
    "smooth_bezier_arc_waypoints",
    "interpolate_pose_samples",
    "position_tangents",
    "cubic_bezier_from_tangents",
    "refine_segment_steps_for_limits",
    "trajectory_linear_velocities",
    "effective_trajectory_step_limits",
]


@dataclass(frozen=True)
class TrajectoryPlan:
    """Interpolated Cartesian trajectory plus metadata for logging/plotting."""

    raw_waypoints: list[PoseWaypoint]
    samples: list[PoseWaypoint]
    segment_steps: list[int]


def plan_trajectory(
    start_position: np.ndarray,
    start_orientation: tuple[float, float, float, float],
    waypoints: list[PoseWaypoint],
    *,
    max_step_m: float,
    max_step_deg: float,
    min_steps: int = 1,
) -> TrajectoryPlan:
    """Interpolate a Cartesian path without stopping at each intermediate point."""

    current_position = checked_position(start_position).copy()
    current_orientation = normalize_quaternion(start_orientation)
    raw_waypoints: list[PoseWaypoint] = [(current_position.copy(), current_orientation)]
    segment_steps: list[int] = []

    for end_position_raw, end_orientation_raw in waypoints:
        end_position = checked_position(end_position_raw)
        end_orientation = normalize_quaternion(end_orientation_raw)
        steps = path_segment_steps(
            current_position,
            current_orientation,
            end_position,
            end_orientation,
            max_step_m=max_step_m,
            max_step_deg=max_step_deg,
            min_steps=min_steps,
        )
        raw_waypoints.append((end_position.copy(), end_orientation))
        segment_steps.append(steps)

        current_position = end_position
        current_orientation = end_orientation

    prepared = prepare_pose_path(raw_waypoints)
    segment_steps = refine_segment_steps_for_limits(
        raw_waypoints,
        segment_steps,
        max_step_m=max_step_m,
        max_step_deg=max_step_deg,
        prepared=prepared,
    )
    samples = interpolate_pose_samples(
        raw_waypoints, segment_steps, prepared=prepared
    )
    if segment_steps:
        # Always publish the exact measured start pose before any angular
        # interpolation; this is the controller handover orientation.
        samples.insert(0, (raw_waypoints[0][0].copy(), raw_waypoints[0][1]))
        segment_steps[0] += 1
    return TrajectoryPlan(
        raw_waypoints=raw_waypoints,
        samples=samples,
        segment_steps=segment_steps,
    )


def plan_pose_path(
    start_position: np.ndarray,
    start_orientation: tuple[float, float, float, float],
    waypoints: list[PoseWaypoint],
    *,
    max_step_m: float,
    max_step_deg: float,
    min_steps: int = 1,
) -> TrajectoryPlan:
    """Compatibility wrapper for :func:`plan_trajectory`."""

    return plan_trajectory(
        start_position,
        start_orientation,
        waypoints,
        max_step_m=max_step_m,
        max_step_deg=max_step_deg,
        min_steps=min_steps,
    )


def plan_pose_target(
    start_position: np.ndarray,
    start_orientation: tuple[float, float, float, float],
    end_position: np.ndarray,
    end_orientation: tuple[float, float, float, float],
    *,
    max_step_m: float,
    max_step_deg: float,
    min_steps: int = 1,
) -> TrajectoryPlan:
    """Interpolate a single target as a one-segment path."""

    return plan_trajectory(
        start_position,
        start_orientation,
        [(checked_position(end_position), normalize_quaternion(end_orientation))],
        max_step_m=max_step_m,
        max_step_deg=max_step_deg,
        min_steps=min_steps,
    )
