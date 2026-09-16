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
    quaternion_angle_rad,
    slerp_quaternion,
)
from grasp_core.planning.trajectory.interpolation import cubic_bezier_position
from grasp_core.core.math.easing import smootherstep
from grasp_core.planning.trajectory.bezier import (
    smooth_bezier_arc_waypoints as _smooth_bezier_arc_waypoints,
)
from grasp_core.planning.trajectory.constraints import (
    refine_segment_steps_for_limits as _refine_segment_steps_for_limits,
    segment_steps_for_path as _segment_steps_for_path,
)
from grasp_core.planning.trajectory.interpolation import (
    interpolate_pose_samples as _interpolate_pose_samples,
    prepare_pose_path as _prepare_pose_path,
    smooth_pose_sample_chain as _smooth_pose_sample_chain,
)

# Only planner entry points are public. Low-level implementations are imported
# privately so callers must use their owning module for Bézier, interpolation,
# constraint, and timing helpers.
__all__ = [
    "TrajectoryPlan",
    "plan_trajectory",
    "plan_pose_path",
    "plan_pose_target",
    "build_smooth_waypoints",
]


def build_smooth_waypoints(
    start_position: np.ndarray,
    start_orientation: tuple[float, float, float, float],
    end_position: np.ndarray,
    end_orientation: tuple[float, float, float, float],
    args,
    *,
    lift_arc: bool = True,
    ease_orientation: bool = False,
    lift_height_m: float = 0.12,
    safe_z_m: float | None = None,
    lift_ramp_ratio: float = 0.0,
) -> list[PoseWaypoint]:
    """Build geometric path waypoints through the trajectory planner API."""
    return _smooth_bezier_arc_waypoints(
        start_position,
        start_orientation,
        end_position,
        end_orientation,
        args,
        lift_arc=lift_arc,
        ease_orientation=ease_orientation,
        lift_height_m=lift_height_m,
        safe_z_m=safe_z_m,
        lift_ramp_ratio=lift_ramp_ratio,
    )


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
    final_slowdown_ratio: float = 0.0,
    smoothing_passes: int = 2,
    direct_bezier: bool = False,
) -> TrajectoryPlan:
    """Interpolate a Cartesian path without stopping at intermediate points.

    Multi-segment paths get two additional global chain passes after the
    initial sampling pass, which blends velocity changes across joins.
    """

    current_position = checked_position(start_position).copy()
    current_orientation = normalize_quaternion(start_orientation)
    raw_waypoints: list[PoseWaypoint] = [(current_position.copy(), current_orientation)]
    segment_steps: list[int] = []

    for end_position_raw, end_orientation_raw in waypoints:
        end_position = checked_position(end_position_raw)
        end_orientation = normalize_quaternion(end_orientation_raw)
        raw_waypoints.append((end_position.copy(), end_orientation))
        segment_steps.append(1)

        current_position = end_position
        current_orientation = end_orientation

    if direct_bezier:
        if len(raw_waypoints) != 4:
            raise ValueError("direct_bezier requires exactly four control poses")
        controls = [position for position, _ in raw_waypoints]
        dense_alphas = np.linspace(0.0, 1.0, 1001)
        dense_positions = np.asarray(
            [
                cubic_bezier_position(
                    controls[0], controls[1], controls[2], controls[3], alpha
                )
                for alpha in dense_alphas
            ],
            dtype=np.float64,
        )
        curve_length = float(
            np.linalg.norm(np.diff(dense_positions, axis=0), axis=1).sum()
        )
        angle = quaternion_angle_rad(
            raw_waypoints[0][1], raw_waypoints[-1][1]
        )
        linear_steps = int(np.ceil(curve_length / max(float(max_step_m), 1e-4)))
        angular_steps = int(
            np.ceil(angle / np.deg2rad(max(float(max_step_deg), 0.1)))
        )
        total_steps = max(int(min_steps), linear_steps, angular_steps, 1)

        # Arc length gives an initial estimate, but a cubic Bézier is not
        # uniform in parameter ``t``.  Recheck the actual local chords and
        # refine until every published position step respects the limit.
        linear_limit = max(float(max_step_m), 1e-4)
        for _ in range(8):
            check_alphas = np.linspace(0.0, 1.0, total_steps + 1)
            check_positions = np.asarray(
                [
                    cubic_bezier_position(
                        controls[0], controls[1], controls[2], controls[3], alpha
                    )
                    for alpha in check_alphas
                ],
                dtype=np.float64,
            )
            max_local_step = float(
                np.linalg.norm(np.diff(check_positions, axis=0), axis=1).max()
            )
            if max_local_step <= linear_limit * (1.0 + 1e-6):
                break
            total_steps = max(
                total_steps + 1,
                int(np.ceil(total_steps * max_local_step / linear_limit * 1.05)),
            )
        samples: list[PoseWaypoint] = []
        final_ratio = float(np.clip(final_slowdown_ratio, 0.0, 1.0))
        for step in range(1, total_steps + 1):
            progress = step / float(total_steps)
            if final_ratio > 0.0 and progress > 1.0 - final_ratio:
                tail_progress = (progress - (1.0 - final_ratio)) / final_ratio
                progress = (1.0 - final_ratio) + final_ratio * smootherstep(
                    tail_progress
                )
            position = cubic_bezier_position(
                controls[0], controls[1], controls[2], controls[3], progress
            )
            orientation = slerp_quaternion(
                raw_waypoints[0][1], raw_waypoints[-1][1], progress
            )
            samples.append((position, orientation))
        # The direct curve is one executable path. Zero-length metadata
        # entries retain the original semantic control poses for diagnostics.
        segment_steps = [total_steps, 0, 0]
        samples.insert(0, (raw_waypoints[0][0].copy(), raw_waypoints[0][1]))
        segment_steps[0] += 1
        return TrajectoryPlan(raw_waypoints, samples, segment_steps)

    prepared = _prepare_pose_path(raw_waypoints)
    segment_steps = _segment_steps_for_path(
        prepared,
        max_step_m=max_step_m,
        max_step_deg=max_step_deg,
        min_steps=min_steps,
    )
    segment_steps = _refine_segment_steps_for_limits(
        raw_waypoints,
        segment_steps,
        max_step_m=max_step_m,
        max_step_deg=max_step_deg,
        prepared=prepared,
    )
    samples = _interpolate_pose_samples(
        raw_waypoints,
        segment_steps,
        prepared=prepared,
        final_slowdown_ratio=final_slowdown_ratio,
    )
    # The extra chain pass is specifically for blending multiple segments.
    # Keep single-target paths on the original sampler so their startup and
    # terminal timing characteristics remain unchanged.
    if smoothing_passes > 0 and len(raw_waypoints) > 2 and samples:
        samples = _smooth_pose_sample_chain(
            [raw_waypoints[0]] + samples,
            segment_steps=segment_steps,
            passes=smoothing_passes,
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
