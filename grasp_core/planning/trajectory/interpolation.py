"""Pure interpolation helpers used by Cartesian trajectory planners."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from grasp_core.core.math.pose import (
    PoseWaypoint,
    checked_position,
    normalize_quaternion,
    slerp_quaternion,
)
from grasp_core.core.math.vector import clamp_vector_length
from grasp_core.core.math.easing import smootherstep


@dataclass(frozen=True)
class PreparedPosePath:
    positions: tuple[np.ndarray, ...]
    orientations: tuple[np.ndarray, ...]
    control_1: tuple[np.ndarray, ...]
    control_2: tuple[np.ndarray, ...]


def prepare_pose_path(raw_waypoints: list[PoseWaypoint]) -> PreparedPosePath:
    if len(raw_waypoints) < 2:
        return PreparedPosePath((), (), (), ())
    positions = tuple(
        checked_position(position).astype(np.float64, copy=True)
        for position, _ in raw_waypoints
    )
    orientations = tuple(
        np.asarray(normalize_quaternion(orientation), dtype=np.float64)
        for _, orientation in raw_waypoints
    )
    tangents = position_tangents(list(positions), stop_at_endpoints=True)
    control_1 = tuple(
        positions[index] + tangents[index] / 3.0
        for index in range(len(tangents) - 1)
    )
    control_2 = tuple(
        positions[index + 1] - tangents[index + 1] / 3.0
        for index in range(len(tangents) - 1)
    )
    return PreparedPosePath(positions, orientations, control_1, control_2)


def _evaluate_prepared_pose(
    path: PreparedPosePath, segment_index: int, alpha: float
) -> PoseWaypoint:
    t = float(np.clip(alpha, 0.0, 1.0))
    one_minus_t = 1.0 - t
    position = (
        one_minus_t**3 * path.positions[segment_index]
        + 3.0 * one_minus_t**2 * t * path.control_1[segment_index]
        + 3.0 * one_minus_t * t**2 * path.control_2[segment_index]
        + t**3 * path.positions[segment_index + 1]
    )
    start = path.orientations[segment_index]
    end = path.orientations[segment_index + 1]
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    if dot > 0.9995:
        orientation = start + t * (end - start)
        orientation /= np.linalg.norm(orientation)
    else:
        theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        sin_theta_0 = float(np.sin(theta_0))
        theta = theta_0 * t
        scale_1 = float(np.sin(theta) / sin_theta_0)
        scale_0 = float(np.cos(theta) - dot * scale_1)
        orientation = scale_0 * start + scale_1 * end
    return position, tuple(float(value) for value in orientation)

def cubic_bezier_position(
    start: np.ndarray,
    control_1: np.ndarray,
    control_2: np.ndarray,
    end: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Evaluate a cubic Bezier position at ``alpha`` in [0, 1]."""

    t = float(np.clip(alpha, 0.0, 1.0))
    one_minus_t = 1.0 - t
    return (
        one_minus_t**3 * checked_position(start)
        + 3.0 * one_minus_t**2 * t * checked_position(control_1)
        + 3.0 * one_minus_t * t**2 * checked_position(control_2)
        + t**3 * checked_position(end)
    )


def cubic_bezier_from_tangents(
    start_position: np.ndarray,
    end_position: np.ndarray,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Evaluate a cubic Bezier segment defined by endpoint tangents."""

    return cubic_bezier_position(
        start_position,
        start_position + start_tangent / 3.0,
        end_position - end_tangent / 3.0,
        end_position,
        alpha,
    )


def startup_ramp_alpha(alpha: float, ramp_fraction: float) -> float:
    """Ease from zero velocity/acceleration, then rejoin unit-rate time."""

    value = float(np.clip(alpha, 0.0, 1.0))
    ramp = float(np.clip(ramp_fraction, 0.0, 1.0))
    if ramp <= 0.0 or value >= ramp:
        return value
    ratio = value / ramp
    # g(0)=g'(0)=g''(0)=0 and g(1)=g'(1)=1, g''(1)=0.
    eased_ratio = 3.0 * ratio**5 - 8.0 * ratio**4 + 6.0 * ratio**3
    return ramp * eased_ratio


def position_tangents(
    positions: list[np.ndarray],
    *,
    stop_at_endpoints: bool = False,
) -> list[np.ndarray]:
    """Return clamped waypoint tangents so the target glides through waypoints."""

    count = len(positions)
    if count == 0:
        return []
    if count == 1:
        return [np.zeros(3, dtype=np.float64)]

    tangents: list[np.ndarray] = []
    for index, position in enumerate(positions):
        if stop_at_endpoints and index in {0, count - 1}:
            tangent = np.zeros(3, dtype=np.float64)
            max_length = 0.0
        elif index == 0:
            tangent = positions[1] - position
            max_length = float(np.linalg.norm(tangent))
        elif index == count - 1:
            tangent = position - positions[index - 1]
            max_length = float(np.linalg.norm(tangent))
        else:
            previous_delta = position - positions[index - 1]
            next_delta = positions[index + 1] - position
            tangent = 0.5 * (previous_delta + next_delta)
            max_length = 0.5 * min(
                float(np.linalg.norm(previous_delta)),
                float(np.linalg.norm(next_delta)),
            )
        tangents.append(clamp_vector_length(tangent, max_length))
    return tangents


def interpolate_pose_samples(
    raw_waypoints: list[PoseWaypoint],
    segment_steps: list[int],
    *,
    prepared: PreparedPosePath | None = None,
) -> list[PoseWaypoint]:
    """Sample all Bezier segments and interpolate their orientations."""

    if len(raw_waypoints) < 2:
        return []

    path = prepared if prepared is not None else prepare_pose_path(raw_waypoints)
    samples: list[PoseWaypoint] = []

    for segment_index, steps in enumerate(segment_steps):
        ramp_fraction = min(10.0 / float(max(int(steps), 1)), 1.0)
        for step in range(1, int(steps) + 1):
            alpha = float(step) / float(steps)
            if segment_index == 0:
                alpha = startup_ramp_alpha(alpha, ramp_fraction)
            samples.append(_evaluate_prepared_pose(path, segment_index, alpha))
    return samples


__all__ = [
    "cubic_bezier_position",
    "cubic_bezier_from_tangents",
    "startup_ramp_alpha",
    "interpolate_pose_samples",
    "PreparedPosePath",
    "prepare_pose_path",
    "position_tangents",
    "smootherstep",
]
