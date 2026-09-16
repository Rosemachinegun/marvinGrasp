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
    segment_lengths: tuple[float, ...]
    cumulative_lengths: tuple[float, ...]
    total_length: float


def _cubic_position(
    start: np.ndarray,
    control_1: np.ndarray,
    control_2: np.ndarray,
    end: np.ndarray,
    alpha: float,
) -> np.ndarray:
    t = float(np.clip(alpha, 0.0, 1.0))
    one_minus_t = 1.0 - t
    return (
        one_minus_t**3 * start
        + 3.0 * one_minus_t**2 * t * control_1
        + 3.0 * one_minus_t * t**2 * control_2
        + t**3 * end
    )


def _estimate_segment_length(
    start: np.ndarray,
    control_1: np.ndarray,
    control_2: np.ndarray,
    end: np.ndarray,
) -> float:
    alphas = np.linspace(0.0, 1.0, 33)
    positions = np.asarray(
        [_cubic_position(start, control_1, control_2, end, alpha) for alpha in alphas]
    )
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def prepare_pose_path(raw_waypoints: list[PoseWaypoint]) -> PreparedPosePath:
    if len(raw_waypoints) < 2:
        return PreparedPosePath((), (), (), (), (), (), 0.0)
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
    segment_lengths = tuple(
        _estimate_segment_length(
            positions[index],
            control_1[index],
            control_2[index],
            positions[index + 1],
        )
        for index in range(len(control_1))
    )
    cumulative = [0.0]
    for length in segment_lengths:
        cumulative.append(cumulative[-1] + length)
    return PreparedPosePath(
        positions,
        orientations,
        control_1,
        control_2,
        segment_lengths,
        tuple(cumulative),
        float(cumulative[-1]),
    )


def _evaluate_prepared_pose(
    path: PreparedPosePath,
    segment_index: int,
    alpha: float,
    *,
    global_progress: float | None = None,
) -> PoseWaypoint:
    t = float(np.clip(alpha, 0.0, 1.0))
    one_minus_t = 1.0 - t
    position = (
        one_minus_t**3 * path.positions[segment_index]
        + 3.0 * one_minus_t**2 * t * path.control_1[segment_index]
        + 3.0 * one_minus_t * t**2 * path.control_2[segment_index]
        + t**3 * path.positions[segment_index + 1]
    )
    if global_progress is None:
        global_progress = (
            path.cumulative_lengths[segment_index]
            + path.segment_lengths[segment_index] * t
        ) / max(path.total_length, 1e-12)
    orientation_progress = float(np.clip(global_progress, 0.0, 1.0))
    if path.total_length <= 1e-12:
        orientation = path.orientations[-1]
    else:
        orientation_distances = np.asarray(path.cumulative_lengths) / path.total_length
        orientation_index = min(
            int(np.searchsorted(orientation_distances, orientation_progress, side="right")) - 1,
            len(path.orientations) - 2,
        )
        local_start = orientation_distances[orientation_index]
        local_end = orientation_distances[orientation_index + 1]
        orientation_alpha = (
            (orientation_progress - local_start) / max(local_end - local_start, 1e-12)
        )
        orientation = np.asarray(
            slerp_quaternion(
                path.orientations[orientation_index],
                path.orientations[orientation_index + 1],
                orientation_alpha,
            ),
            dtype=np.float64,
        )
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
    final_slowdown_ratio: float = 0.0,
) -> list[PoseWaypoint]:
    """Sample the complete path once with optional final-segment slowdown."""

    if len(raw_waypoints) < 2:
        return []

    path = prepared if prepared is not None else prepare_pose_path(raw_waypoints)
    samples: list[PoseWaypoint] = []
    total_samples = max(sum(max(int(steps), 1) for steps in segment_steps), 1)
    first_segment_fraction = (
        path.segment_lengths[0] / max(path.total_length, 1e-12)
        if path.segment_lengths and path.total_length > 1e-12
        else 0.0
    )
    # Ease only the global beginning of the trajectory. The ramp ends before
    # the first waypoint so intermediate waypoint orientation remains exact.
    startup_ramp_fraction = min(
        10.0 / float(total_samples),
        first_segment_fraction * 0.5,
    )
    final_ratio = float(np.clip(final_slowdown_ratio, 0.0, 1.0))
    for segment_index, steps in enumerate(segment_steps):
        segment_length = path.segment_lengths[segment_index]
        segment_start = path.cumulative_lengths[segment_index]
        for step in range(1, int(steps) + 1):
            position_alpha = float(step) / float(max(int(steps), 1))
            global_progress = (
                segment_start + segment_length * position_alpha
            ) / max(path.total_length, 1e-12)
            if final_ratio > 0.0 and global_progress > 1.0 - final_ratio:
                slowdown_progress = (
                    global_progress - (1.0 - final_ratio)
                ) / final_ratio
                eased_global_progress = (1.0 - final_ratio) + final_ratio * smootherstep(
                    slowdown_progress
                )
                position_alpha = (
                    eased_global_progress * path.total_length - segment_start
                ) / max(segment_length, 1e-12)
                global_progress = eased_global_progress
            eased_orientation_progress = startup_ramp_alpha(
                global_progress,
                startup_ramp_fraction,
            )
            samples.append(
                _evaluate_prepared_pose(
                    path,
                    segment_index,
                    position_alpha,
                    global_progress=eased_orientation_progress,
                )
            )
    return samples


def smooth_pose_sample_chain(
    samples: list[PoseWaypoint],
    *,
    segment_steps: list[int],
    passes: int = 1,
) -> list[PoseWaypoint]:
    """Smooth an already concatenated path as one continuous sample chain.

    Each pass inserts an interpolated point between every adjacent sample and
    rebuilds the global tangent field. Original sample endpoints remain in the
    chain, while velocity changes at former segment boundaries are blended.
    The dense chain is an internal representation only.  It is finally
    resampled by arc length using the original per-segment sample counts, so
    smoothing does not change the published trajectory duration.
    """
    pass_count = max(int(passes), 0)
    if pass_count == 0 or len(samples) < 2:
        return list(samples)

    chain = list(samples)
    for _ in range(pass_count):
        prepared = prepare_pose_path(chain)
        chain = [chain[0]] + interpolate_pose_samples(
            chain,
            [2] * (len(chain) - 1),
            prepared=prepared,
        )

    target_steps = [max(int(steps), 1) for steps in segment_steps]
    multiplier = 2**pass_count
    result: list[PoseWaypoint] = []
    dense_start = 0
    for steps in target_steps:
        dense_end = dense_start + steps * multiplier
        dense_segment = chain[dense_start : dense_end + 1]
        if len(dense_segment) < 2:
            dense_start = dense_end
            continue

        positions = np.asarray([pose[0] for pose in dense_segment], dtype=np.float64)
        distances = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        )
        total_distance = float(distances[-1])
        for sample_index in range(1, steps + 1):
            if sample_index == steps:
                result.append(dense_segment[-1])
                continue
            distance = total_distance * sample_index / float(steps)
            upper = int(np.searchsorted(distances, distance, side="right"))
            upper = min(max(upper, 1), len(dense_segment) - 1)
            lower = upper - 1
            local_fraction = (distance - distances[lower]) / max(
                distances[upper] - distances[lower], 1e-12
            )
            position = positions[lower] + local_fraction * (
                positions[upper] - positions[lower]
            )
            orientation = slerp_quaternion(
                dense_segment[lower][1],
                dense_segment[upper][1],
                float(local_fraction),
            )
            result.append((position, tuple(float(value) for value in orientation)))
        dense_start = dense_end
    return result


__all__ = [
    "cubic_bezier_position",
    "cubic_bezier_from_tangents",
    "startup_ramp_alpha",
    "interpolate_pose_samples",
    "smooth_pose_sample_chain",
    "PreparedPosePath",
    "prepare_pose_path",
    "position_tangents",
    "smootherstep",
]
