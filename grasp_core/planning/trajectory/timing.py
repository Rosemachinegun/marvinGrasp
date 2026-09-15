"""Timing profiles for publishing Cartesian trajectory samples."""

from __future__ import annotations

import numpy as np

from grasp_core.core.math.pose import checked_position


def trajectory_linear_velocities(
    positions: list[np.ndarray],
    periods_sec: list[float],
) -> list[np.ndarray]:
    """Estimate waypoint velocities with zero start/end velocity boundaries."""

    if not positions:
        return []
    if len(positions) == 1:
        return [np.zeros(3, dtype=np.float64)]

    velocities: list[np.ndarray] = []
    for index in range(len(positions)):
        if index == 0 or index == len(positions) - 1:
            velocities.append(np.zeros(3, dtype=np.float64))
            continue

        previous_position = checked_position(positions[index - 1])
        next_position = checked_position(positions[index + 1])
        previous_dt = max(float(periods_sec[index]), 1e-4)
        next_dt = max(float(periods_sec[index + 1]), 1e-4)
        velocities.append(
            (next_position - previous_position) / (previous_dt + next_dt)
        )
    return velocities


def terminal_sample_periods(
    sample_count: int,
    base_period_sec: float,
    *,
    enabled: bool,
    tail_count: int = 8,
    max_scale: float = 3.0,
) -> list[float]:
    count = max(int(sample_count), 0)
    base_period = max(float(base_period_sec), 1e-4)
    periods = [base_period] * count
    if not enabled or count <= 1:
        return periods

    tail = min(max(int(tail_count), 1), count)
    max_scale = max(float(max_scale), 1.0)
    for tail_index in range(tail):
        ratio = float(tail_index + 1) / float(tail)
        scale = 1.0 + (max_scale - 1.0) * ratio * ratio
        periods[count - tail + tail_index] = base_period * scale
    return periods


def trajectory_sample_periods(
    sample_count: int,
    base_period_sec: float,
    *,
    startup_slowdown: bool = False,
    terminal_slowdown: bool = False,
    startup_hold_sec: float = 0.0,
    startup_ramp_count: int = 16,
    startup_max_scale: float = 4.0,
) -> list[float]:
    """Build timing with a stationary handover and gentle acceleration."""

    periods = terminal_sample_periods(
        sample_count,
        base_period_sec,
        enabled=terminal_slowdown,
    )
    if not startup_slowdown or not periods:
        return periods

    base_period = max(float(base_period_sec), 1e-4)
    periods[0] = max(float(startup_hold_sec), base_period)
    ramp = min(max(int(startup_ramp_count), 0), max(len(periods) - 1, 0))
    max_scale = max(float(startup_max_scale), 1.0)
    for ramp_index in range(ramp):
        ratio = float(ramp_index + 1) / float(max(ramp, 1))
        scale = 1.0 + (max_scale - 1.0) * (1.0 - ratio) ** 2
        sample_index = ramp_index + 1
        periods[sample_index] = max(periods[sample_index], base_period * scale)
    return periods


__all__ = [
    "terminal_sample_periods",
    "trajectory_sample_periods",
    "trajectory_linear_velocities",
]
