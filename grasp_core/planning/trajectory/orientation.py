"""Orientation calculations used by trajectory planning."""

from __future__ import annotations

import numpy as np

from grasp_core.core.math.pose import normalize_quaternion


def quaternion_step_rad(
    start_xyzw: tuple[float, float, float, float],
    end_xyzw: tuple[float, float, float, float],
) -> float:
    q0 = np.asarray(normalize_quaternion(start_xyzw), dtype=np.float64)
    q1 = np.asarray(normalize_quaternion(end_xyzw), dtype=np.float64)
    dot = abs(float(np.dot(q0, q1)))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


__all__ = ["quaternion_step_rad"]
