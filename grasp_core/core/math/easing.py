"""Reusable scalar easing functions."""

from __future__ import annotations

import numpy as np


def smoothstep(alpha: float) -> float:
    t = float(np.clip(alpha, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def smootherstep(alpha: float) -> float:
    t = float(np.clip(alpha, 0.0, 1.0))
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)
