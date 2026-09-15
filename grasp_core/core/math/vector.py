"""Small, policy-independent vector normalization helpers."""

from __future__ import annotations

import numpy as np


def normalized(vector: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm < float(epsilon):
        raise ValueError("cannot normalize near-zero vector")
    return array / norm


def horizontal_unit_vector(
    vector: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray | None:
    projected = np.asarray(vector, dtype=np.float64).copy()
    projected[2] = 0.0
    norm = float(np.linalg.norm(projected))
    if norm < float(epsilon):
        return None
    return projected / norm


def clamp_vector_length(vector: np.ndarray, max_length: float) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(values))
    limit = max(float(max_length), 0.0)
    if length < 1e-9 or limit <= 0.0:
        return np.zeros_like(values)
    if length <= limit:
        return values
    return values * (limit / length)
