"""Canonical perception axes for pens and screwdriver handles.

Published object frame: X = horizontal length, Y = horizontal width,
Z = base/world up. Size is expressed in that same frame.
"""

import re

import numpy as np

BOX_CUBE_ASPECT_RATIO_MAX = 1.2


def classify_box_shape(size: np.ndarray | None) -> str | None:
    """Classify a target by its three edge lengths.

    A target is cube-like when its longest edge is no more than 1.2 times its
    shortest edge. Missing or invalid dimensions remain unclassified.
    """
    if size is None:
        return None
    dimensions = np.asarray(size, dtype=np.float64).reshape(-1)
    if (
        dimensions.size != 3
        or not np.all(np.isfinite(dimensions))
        or np.any(dimensions <= 0)
    ):
        return None
    ratio = float(np.max(dimensions) / np.min(dimensions))
    return "cube" if ratio <= BOX_CUBE_ASPECT_RATIO_MAX else "cuboid"


def is_long_object(label: str, size: np.ndarray | None = None) -> bool:
    """Return whether a target uses the long-object axis convention.

    Dimensions are authoritative for the unified grasp pipeline. The label
    fallback keeps older callers without dimensions compatible.
    """
    shape = classify_box_shape(size)
    if shape is not None:
        return shape == "cuboid"
    name = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
    return "screwdriver_handle" in name or "pen" in name


def canonical_long_object_pose(
    base_pose: np.ndarray,
    size: np.ndarray | None = None,
    *,
    source_long_axis: int | None = None,
    allow_horizontal_fallback: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Rebuild Z-up axes from a pose and its *matching* box dimensions.

    With no explicit source axis, identify length from the raw box dimensions.
    Legacy calibrated poses require source_long_axis=0 because their saved
    dimensions were not permuted along with their axes. Preserve the source
    long-axis sign; choosing a hand must never change relative offsets.
    """
    pose = np.asarray(base_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("long-object pose must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("long-object pose has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if (
        np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 0.05
        or abs(np.linalg.det(rotation) - 1) > 0.05
    ):
        raise ValueError("long-object pose must contain a right-handed rotation")
    dimensions = None if size is None else np.asarray(size, dtype=np.float64)
    if dimensions is not None and (
        dimensions.shape != (3,)
        or not np.all(np.isfinite(dimensions))
        or np.any(dimensions <= 0)
    ):
        raise ValueError("long-object size must contain three positive finite values")
    if source_long_axis is None:
        if dimensions is None:
            raise ValueError("raw long-object axis selection requires box dimensions")
        if allow_horizontal_fallback:
            horizontal_norms = np.linalg.norm(pose[:2, :3], axis=0)
            weighted_norms = dimensions * horizontal_norms
            source_long_axis = int(np.argmax(weighted_norms))
            if weighted_norms[source_long_axis] < 1e-8:
                source_long_axis = int(np.argmax(horizontal_norms))
        else:
            source_long_axis = int(np.argmax(dimensions))
    if source_long_axis not in (0, 1, 2):
        raise ValueError("source long axis must be 0, 1 or 2")
    x = pose[:3, source_long_axis].copy()
    x[2] = 0.0
    norm = np.linalg.norm(x)
    if norm < 1e-8:
        raise ValueError("long-object long axis has no valid horizontal component")
    x /= norm
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    result = pose.copy()
    result[:3, :3] = np.column_stack((x, y, z))
    # Enclose the original oriented box after flattening its coordinate frame.
    # A simple dimension permutation would lose the contribution of its tilt.
    result_size = None
    if dimensions is not None:
        result_size = np.abs(result[:3, :3].T @ pose[:3, :3]) @ dimensions
    return result, result_size
