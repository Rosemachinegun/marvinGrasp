"""Camera calibration data types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grasp_core.core.types.target_pose import rpy_to_matrix


@dataclass(frozen=True)
class CameraExtrinsic:
    parent_frame_id: str
    child_frame_id: str
    xyz: np.ndarray
    rpy: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        transform = rpy_to_matrix(self.rpy)
        transform[:3, 3] = self.xyz
        return transform
