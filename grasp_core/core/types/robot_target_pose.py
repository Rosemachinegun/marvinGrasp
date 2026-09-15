"""Compatibility exports for target pose types.

New code should import types from ``core.types.target_pose`` and file loaders
from ``core.io.target_pose_loader``.
"""

from grasp_core.core.types.target_pose import *  # noqa: F401,F403
from grasp_core.core.types.camera import CameraExtrinsic
from grasp_core.core.io.target_pose_loader import (
    load_camera_extrinsic_from_xacro,
    load_target_objects_from_flowpose_json,
)
