from argparse import Namespace

import numpy as np

from grasp_core.core.types.robot_target_pose import TargetObjectPose
from grasp_core.core.math.object_axes import classify_box_shape
from grasp_core.planning.grasp.policies.box_symmetry import (
    apply_box_z_symmetry_grasp_policy,
)


def test_box_shape_uses_1_2_aspect_ratio_boundary() -> None:
    assert classify_box_shape(np.array([0.10, 0.12, 0.10])) == "cube"
    assert classify_box_shape(np.array([0.10, 0.1201, 0.10])) == "cuboid"


def test_box_symmetry_aligns_world_z_for_cube_like_object() -> None:
    base_pose = np.eye(4)
    base_pose[:3, 0] = np.array([1.0, 0.0, 0.0])
    base_pose[:3, 1] = np.array([0.0, 0.5, 0.5])
    base_pose[:3, 2] = np.array([0.0, -0.5, 0.5])
    base_pose[:3, 3] = np.array([0.2, 0.1, 0.8])
    target = TargetObjectPose(
        label="yellow_cube",
        frame_id="yellow_cube_1",
        camera_pose=np.eye(4),
        base_pose=base_pose,
        size=np.array([0.10, 0.11, 0.10]),
    )

    selection = apply_box_z_symmetry_grasp_policy(
        target,
        hand="left",
        args=Namespace(use_box_z_symmetry_grasp_policy=True),
    )

    assert selection is not None
    np.testing.assert_allclose(selection.target.base_pose[:3, 2], [0.0, 0.0, 1.0])


def test_screwdriver_handle_keeps_raw_pose_for_dedicated_policy() -> None:
    target = make_target("red_screwdriver_handle")

    selection = apply_box_z_symmetry_grasp_policy(
        target,
        hand="left",
        args=Namespace(use_box_z_symmetry_grasp_policy=True),
    )

    assert selection is None


def make_target(
    label: str,
    *,
    size: np.ndarray | None = None,
    xyz: np.ndarray | None = None,
) -> TargetObjectPose:
    base_pose = np.eye(4)
    base_pose[:3, 3] = np.array([0.2, 0.1, 0.8]) if xyz is None else xyz
    return TargetObjectPose(
        label=label,
        frame_id=f"{label}_1",
        camera_pose=np.eye(4),
        base_pose=base_pose,
        size=size,
    )
