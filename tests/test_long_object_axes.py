from argparse import Namespace
from itertools import permutations

import numpy as np
import pytest

from grasp_core.core.long_object_axes import canonical_long_object_pose
from grasp_core.core.robot_target_pose import make_target_object_pose, rpy_to_matrix
from grasp_core.perception.flowpose_pipeline import (
    FlowPoseObject,
    apply_long_object_axes_to_flowpose_output,
)
from grasp_core.tasks.screwdriver_handle_grasp_policy import (
    make_screwdriver_handle_gripper_pose,
)


@pytest.mark.parametrize("order", list(permutations(range(3))))
@pytest.mark.parametrize("sign", [-1, 1])
def test_raw_axis_permutations_keep_long_short_and_z_up(order, sign):
    x = np.array([0.6, 0.8, 0.0]) * sign
    z = np.array([0.0, 0.0, 1.0])
    rotation = np.column_stack((x, np.cross(z, x), z))
    pose = np.eye(4)
    pose[:3, :3] = rotation[:, order]
    if np.linalg.det(pose[:3, :3]) < 0:
        # Flip a short axis to preserve handedness without changing length sign.
        pose[:3, list(order).index(1)] *= -1
    pose[:3, 3] = [0.2, -0.1, 0.7]
    size = np.array([0.17, 0.02, 0.01])[list(order)]
    result, dimensions = canonical_long_object_pose(pose, size)
    np.testing.assert_allclose(result[:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(result[:3, 3], pose[:3, 3])
    np.testing.assert_allclose(dimensions, [0.17, 0.02, 0.01], atol=1e-12)
    again, again_size = canonical_long_object_pose(result, dimensions)
    np.testing.assert_allclose(again, result, atol=1e-12)
    np.testing.assert_allclose(again_size, dimensions, atol=1e-12)


def test_flattened_box_encloses_original_tilted_box():
    angle = np.deg2rad(20)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                       [-np.sin(angle), 0, np.cos(angle)]]
    size = np.array([0.17, 0.02, 0.01])
    result, dimensions = canonical_long_object_pose(pose, size)
    np.testing.assert_allclose(result[:3, :3], np.eye(3), atol=1e-12)
    for signs in np.ndindex(2, 2, 2):
        corner = pose[:3, :3] @ ((np.array(signs) * 2 - 1) * size / 2)
        assert np.all(np.abs(corner) <= dimensions / 2 + 1e-12)


@pytest.mark.parametrize("label", ["pen_1", "yellow_screwdriver_handle_1"])
@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("tilted_camera", [False, True])
def test_pipeline_uses_raw_paired_size_before_calibration_and_grasps_short_axis(
    label, hand, tilted_camera,
):
    # Raw length is Z, while the general calibration permutes Z into Y.
    raw = np.eye(4)
    raw[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    raw[:3, 3] = [0.2, 0.1, 0.7]
    calibrated = raw.copy()
    calibrated[:3, :3] = raw[:3, [1, 2, 0]]
    extrinsic = np.eye(4)
    if tilted_camera:
        extrinsic = rpy_to_matrix([-2.736556, -0.006884, -1.598774])
        extrinsic[:3, 3] = [0.099802, 0.037719, 1.227970]
    raw = np.linalg.inv(extrinsic) @ raw
    calibrated = np.linalg.inv(extrinsic) @ calibrated
    size = [0.02, 0.01, 0.17]
    obj = FlowPoseObject(label, [1, 1], calibrated.tolist(), size, 0.9)
    generic = FlowPoseObject("cube", [2, 2], calibrated.tolist(), size, 0.8)
    output = {"objects": [obj, generic], "pose_all": [obj.pose, generic.pose],
              "length_all": [size, size], "raw_pose_all": [raw.tolist(), raw.tolist()],
              "raw_length_all": [size, size]}
    corrected = apply_long_object_axes_to_flowpose_output(output, extrinsic)
    assert corrected["objects"][1] is generic
    assert output["objects"][0] is obj
    corrected_obj = corrected["objects"][0]
    np.testing.assert_allclose((extrinsic @ np.array(corrected_obj.pose))[:3, :3],
                               np.eye(3), atol=1e-12)
    np.testing.assert_allclose(corrected_obj.size, [0.17, 0.02, 0.01])
    assert corrected["pose_all"][0] == corrected_obj.pose
    assert corrected["length_all"][0] == corrected_obj.size
    target = make_target_object_pose(label=label, frame_id=label,
        camera_pose=np.array(corrected_obj.pose), base_to_camera=extrinsic,
        size=np.array(corrected_obj.size))
    args = Namespace(ik_grasp_tcp_offset_m=(0, 0, 0), ik_target_stage="grasp",
                     ik_orientation_quat=(0, 0, 0, 1), ik_downward_tilt_deg=0,
                     ik_downward_tilt_y_deg=45, ik_downward_tilt_frame="local")
    gripper, metadata = make_screwdriver_handle_gripper_pose(target, args, hand=hand)
    assert metadata.closing_axis_name == "y"
    assert abs(gripper[:3, 1] @ metadata.long_axis) < 1e-12
    assert abs(abs(gripper[:3, 1] @ metadata.side_axis) - 1) < 1e-12


def test_reported_capture_is_z_up_in_base_not_camera_frame():
    # 20260908_172623_frame029353_flowpose.json; size already has X longest.
    camera = np.array([
        [0.4694228172, 0.0004556799, 0.8829733729, 0.0936751738],
        [-0.8600645065, -0.2260762900, 0.4573603272, -0.0697004944],
        [0.1998277456, -0.9741094708, -0.1057334915, 0.6331254840],
        [0, 0, 0, 1],
    ])
    extrinsic = rpy_to_matrix([-2.736556, -0.006884, -1.598774])
    extrinsic[:3, 3] = [0.099802, 0.037719, 1.227970]
    target = make_target_object_pose(label="pen_1", frame_id="pen_1",
        camera_pose=camera, base_to_camera=extrinsic,
        size=np.array([0.1702255, 0.0123431, 0.0138738]))
    expected_long = (extrinsic @ camera)[:3, 0].copy()
    expected_long[2] = 0
    expected_long /= np.linalg.norm(expected_long)
    np.testing.assert_allclose(target.base_pose[:3, 0], expected_long)
    np.testing.assert_allclose(target.base_pose[:3, 2], [0, 0, 1])
    np.testing.assert_allclose(target.base_pose[:3, :3].T @ target.base_pose[:3, :3],
                               np.eye(3), atol=1e-12)
    assert np.linalg.det(target.base_pose[:3, :3]) == pytest.approx(1)
    np.testing.assert_allclose(target.base_pose[:3, 3], (extrinsic @ camera)[:3, 3])
    np.testing.assert_allclose(extrinsic @ target.camera_pose, target.base_pose, atol=1e-12)


@pytest.mark.parametrize("size", [[0, 1, 1], [1, np.nan, 1], [1, 2]])
def test_invalid_size_is_rejected(size):
    with pytest.raises(ValueError):
        canonical_long_object_pose(np.eye(4), size)


def test_vertical_long_axis_is_rejected():
    with pytest.raises(ValueError, match="horizontal"):
        canonical_long_object_pose(np.eye(4), [0.01, 0.01, 0.17])
