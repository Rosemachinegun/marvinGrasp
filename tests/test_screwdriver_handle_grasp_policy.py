from argparse import Namespace

import numpy as np

from grasp_core.core.pose_math import quaternion_to_rotation_matrix
from grasp_core.core.long_object_axes import canonical_long_object_pose
from grasp_core.core.robot_target_pose import TargetObjectPose
from grasp_core.planning.grasp_pose import make_gripper_target_pose
from grasp_core.planning.tool_pick_templates import build_pick_template_waypoints
import grasp_core.tasks.screwdriver_handle_grasp_policy as screwdriver_policy
from grasp_core.tasks.screwdriver_handle_grasp_policy import (
    LongObjectGraspPolicy,
    approach_axis_from_orientation,
    build_screwdriver_handle_pick_waypoints,
    closing_axis_from_orientation,
    make_screwdriver_handle_gripper_pose,
    screwdriver_handle_long_axis,
    screwdriver_handle_long_axis_index,
    screwdriver_handle_z_up_object_pose,
)


def test_screwdriver_handle_keeps_y_tilt_and_closing_axis_is_lateral_left() -> None:
    target = make_screwdriver_target(long_axis=np.array([1.0, 1.0, 0.4]))

    result = make_screwdriver_handle_gripper_pose(
        target,
        args(ik_downward_tilt_y_left_deg=45.0),
        hand="left",
    )

    assert result is not None
    gripper_pose, metadata = result
    object_pose = metadata.object_pose
    np.testing.assert_allclose(object_pose[:3, 2], [0.0, 0.0, 1.0])
    assert metadata.tilt_y_deg == 45.0
    assert abs(float(closing_axis_from_orientation(gripper_pose[:3, :3]) @ metadata.long_axis)) < 1e-8
    assert horizontal_norm(closing_axis_from_orientation(gripper_pose[:3, :3])) > 0.70
    assert float(approach_axis_from_orientation(gripper_pose[:3, :3]) @ object_pose[:3, 2]) > 0.70


def test_screwdriver_handle_pose_selects_right_halfspace_side() -> None:
    target = make_screwdriver_target(long_axis=np.array([1.0, 1.0, 0.0]))

    result = make_screwdriver_handle_gripper_pose(
        target,
        args(ik_downward_tilt_y_right_deg=45.0),
        hand="right",
    )

    assert result is not None
    gripper_pose, metadata = result
    object_pose = metadata.object_pose
    closing_axis = closing_axis_from_orientation(gripper_pose[:3, :3])
    assert metadata.tilt_y_deg == 45.0
    assert abs(float(closing_axis @ metadata.long_axis)) < 1e-8
    assert horizontal_norm(closing_axis) > 0.70


def test_screwdriver_handle_right_hand_avoids_equivalent_180deg_wrist_flip() -> None:
    target = make_screwdriver_target(long_axis=np.array([1.0, 0.0, 0.0]))

    result = make_screwdriver_handle_gripper_pose(
        target,
        args(
            ik_downward_tilt_right_deg=45.0,
            ik_downward_tilt_y_right_deg=45.0,
            ik_downward_tilt_axis="z",
        ),
        hand="right",
    )

    assert result is not None
    _gripper_pose, metadata = result
    assert abs(metadata.yaw_deg) <= 90.0


def test_screwdriver_handle_right_hand_aligns_local_y_closing_axis() -> None:
    target = make_screwdriver_target(long_axis=np.array([-1.0, 1.0, 0.0]))

    result = make_screwdriver_handle_gripper_pose(
        target,
        args(
            ik_downward_tilt_right_deg=45.0,
            ik_downward_tilt_y_right_deg=45.0,
            ik_downward_tilt_axis="z",
        ),
        hand="right",
    )

    assert result is not None
    gripper_pose, metadata = result
    closing_axis = closing_axis_from_orientation(gripper_pose[:3, :3])
    assert abs(float(closing_axis @ metadata.long_axis)) < 1e-8
    assert abs(float(abs(closing_axis @ metadata.side_axis) - 1.0)) < 1e-8


def test_screwdriver_handle_template_waypoints_keep_y_tilt_ignore_yaml_quaternion() -> None:
    target = make_screwdriver_target(long_axis=np.array([1.0, 0.0, 0.0]))
    relative_waypoints = [
        (
            np.array([0.0, 0.0, 0.2]),
            (0.171010, 0.171010, -0.03015, 0.96984),
            0.0,
        ),
        (
            np.array([0.0, 0.0, 0.04]),
            (0.171010, 0.171010, -0.03015, 0.96984),
            1.0,
        ),
    ]

    waypoints = build_screwdriver_handle_pick_waypoints(
        target,
        relative_waypoints,
        args(ik_downward_tilt_y_left_deg=45.0),
        hand="left",
    )

    assert waypoints is not None
    object_pose = screwdriver_handle_z_up_object_pose(target.base_pose, target.size)
    long_axis = screwdriver_handle_long_axis(target.base_pose, target.size)
    for position, orientation, _gripper_value in waypoints:
        rotation = quaternion_to_rotation_matrix(orientation)
        assert float(approach_axis_from_orientation(rotation) @ object_pose[:3, 2]) > 0.70
        assert (
            abs(float(closing_axis_from_orientation(rotation) @ long_axis))
            < 1e-8
        )
        assert position[2] > target.base_pose[2, 3]


def test_screwdriver_handle_waypoint_x_long_axis_offset_is_hand_independent() -> None:
    target = make_screwdriver_target(long_axis=np.array([0.2, 1.0, 0.0]))
    relative_waypoints = [
        (
            np.array([-0.05, 0.0, 0.03]),
            (0.0, 0.0, 0.0, 1.0),
            1.0,
        ),
    ]

    left_waypoints = build_screwdriver_handle_pick_waypoints(
        target,
        relative_waypoints,
        args(ik_downward_tilt_left_deg=-45.0),
        hand="left",
    )
    right_waypoints = build_screwdriver_handle_pick_waypoints(
        target,
        relative_waypoints,
        args(ik_downward_tilt_right_deg=45.0),
        hand="right",
    )

    assert left_waypoints is not None
    assert right_waypoints is not None
    np.testing.assert_allclose(
        left_waypoints[0][0],
        right_waypoints[0][0],
    )

    object_pose = screwdriver_handle_z_up_object_pose(target.base_pose, target.size)
    expected_position = (
        object_pose
        @ np.array([-0.05, 0.0, 0.03, 1.0], dtype=np.float64)
    )[:3]
    np.testing.assert_allclose(
        left_waypoints[0][0],
        expected_position,
    )


def test_screwdriver_handle_integrates_with_global_y_tilt_but_custom_z_yaw() -> None:
    target = make_screwdriver_target(long_axis=np.array([1.0, 0.0, 0.0]))

    gripper_pose, _template, fallback_reason = make_gripper_target_pose(
        target,
        args(
            ik_downward_tilt_left_deg=-45.0,
            ik_downward_tilt_right_deg=45.0,
            ik_downward_tilt_axis="z",
            ik_downward_tilt_y_left_deg=45.0,
            ik_downward_tilt_y_right_deg=45.0,
        ),
        hand="left",
    )

    object_pose = screwdriver_handle_z_up_object_pose(target.base_pose, target.size)
    long_axis = screwdriver_handle_long_axis(target.base_pose, target.size)
    assert fallback_reason == "screwdriver_handle_z_yaw_policy"
    assert (
        float(approach_axis_from_orientation(gripper_pose[:3, :3]) @ object_pose[:3, 2])
        > 0.70
    )
    assert (
        abs(
            float(closing_axis_from_orientation(gripper_pose[:3, :3]) @ long_axis)
        )
        < 1e-8
    )


def test_screwdriver_handle_uses_z_up_local_y_as_long_axis() -> None:
    pose = np.array(
        [
            [0.9838460166918025, -0.1788843948818775, 0.006883945628626315, 0.1006392166018486],
            [-0.16174550929752052, -0.9047477087225659, -0.39404311157962474, -0.11112722009420395],
            [0.07671639760676369, 0.38656429844077866, -0.9190661768932799, 0.6505053639411926],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    size = np.array([0.0444459393620491, 0.020002959296107292, 0.10555826872587204])

    assert screwdriver_handle_long_axis_index(size) == 0
    pose = canonical_long_object_pose(pose, source_long_axis=0)[0]
    long_axis = screwdriver_handle_long_axis(pose, size)

    z_up_pose = screwdriver_handle_z_up_object_pose(pose, size)
    expected = z_up_pose[:3, 0]
    np.testing.assert_allclose(long_axis, expected)


def test_screwdriver_handle_real_capture_closing_x_is_not_long_axis() -> None:
    pose = np.array(
        [
            [-0.3830627202987671, -0.9236584305763245, 0.010864862240850925, 0.05669158324599266],
            [-0.7809002995491028, 0.31752994656562805, -0.5379307270050049, -0.10034602135419846],
            [0.49341437220573425, -0.2145455926656723, -0.8429189920425415, 0.6472218632698059],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    size = np.array([0.0379534587264061, 0.02592730149626732, 0.09254536032676697])
    target = TargetObjectPose(
        label="red_screwdriver_handle",
        frame_id="red_screwdriver_handle_1",
        camera_pose=np.eye(4),
        base_pose=canonical_long_object_pose(pose, source_long_axis=0)[0],
        size=size,
    )

    result = make_screwdriver_handle_gripper_pose(
        target,
        args(
            ik_downward_tilt_right_deg=45.0,
            ik_downward_tilt_y_right_deg=45.0,
            ik_downward_tilt_axis="z",
        ),
        hand="right",
    )

    assert result is not None
    gripper_pose, metadata = result
    closing_axis = closing_axis_from_orientation(gripper_pose[:3, :3])
    assert metadata.long_axis_index == 0
    assert abs(float(closing_axis @ metadata.long_axis)) < 1e-8


def test_screwdriver_handle_captures_close_perpendicular_to_flowpose_x_long_axis() -> None:
    capture_poses = (
        np.array(
            [
                [0.1452111005783081, -0.9891809821128845, -0.020848996937274933, 0.20597906410694122],
                [-0.8634114265441895, -0.11640176177024841, -0.4908883273601532, -0.14546464383602142],
                [0.48315054178237915, 0.08928371220827103, -0.8709729909896851, 0.6647924780845642],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [-0.40771448612213135, -0.9130849242210388, -0.006684210151433945, 0.17271284759044647],
                [-0.8099409937858582, 0.36501896381378174, -0.4590824544429779, -0.027604399248957634],
                [0.421621173620224, -0.1817607879638672, -0.888368546962738, 0.6149181723594666],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )

    for index, pose in enumerate(capture_poses):
        target = TargetObjectPose(
            label="yellow_screwdriver_handle",
            frame_id=f"yellow_screwdriver_handle_{index}",
            camera_pose=pose,
            base_pose=canonical_long_object_pose(pose, source_long_axis=0)[0],
            size=np.array([0.03, 0.02, 0.10], dtype=np.float64),
        )

        result = make_screwdriver_handle_gripper_pose(
            target,
            args(
                ik_downward_tilt_right_deg=45.0,
                ik_downward_tilt_y_right_deg=45.0,
                ik_downward_tilt_axis="z",
            ),
            hand="right",
        )

        assert result is not None
        gripper_pose, metadata = result
        assert metadata.long_axis_index == 0
        raw_x_long_axis = pose[:3, 0].copy()
        raw_x_long_axis[2] = 0.0
        raw_x_long_axis /= np.linalg.norm(raw_x_long_axis)
        np.testing.assert_allclose(
            np.abs(metadata.long_axis),
            np.abs(raw_x_long_axis),
        )
        closing_axis = closing_axis_from_orientation(gripper_pose[:3, :3])
        assert (
            abs(float(closing_axis @ metadata.object_pose[:3, 0]))
            < 1e-8
        )
        assert abs(float(abs(closing_axis @ metadata.object_pose[:3, 1]) - 1.0)) < 1e-8


def test_pen_uses_screwdriver_handle_long_object_policy_by_default() -> None:
    target = make_screwdriver_target(long_axis=np.array([1.0, 0.0, 0.0]), label="pen")

    result = make_screwdriver_handle_gripper_pose(
        target,
        args(
            ik_downward_tilt_right_deg=45.0,
            ik_downward_tilt_y_right_deg=45.0,
            ik_downward_tilt_axis="z",
        ),
        hand="right",
    )

    assert result is not None
    gripper_pose, metadata = result
    closing_axis = closing_axis_from_orientation(gripper_pose[:3, :3])
    assert metadata.closing_axis_name == "y"
    assert metadata.closing_axis_index == 1
    assert abs(float(closing_axis @ metadata.long_axis)) < 1e-8
    assert abs(float(abs(closing_axis @ metadata.object_pose[:3, 1]) - 1.0)) < 1e-8


def test_pen_closing_axis_can_be_configured_to_policy_y(monkeypatch) -> None:
    target = make_screwdriver_target(long_axis=np.array([1.0, 0.0, 0.0]), label="pen")
    monkeypatch.setattr(
        screwdriver_policy,
        "LONG_OBJECT_POLICIES",
        (
            LongObjectGraspPolicy(
                keyword="screwdriver_handle",
                closing_axis="y",
            ),
            LongObjectGraspPolicy(
                keyword="pen",
                closing_axis="x",
            ),
        ),
    )

    result = make_screwdriver_handle_gripper_pose(
        target,
        args(
            ik_downward_tilt_right_deg=45.0,
            ik_downward_tilt_y_right_deg=45.0,
            ik_downward_tilt_axis="z",
        ),
        hand="right",
    )

    assert result is not None
    gripper_pose, metadata = result
    closing_axis = closing_axis_from_orientation(gripper_pose[:3, :3])
    assert metadata.closing_axis_name == "x"
    assert metadata.closing_axis_index == 0
    assert abs(float(abs(closing_axis @ metadata.long_axis) - 1.0)) < 1e-8


def test_generic_template_path_is_unchanged() -> None:
    target = make_target("yellow_cube")
    relative_waypoints = [(np.array([0.0, 0.0, 0.1]), (0.0, 0.0, 0.0, 1.0), 0.0)]

    waypoints = build_pick_template_waypoints(
        target,
        relative_waypoints,
        args(
            ik_downward_tilt_left_deg=-45.0,
            ik_downward_tilt_axis="z",
            ik_downward_tilt_y_left_deg=45.0,
        ),
        hand="left",
    )

    rotation = quaternion_to_rotation_matrix(waypoints[0][1])
    assert not np.allclose(rotation[:3, 2], [0.0, 0.0, 1.0])


def args(**overrides) -> Namespace:
    values = {
        "ik_grasp_tcp_offset_m": (0.0, 0.0, 0.0),
        "ik_target_stage": "grasp",
        "ik_orientation_quat": (0.0, 0.0, 0.0, 1.0),
        "ik_downward_tilt_deg": 0.0,
        "ik_downward_tilt_left_deg": None,
        "ik_downward_tilt_right_deg": None,
        "ik_downward_tilt_axis": "y",
        "ik_downward_tilt_y_deg": 0.0,
        "ik_downward_tilt_y_left_deg": None,
        "ik_downward_tilt_y_right_deg": None,
        "ik_downward_tilt_frame": "local",
        "use_flowpose_grasp_rotation": True,
        "approach_axis": "z",
        "approach_sign": -1.0,
        "pregrasp_distance_m": 0.05,
        "ik_pregrasp_extra_offset_m": (0.0, 0.0, 0.0),
    }
    values.update(overrides)
    return Namespace(**values)


def make_screwdriver_target(
    long_axis: np.ndarray,
    label: str = "red_screwdriver_handle",
) -> TargetObjectPose:
    pose = np.eye(4)
    x_axis = long_axis / np.linalg.norm(long_axis)
    y_axis = np.cross([0.0, 0.0, 1.0], x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    pose = canonical_long_object_pose(pose, source_long_axis=0)[0]
    pose[:3, 3] = [0.2, 0.1, 0.8]
    return TargetObjectPose(
        label=label,
        frame_id=f"{label}_1",
        camera_pose=np.eye(4),
        base_pose=pose,
        size=np.array([0.02, 0.12, 0.02]),
    )


def horizontal_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)[:2]))


def make_target(label: str) -> TargetObjectPose:
    pose = np.eye(4)
    pose[:3, 3] = [0.2, 0.1, 0.8]
    return TargetObjectPose(
        label=label,
        frame_id=f"{label}_1",
        camera_pose=np.eye(4),
        base_pose=pose,
    )
