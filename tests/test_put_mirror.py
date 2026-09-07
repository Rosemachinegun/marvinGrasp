from argparse import Namespace

import numpy as np

from grasp_core.core.pose_math import (
    ik_wrist_orientation_quat,
    quaternion_to_rotation_matrix,
)
from grasp_core.tasks.put import (
    cubic_bezier_position,
    execute_fixed_put_after_grasp,
    fixed_put_xyz_for_hand,
    humanlike_put_waypoints,
    position_for_home,
    put_outward_z_axis_orientation,
    smooth_bezier_arc_waypoints,
)


def assert_xyz_mirrored(
    left_xyz: tuple[float, float, float],
    right_xyz: tuple[float, float, float],
) -> None:
    assert left_xyz[0] == right_xyz[0]
    assert left_xyz[1] == -right_xyz[1]
    assert left_xyz[2] == right_xyz[2]


def test_default_put_targets_are_mirrored() -> None:
    assert_xyz_mirrored(
        fixed_put_xyz_for_hand("left"),
        fixed_put_xyz_for_hand("right"),
    )


def test_object_put_targets_are_mirrored() -> None:
    for object_type in ("yellow_cube", "yellow_duck", "blue_cube"):
        assert_xyz_mirrored(
            fixed_put_xyz_for_hand("left", object_type),
            fixed_put_xyz_for_hand("right", object_type),
        )


def test_put_waypoints_are_mirrored_from_mirrored_starts() -> None:
    args = Namespace(home_safe_z_m=0.95)
    orientation = (0.0, 0.0, 0.0, 1.0)
    left_publisher = FakePublisher(
        (np.array([0.20, 0.10, 0.82]), orientation),
    )
    right_publisher = FakePublisher(
        (np.array([0.20, -0.10, 0.82]), orientation),
    )

    left_waypoints = humanlike_put_waypoints(
        left_publisher,
        "left",
        np.array(fixed_put_xyz_for_hand("left")),
        orientation,
        args,
    )
    right_waypoints = humanlike_put_waypoints(
        right_publisher,
        "right",
        np.array(fixed_put_xyz_for_hand("right")),
        orientation,
        args,
    )

    assert len(left_waypoints) == len(right_waypoints)
    for (left_position, _), (right_position, _) in zip(left_waypoints, right_waypoints):
        np.testing.assert_allclose(
            left_position,
            np.array([right_position[0], -right_position[1], right_position[2]]),
        )


def test_put_waypoints_form_one_smooth_bezier_arc() -> None:
    args = Namespace(home_safe_z_m=0.95)
    orientation = (0.0, 0.0, 0.0, 1.0)
    start = np.array([0.20, -0.10, 0.82])
    end = np.array(fixed_put_xyz_for_hand("right"))
    publisher = FakePublisher((start, orientation))

    waypoints = humanlike_put_waypoints(
        publisher, "right", end, orientation, args,
    )

    assert len(waypoints) >= 16
    np.testing.assert_allclose(waypoints[-1][0], end)
    # The curve leaves upward and arrives downward instead of turning at a
    # lift/pre-put waypoint.
    assert waypoints[0][0][2] > start[2]
    assert waypoints[-2][0][2] > end[2]
    assert max(position[2] for position, _ in waypoints) >= 0.949
    sampled_positions = np.vstack((start, [position for position, _ in waypoints]))
    step_lengths = np.linalg.norm(np.diff(sampled_positions, axis=0), axis=1)
    assert np.std(step_lengths) / np.mean(step_lengths) < 0.01


def test_cubic_bezier_hits_both_endpoints() -> None:
    points = [np.array([float(i), 0.0, 0.0]) for i in range(4)]

    np.testing.assert_allclose(cubic_bezier_position(*points, 0.0), points[0])
    np.testing.assert_allclose(cubic_bezier_position(*points, 1.0), points[-1])


def test_non_lifting_bezier_grasp_path_stays_between_endpoint_heights() -> None:
    start = np.array([0.25, -0.25, 0.81])
    end = np.array([0.42, -0.18, 0.70])
    waypoints = smooth_bezier_arc_waypoints(
        start,
        (0.0, 0.0, 0.0, 1.0),
        end,
        (0.0, 0.0, 0.0, 1.0),
        Namespace(),
        lift_arc=False,
    )

    z_values = np.asarray([position[2] for position, _ in waypoints])
    assert np.all(z_values <= start[2] + 1e-12)
    assert np.all(z_values >= end[2] - 1e-12)
    assert np.all(np.diff(z_values) <= 1e-12)
    positions = np.asarray([position for position, _ in waypoints])
    half_height = 0.5 * (start[2] + end[2])
    half_height_index = int(np.argmin(np.abs(positions[:, 2] - half_height)))
    lateral_total = float(np.linalg.norm(end[:2] - start[:2]))
    lateral_remaining = float(
        np.linalg.norm(end[:2] - positions[half_height_index, :2])
    )
    # By the time the gripper enters the lower half of the descent it is
    # already nearly above the object, rather than sweeping sideways into it.
    assert lateral_remaining < 0.2 * lateral_total


def test_home_targets_are_mirrored() -> None:
    args = Namespace(
        left_home_xyz=(0.25, 0.25, 0.81),
        right_home_xyz=(0.25, -0.25, 0.81),
    )

    assert_xyz_mirrored(
        position_for_home("left", args),
        position_for_home("right", args),
    )


def test_put_orientation_is_mirrored_between_hands() -> None:
    args = Namespace(
        ik_orientation_quat=(0.0, 0.0, 0.0, 1.0),
        ik_downward_tilt_deg=0.0,
        ik_downward_tilt_left_deg=-45.0,
        ik_downward_tilt_right_deg=45.0,
        ik_downward_tilt_axis="z",
        ik_downward_tilt_y_deg=0.0,
        ik_downward_tilt_y_left_deg=45.0,
        ik_downward_tilt_y_right_deg=45.0,
        ik_downward_tilt_frame="local",
    )
    right_orientation = put_outward_z_axis_orientation(
        "right",
        ik_wrist_orientation_quat(args, hand="right"),
    )
    left_orientation = put_outward_z_axis_orientation(
        "left",
        ik_wrist_orientation_quat(args, hand="left"),
    )
    right_rotation = quaternion_to_rotation_matrix(right_orientation)
    left_rotation = quaternion_to_rotation_matrix(left_orientation)
    mirror_y = np.diag([1.0, -1.0, 1.0])

    np.testing.assert_allclose(left_rotation, mirror_y @ right_rotation @ mirror_y)


def test_keep_put_pose_true_skips_home(monkeypatch) -> None:
    publisher = FakePublisher(
        (np.array([0.30, -0.20, 0.85]), (0.0, 0.0, 0.0, 1.0)),
    )
    calls = {"home": 0}
    monkeypatch.setattr("grasp_core.tasks.put.publish_request_ik_path", lambda *a, **k: 1)
    monkeypatch.setattr("grasp_core.tasks.put.send_gripper_signal", lambda *a, **k: "release ok")

    def fake_home(*args, **kwargs):
        calls["home"] += 1
        return "home ok"

    monkeypatch.setattr("grasp_core.tasks.put.publish_home_request_ik_target", fake_home)

    result = execute_fixed_put_after_grasp(
        publisher,
        "right",
        put_args(),
        grasp_confirmed=True,
        keep_put_pose=True,
    )

    assert result.ok
    assert "kept at put pose" in result.status
    assert calls["home"] == 0


def test_keep_put_pose_false_returns_home(monkeypatch) -> None:
    publisher = FakePublisher(
        (np.array([0.30, -0.20, 0.85]), (0.0, 0.0, 0.0, 1.0)),
    )
    calls = {"home": 0}
    monkeypatch.setattr("grasp_core.tasks.put.publish_request_ik_path", lambda *a, **k: 1)
    monkeypatch.setattr("grasp_core.tasks.put.send_gripper_signal", lambda *a, **k: "release ok")

    def fake_home(*args, **kwargs):
        calls["home"] += 1
        return "home ok"

    monkeypatch.setattr("grasp_core.tasks.put.publish_home_request_ik_target", fake_home)

    result = execute_fixed_put_after_grasp(
        publisher,
        "right",
        put_args(),
        grasp_confirmed=True,
        keep_put_pose=False,
    )

    assert result.ok
    assert result.status.endswith("home ok")
    assert calls["home"] == 1


def put_args() -> Namespace:
    return Namespace(
        home_safe_z_m=0.95,
        put_target_hold_sec=0.0,
        put_home_hold_sec=0.0,
        left_home_xyz=(0.25, 0.25, 0.81),
        right_home_xyz=(0.25, -0.25, 0.81),
        ik_orientation_quat=(0.0, 0.0, 0.0, 1.0),
        ik_downward_tilt_deg=0.0,
        ik_downward_tilt_left_deg=None,
        ik_downward_tilt_right_deg=None,
        ik_downward_tilt_axis="y",
        ik_downward_tilt_y_deg=0.0,
        ik_downward_tilt_y_left_deg=None,
        ik_downward_tilt_y_right_deg=None,
        ik_downward_tilt_frame="local",
    )


class FakePublisher:
    def __init__(
        self,
        remembered: tuple[np.ndarray, tuple[float, float, float, float]],
    ) -> None:
        self._remembered = remembered

    def remembered_target(
        self,
        hand: str,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        return self._remembered

    def stop_requested(self) -> bool:
        return False
