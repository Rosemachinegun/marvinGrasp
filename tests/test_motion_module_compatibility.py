import numpy as np

from grasp_core.planning.trajectory.planner import (
    plan_pose_path,
    plan_pose_target,
    plan_trajectory,
)


def test_unified_planner_matches_path_and_target_entry_points() -> None:
    start_position = np.array([0.1, -0.2, 0.8])
    start_orientation = (0.0, 0.0, 0.0, 1.0)
    waypoints = [
        (np.array([0.2, -0.1, 0.9]), (0.0, 0.0, 0.0, 1.0)),
        (np.array([0.4, -0.3, 0.7]), (0.0, 0.0, 1.0, 0.0)),
    ]
    kwargs = {"max_step_m": 0.01, "max_step_deg": 3.0, "min_steps": 2}

    unified = plan_trajectory(start_position, start_orientation, waypoints, **kwargs)
    path_api = plan_pose_path(start_position, start_orientation, waypoints, **kwargs)
    target_api = plan_pose_target(
        start_position,
        start_orientation,
        waypoints[-1][0],
        waypoints[-1][1],
        **kwargs,
    )
    unified_target = plan_trajectory(
        start_position,
        start_orientation,
        [waypoints[-1]],
        **kwargs,
    )

    assert unified.segment_steps == path_api.segment_steps
    assert unified_target.segment_steps == target_api.segment_steps
    for actual, expected in zip(unified.samples, path_api.samples, strict=True):
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
    for actual, expected in zip(
        unified_target.samples,
        target_api.samples,
        strict=True,
    ):
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
