from argparse import Namespace

import numpy as np

from grasp_core.execution.motion_executor import (
    terminal_sample_periods as legacy_terminal_sample_periods,
    trajectory_sample_periods as legacy_trajectory_sample_periods,
)
from grasp_core.planning.trajectory.constraints import effective_trajectory_step_limits
from grasp_core.planning.trajectory.interpolation import cubic_bezier_position, smootherstep
from grasp_core.planning.trajectory.timing import terminal_sample_periods, trajectory_sample_periods
from grasp_core.planning.trajectory.planner import (
    effective_trajectory_step_limits as legacy_step_limits,
    plan_pose_path,
    plan_pose_target,
    plan_trajectory,
    smooth_bezier_arc_waypoints,
)
from grasp_core.execution.skills.place import (
    cubic_bezier_position as legacy_cubic_bezier_position,
    smootherstep as legacy_smootherstep,
    smooth_bezier_arc_waypoints as legacy_smooth_bezier_arc_waypoints,
)


def test_interpolation_compatibility_exports_are_numerically_equal() -> None:
    points = tuple(np.array([index, index**2, -index], dtype=np.float64) for index in range(4))
    for alpha in (-0.2, 0.0, 0.25, 0.75, 1.0, 1.2):
        np.testing.assert_array_equal(
            legacy_cubic_bezier_position(*points, alpha),
            cubic_bezier_position(*points, alpha),
        )
        assert legacy_smootherstep(alpha) == smootherstep(alpha)


def test_timing_compatibility_exports_are_equal() -> None:
    assert legacy_terminal_sample_periods(12, 0.02, enabled=True) == (
        terminal_sample_periods(12, 0.02, enabled=True)
    )


def test_bezier_waypoint_compatibility_export_is_equal() -> None:
    args = Namespace(home_safe_z_m=0.9)
    positional = (
        np.array([0.2, -0.1, 0.8]),
        (0.0, 0.0, 0.0, 1.0),
        np.array([0.4, -0.3, 0.75]),
        (0.0, 0.0, 1.0, 0.0),
        args,
    )
    legacy = legacy_smooth_bezier_arc_waypoints(*positional, ease_orientation=True)
    moved = smooth_bezier_arc_waypoints(*positional, ease_orientation=True)
    assert len(legacy) == len(moved)
    for actual, expected in zip(legacy, moved, strict=True):
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
    assert legacy_trajectory_sample_periods(
        24,
        0.02,
        startup_slowdown=True,
        terminal_slowdown=True,
    ) == trajectory_sample_periods(
        24,
        0.02,
        startup_slowdown=True,
        terminal_slowdown=True,
    )


def test_constraint_compatibility_export_is_equal() -> None:
    args = Namespace(
        target_publish_rate_hz=50.0,
        target_trajectory_step_m=0.02,
        target_trajectory_step_deg=4.0,
        target_trajectory_speed_mps=0.3,
        target_trajectory_angular_speed_dps=90.0,
    )
    assert legacy_step_limits(args) == effective_trajectory_step_limits(args)


def test_unified_planner_matches_legacy_path_and_target_entry_points() -> None:
    start_position = np.array([0.1, -0.2, 0.8])
    start_orientation = (0.0, 0.0, 0.0, 1.0)
    waypoints = [
        (np.array([0.2, -0.1, 0.9]), (0.0, 0.0, 0.0, 1.0)),
        (np.array([0.4, -0.3, 0.7]), (0.0, 0.0, 1.0, 0.0)),
    ]
    kwargs = {"max_step_m": 0.01, "max_step_deg": 3.0, "min_steps": 2}

    unified = plan_trajectory(start_position, start_orientation, waypoints, **kwargs)
    legacy_path = plan_pose_path(start_position, start_orientation, waypoints, **kwargs)
    legacy_target = plan_pose_target(
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

    assert unified.segment_steps == legacy_path.segment_steps
    assert unified_target.segment_steps == legacy_target.segment_steps
    for actual, expected in zip(unified.samples, legacy_path.samples, strict=True):
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
    for actual, expected in zip(
        unified_target.samples,
        legacy_target.samples,
        strict=True,
    ):
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
