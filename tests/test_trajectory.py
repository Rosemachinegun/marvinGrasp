import pytest
import numpy as np

from grasp_core.execution.motion_executor import terminal_sample_periods
from grasp_core.planning.trajectory.timing import trajectory_linear_velocities
from grasp_core.planning.trajectory.interpolation import startup_ramp_alpha
from grasp_core.planning.trajectory.interpolation import (
    cubic_bezier_from_tangents,
    position_tangents,
)
from grasp_core.planning.trajectory.planner import plan_pose_path
from grasp_core.core.math.pose import quaternion_angle_rad


IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def test_startup_ramp_starts_at_rest_and_rejoins_unit_rate() -> None:
    ramp = 0.25
    epsilon = 1e-6
    assert startup_ramp_alpha(0.0, ramp) == 0.0
    assert startup_ramp_alpha(epsilon, ramp) / epsilon < 1e-8
    assert startup_ramp_alpha(ramp, ramp) == pytest.approx(ramp)
    left_slope = (
        startup_ramp_alpha(ramp, ramp)
        - startup_ramp_alpha(ramp - epsilon, ramp)
    ) / epsilon
    assert left_slope == pytest.approx(1.0, rel=1e-4)


def test_bezier_tangent_form_matches_cubic_hermite_polynomial() -> None:
    start = np.array([0.1, -0.2, 0.8])
    end = np.array([0.4, 0.3, 0.7])
    start_tangent = np.array([0.2, 0.1, 0.0])
    end_tangent = np.array([-0.1, 0.2, 0.1])
    for alpha in np.linspace(0.0, 1.0, 21):
        t2 = alpha * alpha
        t3 = t2 * alpha
        expected = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * start
            + (t3 - 2.0 * t2 + alpha) * start_tangent
            + (-2.0 * t3 + 3.0 * t2) * end
            + (t3 - t2) * end_tangent
        )
        np.testing.assert_allclose(
            cubic_bezier_from_tangents(
                start,
                end,
                start_tangent,
                end_tangent,
                alpha,
            ),
            expected,
            atol=1e-15,
        )


def test_position_tangents_preserve_path_shape_at_endpoints() -> None:
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.0, 0.0]),
        np.array([0.2, 0.1, 0.0]),
    ]

    tangents = position_tangents(positions)

    np.testing.assert_allclose(tangents[0], positions[1] - positions[0])
    np.testing.assert_allclose(tangents[-1], positions[-1] - positions[-2])
    assert np.linalg.norm(tangents[1]) > 0.0


def test_plan_pose_path_keeps_final_sample_on_target() -> None:
    target = np.array([0.2, 0.1, 0.0])
    plan = plan_pose_path(
        np.array([0.0, 0.0, 0.0]),
        IDENTITY_QUAT,
        [
            (np.array([0.1, 0.0, 0.0]), IDENTITY_QUAT),
            (target, IDENTITY_QUAT),
        ],
        max_step_m=0.02,
        max_step_deg=5.0,
        min_steps=5,
    )

    assert len(plan.samples) == sum(plan.segment_steps)
    np.testing.assert_allclose(plan.samples[0][0], np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(plan.samples[-1][0], target)


def test_plan_pose_path_first_sample_keeps_exact_initial_orientation() -> None:
    initial = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
    target = (0.0, 0.0, -0.7071067811865476, 0.7071067811865476)
    plan = plan_pose_path(
        np.zeros(3),
        initial,
        [(np.array([0.1, 0.0, 0.0]), target)],
        max_step_m=0.01,
        max_step_deg=2.0,
        min_steps=15,
    )
    assert plan.samples[0][1] == initial


def test_orientation_interpolation_has_zero_velocity_startup_ramp() -> None:
    initial = (0.0, 0.0, 0.0, 1.0)
    target = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
    plan = plan_pose_path(
        np.zeros(3),
        initial,
        [(np.array([0.1, 0.0, 0.0]), target)],
        max_step_m=0.01,
        max_step_deg=5.0,
        min_steps=30,
    )
    angular_steps = [
        quaternion_angle_rad(q0, q1)
        for (_p0, q0), (_p1, q1) in zip(plan.samples, plan.samples[1:])
    ]
    assert angular_steps[0] < angular_steps[1] < angular_steps[2]
    assert angular_steps[0] < angular_steps[-1]


def test_plan_pose_path_limits_actual_curve_step_size() -> None:
    max_step_m = 0.003
    plan = plan_pose_path(
        np.array([0.0, 0.0, 0.0]),
        IDENTITY_QUAT,
        [
            (np.array([0.08, 0.0, 0.0]), IDENTITY_QUAT),
            (np.array([0.08, 0.08, 0.0]), IDENTITY_QUAT),
            (np.array([0.14, 0.08, 0.0]), IDENTITY_QUAT),
        ],
        max_step_m=max_step_m,
        max_step_deg=5.0,
        min_steps=2,
    )

    previous = np.array([0.0, 0.0, 0.0])
    for position, _orientation in plan.samples:
        assert np.linalg.norm(position - previous) <= max_step_m * 1.001
        previous = position


def test_trajectory_linear_velocities_use_smooth_boundaries() -> None:
    velocities = trajectory_linear_velocities(
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.01, 0.0, 0.0]),
            np.array([0.02, 0.0, 0.0]),
        ],
        [0.1, 0.1, 0.1],
    )

    np.testing.assert_allclose(velocities[0], np.zeros(3))
    np.testing.assert_allclose(velocities[1], np.array([0.1, 0.0, 0.0]))
    np.testing.assert_allclose(velocities[2], np.zeros(3))


def test_terminal_sample_periods_only_changes_timing_tail() -> None:
    periods = terminal_sample_periods(
        10,
        0.01,
        enabled=True,
        tail_count=4,
        max_scale=3.0,
    )

    assert periods[:6] == pytest.approx([0.01] * 6)
    assert periods[-1] == pytest.approx(0.03)
    assert periods[-4:] == sorted(periods[-4:])


def test_terminal_sample_periods_disabled_is_uniform() -> None:
    assert terminal_sample_periods(5, 0.01, enabled=False) == pytest.approx(
        [0.01] * 5
    )
