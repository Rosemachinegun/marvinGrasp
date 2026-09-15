from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from grasp_core.execution import robot_skill_service as robot_actions
from grasp_core.communication.request_ik_transport import Ros2PoseTargetPublisher


@pytest.mark.parametrize("hand", ["left", "right"])
def test_interrupted_home_passes_measured_start_and_normal_home_does_not(hand, monkeypatch):
    measured = (np.array([0.1, 0.2, 0.3]), (0., 0., 0., 1.))
    publisher = Mock()
    publisher.stop_requested.return_value = False
    publisher.client.wait_for_settled_tool_pose.return_value = measured
    publish = Mock(return_value="done")
    monkeypatch.setattr("grasp_core.execution.skills.home.publish_home", publish)
    service = robot_actions.RobotActionService(
        args=SimpleNamespace(left_home_xyz=(0, 1, 0), right_home_xyz=(0, -1, 0)),
        ik_publisher=publisher, pick_templates={},
    )
    service.publish_home(hand)
    publisher.client.wait_for_settled_tool_pose.assert_not_called()
    assert publish.call_args.kwargs["start_pose"] is None
    service.publish_home(hand, fresh_measured_start=True)
    assert publish.call_args.kwargs["start_pose"] is measured
    publisher.remembered_target.assert_not_called()
    publish.reset_mock()
    publisher.client.wait_for_settled_tool_pose.side_effect = RuntimeError("not settled")
    with pytest.raises(RuntimeError, match="not settled"):
        service.publish_home(hand, fresh_measured_start=True)
    publish.assert_not_called()


from concurrent.futures import Future
import threading
from pathlib import Path

from grasp_core.tools.flowpose_request_ik_app import GraspDemoApp, RuntimeState, PipelineStage
from grasp_core.communication.request_ik_feedback import (
    MeasuredToolFK,
    SettledFeedback,
    _package_share,
    _running_colcon_install,
)
from grasp_core.execution.motion_executor import RequestIkTargetPublisher
from grasp_core.execution.motion_executor import trajectory_sample_periods
from grasp_core.core.math.pose import quaternion_angle_rad
from grasp_core.planning.trajectory.planner import plan_trajectory


Q = (0., 0., 0., 1.)


def test_stability_requires_stationary_joints_even_when_tool_does_not_move():
    check = SettledFeedback(0)
    pose = (np.zeros(3), Q)
    for i in range(1, 10):
        # Null-space joint motion can leave the tool still; it is not settled.
        assert not check.update(i * 10**8, i * 10**8, np.zeros(7), np.ones(7)*0.02, pose)
    for i in range(10, 15):
        assert not check.update(i * 10**8, i * 10**8, np.zeros(7), np.zeros(7), pose)
    assert check.update(15 * 10**8, 15 * 10**8, np.zeros(7), np.zeros(7), pose)


def test_frozen_stale_and_gapped_feedback_cannot_establish_stability():
    check = SettledFeedback(10**9)
    pose = (np.zeros(3), Q)
    for now in range(11, 20):
        assert not check.update(10**9, now * 10**8, np.zeros(7), np.zeros(7), pose)
    assert not check.update(2*10**9, 2*10**9, np.zeros(7), np.zeros(7), pose)
    # A feedback gap resets the interval rather than counting as standstill.
    assert not check.update(3*10**9, 3*10**9, np.zeros(7), np.zeros(7), pose)


def test_package_share_uses_running_install_without_ament_overlay(tmp_path):
    install = tmp_path / "install"
    share = install / "marvin_description" / "share" / "marvin_description"
    share.mkdir(parents=True)
    assert _package_share("marvin_description", install_root=install) == share


def test_running_install_is_derived_from_controller_executable(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    cmdline = proc / "123" / "cmdline"
    cmdline.parent.mkdir(parents=True)
    executable = (
        tmp_path / "ws" / "install" / "marvin_qp_controller" / "lib"
        / "marvin_qp_controller" / "pose_stamped_request_ik_tester.py"
    )
    cmdline.write_bytes(b"python3\0" + str(executable).encode() + b"\0")
    class ProcPath(type(Path())):
        def __new__(cls, *values):
            if values == ("/proc",):
                values = (proc,)
            return super().__new__(cls, *values)

    monkeypatch.setattr("grasp_core.communication.request_ik_feedback.Path", ProcPath)
    assert _running_colcon_install("marvin_qp_controller") == tmp_path / "ws" / "install"


def test_actual_first_home_command_is_measured_pose_not_cached_endpoint(monkeypatch):
    pub = RequestIkTargetPublisher.__new__(RequestIkTargetPublisher)
    pub.client = Mock()
    pub.node = Mock()
    pub.publish_rate_hz = 50.
    pub.publish_sec = 0.
    pub.record_trajectories = False
    pub.record_joint_trajectory_csv = False
    pub._stop_event = threading.Event()
    pub._last_targets = {'left': (np.ones(3)*4, Q)}
    start = np.array([0.25, 0.25, 0.81])
    end = start + np.array([0.1, 0., 0.])
    pub.client.wait_for_current_tool_pose.return_value = (start, Q)
    sleeps = []
    monkeypatch.setattr(
        'grasp_core.execution.motion_executor.time.sleep', sleeps.append,
    )
    pub.publish_smooth_path(
        'left', [(end, Q)], max_step_m=.003, max_step_deg=1., final_hold_sec=0.0,
    )
    samples = [(c.args[1], c.args[2]) for c in pub.client.publish_pose.call_args_list]
    expected = plan_trajectory(
        start, Q, [(end, Q)], max_step_m=.003, max_step_deg=1., min_steps=1,
    )
    np.testing.assert_allclose(samples[0][0], expected.samples[0][0])
    np.testing.assert_allclose(samples[-1][0], end)
    assert 0.0 < sleeps[0] <= 0.02
    for (p0, q0), (p1, q1) in zip(samples, samples[1:]):
        assert np.linalg.norm(p1-p0) <= .003 + 1e-9
        assert quaternion_angle_rad(q0, q1) <= np.deg2rad(1.) + 1e-9


def test_upper_layer_pose_loop_preserves_path_endpoint(monkeypatch):
    pub = RequestIkTargetPublisher.__new__(RequestIkTargetPublisher)
    pub.client = Mock()
    pub.node = Mock()
    pub.frame_id = "base_link"
    pub.publish_rate_hz = 50.0
    pub.publish_sec = 0.0
    pub.record_trajectories = False
    pub.record_joint_trajectory_csv = False
    pub._stop_event = threading.Event()
    pub._last_targets = {}
    pub.client.topic_for_hand.return_value = "/left_target"
    pub.client.ok.return_value = True
    monkeypatch.setattr("grasp_core.execution.motion_executor.time.sleep", Mock())

    start = np.array([0.0, 0.0, 0.0])
    boundary = np.array([0.03, 0.0, 0.0])
    end = np.array([0.06, 0.01, 0.0])
    pub.client.wait_for_current_tool_pose.return_value = (boundary, Q)
    pub.publish_smooth_path(
        "left",
        [(boundary, Q), (end, Q)],
        start_position_xyz=start,
        start_orientation_xyzw=Q,
        max_step_m=0.01,
        on_after_waypoint={0: lambda *_args: 0},
        final_hold_sec=0.0,
    )

    calls = pub.client.publish_pose.call_args_list
    expected = plan_trajectory(
        start,
        Q,
        [(boundary, Q), (end, Q)],
        max_step_m=0.01,
        max_step_deg=3.0,
        min_steps=1,
    )
    np.testing.assert_allclose(calls[0].args[1], expected.samples[0][0])
    np.testing.assert_allclose(calls[-1].args[1], end)


def test_interrupted_path_remembers_last_actually_published_pose(monkeypatch):
    pub = RequestIkTargetPublisher.__new__(RequestIkTargetPublisher)
    pub.client = Mock()
    pub.node = Mock()
    pub.frame_id = "base_link"
    pub.publish_rate_hz = 200.0
    pub.publish_sec = 0.0
    pub.record_trajectories = False
    pub.record_joint_trajectory_csv = False
    pub._stop_event = threading.Event()
    pub._last_targets = {}
    pub.client.ok.return_value = True
    pub.client.topic_for_hand.return_value = "/control/request_ik_tester/target_poseL"
    sent = []

    def publish(_hand, position, orientation):
        sent.append((position.copy(), orientation))
        if len(sent) == 2:
            pub._stop_event.set()

    pub.client.publish_pose.side_effect = publish
    monkeypatch.setattr("grasp_core.execution.motion_executor.time.sleep", Mock())
    pub.publish_smooth_path(
        "left",
        [(np.array([0.1, 0.0, 0.0]), Q)],
        start_position_xyz=np.zeros(3),
        start_orientation_xyzw=Q,
        max_step_m=0.005,
        final_hold_sec=0.0,
    )

    remembered = pub.remembered_target("left")
    assert remembered is not None
    np.testing.assert_allclose(remembered[0], sent[-1][0])
    assert remembered[1] == sent[-1][1]


def test_interrupted_home_timing_holds_then_accelerates_smoothly():
    periods = trajectory_sample_periods(
        40, .02, startup_slowdown=True, terminal_slowdown=False,
    )
    assert periods[0] == pytest.approx(.02)
    ramp = periods[1:17]
    assert ramp[0] > periods[0]
    assert all(current >= following for current, following in zip(ramp, ramp[1:]))
    assert ramp[-1] == pytest.approx(.02)
    assert periods[17:] == pytest.approx([.02] * 23)


def test_normal_home_timing_is_unchanged():
    assert trajectory_sample_periods(
        20, .02, startup_slowdown=False, terminal_slowdown=False,
    ) == pytest.approx([.02] * 20)


def make_app():
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState()
    app.grasp_future = None
    app.gripper_future = None
    app.manual_home_future = None
    app.manual_home_hand = None
    app.gripper_interrupted = False
    app.ik_publisher = Mock()
    stopped = threading.Event()
    app.ik_publisher.request_stop.side_effect = stopped.set
    app.ik_publisher.clear_stop.side_effect = stopped.clear
    app.ik_publisher.stop_requested.side_effect = stopped.is_set
    app.robot_actions = Mock()
    app.action_executor = Mock()
    app.action_executor.submit.return_value = Future()
    return app


def test_resume_does_not_restart_pipeline_or_publish_and_home_is_async():
    app = make_app()
    app.state.pipeline_stage = PipelineStage.FLOWPOSE
    app.handle_key(ord('s'), None)
    assert app.state.pipeline_stage is PipelineStage.IDLE
    app.handle_key(ord('s'), None)
    app.robot_actions.publish_home.assert_not_called()
    assert app.state.home_needs_sync == {'left', 'right'}
    app.handle_key(ord('j'), None)
    app.action_executor.submit.assert_not_called()
    app.handle_key(ord('h'), None)
    submitted = app.action_executor.submit.call_args.args
    assert submitted[1] == ('left', 'right')
    assert submitted[2] == frozenset({'left', 'right'})
    # S is serviced while HOME is running, and resume cannot resurrect it.
    app.handle_key(ord('s'), None)
    app.handle_key(ord('s'), None)
    assert app.state.paused
    app.manual_home_future.set_result('interrupted')
    app.collect_manual_home_result()
    assert app.state.home_needs_sync == {'left', 'right'}


def test_gripper_result_after_pause_cannot_trigger_automatic_motion():
    app = make_app()
    app.gripper_future = Future()
    app.gripper_future_command = 'grip'
    app.gripper_future_hand = 'left'
    app.auto_put_after_confirmed_grasp = Mock()
    app.start_grip_failure_recovery = Mock()
    app.handle_key(ord('s'), None)
    app.handle_key(ord('s'), None)
    app.gripper_future.set_result('GRIP_SUCCESS hand=left')
    app._collect_gripper_result()
    app.auto_put_after_confirmed_grasp.assert_not_called()
    app.start_grip_failure_recovery.assert_not_called()


def test_completed_grasp_at_resume_is_consumed_with_stop_latched():
    app = make_app()
    app.grasp_future = Future()
    app.auto_put_after_confirmed_grasp = Mock()
    app.handle_key(ord('s'), None)
    app.grasp_future.set_result(SimpleNamespace(status='done', grasp_confirmed=True))
    app.handle_key(ord('s'), None)
    app.auto_put_after_confirmed_grasp.assert_not_called()
    assert app.grasp_future is None


def test_measured_fk_uses_ik_tool_offset_instead_of_urdf_offset(tmp_path):
    pytest.importorskip('pinocchio')
    scene = tmp_path / 'ik.xml'
    scene.write_text('''<mujoco><compiler angle="radian"/><worldbody>
      <body name="base_link"><body name="arm">
        <joint name="Joint1_L" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <body name="left_tool" pos="0 0 .145"/>
      </body></body></worldbody></mujoco>''')
    fk = MeasuredToolFK(scene, 'base_link', 'left')
    msg = SimpleNamespace(name=['Joint1_L'], position=[0.7], velocity=[0.])
    _, _, pose = fk.evaluate(msg)
    np.testing.assert_allclose(pose[0], [0., 0., .145])
    assert np.linalg.norm(pose[0] - [0., 0., .195]) == pytest.approx(.05)
    msg.name = ['unknown_joint']
    with pytest.raises(RuntimeError, match='missing IK joint'):
        fk.evaluate(msg)
