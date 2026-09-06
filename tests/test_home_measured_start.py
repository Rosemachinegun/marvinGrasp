from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from grasp_core.tasks import robot_actions
from grasp_core.communication.ros2_pose_publisher import Ros2PoseTargetPublisher


@pytest.mark.parametrize("hand", ["left", "right"])
def test_interrupted_home_passes_measured_start_and_normal_home_does_not(hand, monkeypatch):
    measured = (np.array([0.1, 0.2, 0.3]), (0., 0., 0., 1.))
    publisher = Mock()
    publisher.stop_requested.return_value = False
    publisher.client.wait_for_settled_tool_pose.return_value = measured
    publish = Mock(return_value="done")
    monkeypatch.setattr(robot_actions, "publish_home_request_ik_target", publish)
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

from grasp_core.apps.flowpose_request_ik_app import GraspDemoApp, RuntimeState, PipelineStage
from grasp_core.communication.measured_home_pose import (
    MeasuredToolFK,
    SettledFeedback,
    _package_share,
    _running_colcon_install,
)
from grasp_core.communication.request_ik_publisher import RequestIkTargetPublisher
from grasp_core.core.pose_math import quaternion_angle_rad


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

    monkeypatch.setattr("grasp_core.communication.measured_home_pose.Path", ProcPath)
    assert _running_colcon_install("marvin_qp_controller") == tmp_path / "ws" / "install"


@pytest.mark.parametrize('mode', ['pose_stream', 'cartesian_trajectory'])
def test_actual_first_home_command_is_measured_pose_not_cached_endpoint(mode, monkeypatch):
    pub = RequestIkTargetPublisher.__new__(RequestIkTargetPublisher)
    pub.client = Mock()
    pub.node = Mock()
    pub.publish_rate_hz = 50.
    pub.publish_sec = 0.
    pub.command_mode = mode
    pub.record_trajectories = False
    pub.record_joint_trajectory_csv = False
    pub.left_trajectory_joint_name = 'left_tcp'
    pub._stop_event = threading.Event()
    pub._last_targets = {'left': (np.ones(3)*4, Q)}
    start = np.array([0.25, 0.25, 0.81])
    end = start + np.array([0.1, 0., 0.])
    monkeypatch.setattr('grasp_core.communication.request_ik_publisher.time.sleep', lambda _: None)
    pub.publish_smooth_target('left', end, Q, start_position_xyz=start,
                              start_orientation_xyzw=(0., 0., 0., -1.),
                              max_step_m=.003, max_step_deg=1., include_start=True)
    if mode == 'pose_stream':
        samples = [(c.args[1], c.args[2]) for c in pub.client.publish_pose.call_args_list]
    else:
        samples = pub.client.publish_pose_trajectory.call_args.args[1]
    np.testing.assert_allclose(samples[0][0], start)
    np.testing.assert_allclose(samples[-1][0], end)
    for (p0, q0), (p1, q1) in zip(samples, samples[1:]):
        assert np.linalg.norm(p1-p0) <= .003 + 1e-9
        assert quaternion_angle_rad(q0, q1) <= np.deg2rad(1.) + 1e-9


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
    app.handle_key(ord('h'), None)
    assert app.action_executor.submit.call_args.kwargs['fresh_measured_start'] is True
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
