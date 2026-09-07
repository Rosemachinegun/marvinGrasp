"""Exercise automatic drop recovery through the real HOME action service."""
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import numpy as np
import pytest

from grasp_core.apps.flowpose_request_ik_app import GraspDemoApp, RuntimeState
from grasp_core.communication.request_ik_publisher import RequestIkTargetPublisher
from grasp_core.tasks import robot_actions


class QueuedExecutor:
    def submit(self, fn, *args, **kwargs):
        self.job = (fn, args, kwargs)
        self.future = Future()
        return self.future

    def run(self):
        fn, args, kwargs = self.job
        try:
            self.future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            self.future.set_exception(exc)


def app_for_drop(monkeypatch):
    publisher = RequestIkTargetPublisher.__new__(RequestIkTargetPublisher)
    publisher._stop_event = threading.Event()
    publisher._stop_lock = threading.Lock()
    publisher._stop_generation = 0
    publisher.node = Mock()
    publisher.client = Mock()
    args = SimpleNamespace(ik_hand="right", right_home_xyz=(.25, -.25, .81),
                           left_home_xyz=(.25, .25, .81),
                           ik_orientation_quat=(0., 0., 0., 1.))
    service = robot_actions.RobotActionService(
        args=args, ik_publisher=publisher, pick_templates={},
    )
    service.send_gripper = Mock(return_value="OK release")
    publish = Mock(return_value="home complete")
    monkeypatch.setattr(robot_actions, "publish_home_request_ik_target", publish)
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = args
    app.state = RuntimeState()
    app.grasp_future = app.manual_home_future = app.gripper_future = None
    app.ik_publisher = publisher
    app.robot_actions = service
    app.action_executor = QueuedExecutor()
    return app, publisher, publish


@pytest.mark.parametrize("hand", ["left", "right"])
def test_drop_keeps_stop_until_release_and_fresh_measured_start(monkeypatch, hand):
    app, publisher, publish = app_for_drop(monkeypatch)
    events = []
    measured = (np.array([.42, .1, .9]), (0., 0., 0., 1.))
    # Simulate a large commanded/measured mismatch after interrupted put.
    publisher._last_targets = {hand: (np.ones(3) * 4, (0., 0., 0., 1.))}

    def release(*args):
        assert publisher.stop_requested()
        events.append("release")
        return "OK release"

    def read(hand, *, cancelled):
        assert publisher.stop_requested()
        assert not cancelled()
        assert events == ["release"]
        events.append("measured after release")
        return measured

    app.robot_actions.send_gripper.side_effect = release
    publisher.client.wait_for_settled_tool_pose.side_effect = read
    app.state.home_needs_sync.add("right" if hand == "left" else "left")
    app.start_drop_recovery("GRASP_DROPPED", hand)
    assert publisher.stop_requested()
    publish.assert_not_called()
    app.action_executor.run()
    assert not publisher.stop_requested()
    assert publish.call_args.kwargs["start_pose"] is measured
    expected = (0.45, 0.20, 0.75) if hand == "left" else (0.45, -0.20, 0.75)
    assert publish.call_args.args[2] == expected
    app.update_drop_recovery()
    assert not app.state.paused
    assert app.state.drop_regrasp_pending
    assert app.state.home_needs_sync == {"right" if hand == "left" else "left"}


@pytest.mark.parametrize("cancel_during_read", [False, True])
def test_new_s_cannot_be_cleared_by_drop_home(monkeypatch, cancel_during_read):
    app, publisher, publish = app_for_drop(monkeypatch)
    app.start_drop_recovery("GRASP_DROPPED", "right")

    def read(hand, *, cancelled):
        app.handle_key(ord("s"), None)
        assert cancelled()
        # Even if cancellation arrives just as the reader returns, the clear
        # must reject the old generation and HOME must not publish.
        return np.zeros(3), (0., 0., 0., 1.)

    publisher.client.wait_for_settled_tool_pose.side_effect = read
    if not cancel_during_read:
        app.handle_key(ord("s"), None)
    app.action_executor.run()
    app.update_drop_recovery()
    publish.assert_not_called()
    assert publisher.stop_requested()
    assert app.state.paused
    assert not app.state.drop_regrasp_pending
    assert "right" in app.state.home_needs_sync


@pytest.mark.parametrize("failure", ["release", "measured"])
def test_failed_drop_recovery_stays_stopped(monkeypatch, failure):
    app, publisher, publish = app_for_drop(monkeypatch)
    if failure == "release":
        app.robot_actions.send_gripper.return_value = "ERR release failed"
    else:
        publisher.client.wait_for_settled_tool_pose.side_effect = RuntimeError("arm moving")
    app.start_drop_recovery("GRASP_DROPPED", "left")
    app.action_executor.run()
    app.update_drop_recovery()
    publish.assert_not_called()
    assert publisher.stop_requested()
    assert app.state.paused
    assert not app.state.drop_regrasp_pending


def test_drop_drains_previous_actions_before_release_and_measurement(monkeypatch):
    app, publisher, publish = app_for_drop(monkeypatch)
    waiting = threading.Event()

    class OldAction(Future):
        def result(self, *args, **kwargs):
            waiting.set()
            return super().result(*args, **kwargs)

    old = OldAction()
    app.manual_home_future = old
    publisher.client.wait_for_settled_tool_pose.return_value = (np.zeros(3), (0., 0., 0., 1.))
    with ThreadPoolExecutor(max_workers=1) as executor:
        app.action_executor = executor
        app.start_drop_recovery("GRASP_DROPPED", "right")
        try:
            assert waiting.wait(timeout=2.)
            assert publisher.stop_requested()
            app.robot_actions.send_gripper.assert_not_called()
            publisher.client.wait_for_settled_tool_pose.assert_not_called()
        finally:
            old.set_result("stopped")
        app.state.drop_recovery_futures[0].result(timeout=2.)
    publish.assert_called_once()
