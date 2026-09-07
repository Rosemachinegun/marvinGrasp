from argparse import Namespace
from concurrent.futures import Future

import numpy as np
import pytest

from grasp_core.apps.flowpose_request_ik_app import GraspDemoApp, RetryStage
from grasp_core.tasks.grasp_request_ik import GripFailedMinLimit, execute_grip_at_pose
from grasp_core.core.robot_target_pose import TargetObjectPose


class ImmediateExecutor:
    def __init__(self) -> None:
        self.calls = []

    def submit(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


class FakeRobotActions:
    def __init__(self) -> None:
        self.calls = []

    def publish_failure_recovery(self, hand: str) -> str:
        self.calls.append(("recovery", hand))
        return f"{hand} recovery ok"

    def send_gripper(self, command: str, hand: str | None = None) -> str:
        self.calls.append((command, hand))
        return f"{hand} {command} ok"


def test_left_grip_failure_returns_home_even_when_retry_loop_disabled() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = Namespace(
        ik_hand="auto",
        grip_retry_loop=False,
        grip_retry_max_attempts=3,
    )
    app.state = app_state()
    app.robot_actions = FakeRobotActions()
    app.action_executor = ImmediateExecutor()

    app.start_grip_failure_recovery(
        "GRIP_FAILED_MIN_LIMIT hand=left: reached min",
        failed_hand="left",
    )

    assert app.state.retry_stage is RetryStage.RECOVERY
    assert app.state.retry_will_regrasp is False
    assert app.robot_actions.calls == [("recovery", "left"), ("release", "left")]
    assert "moving left to recovery" in app.state.status
    assert "retry loop disabled" in app.state.status


def test_manual_right_grip_failure_returns_home() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = Namespace(
        ik_hand="auto",
        grip_retry_loop=False,
        grip_retry_max_attempts=3,
    )
    app.state = app_state()
    app.robot_actions = FakeRobotActions()
    app.action_executor = ImmediateExecutor()
    app.gripper_future = Future()
    app.gripper_future.set_result(
        "Sent right gripper grip to 127.0.0.1:55661: "
        "OK GRASP_FAILED_MIN_LIMIT grip done exit_code=2"
    )
    app.gripper_future_command = "grip"
    app.gripper_future_hand = "right"

    app._collect_gripper_result()

    assert app.state.retry_stage is RetryStage.RECOVERY
    assert app.state.grasp_confirmed is False
    assert app.state.last_gripper_hand == "right"
    assert app.robot_actions.calls == [("recovery", "right"), ("release", "right")]
    assert "moving right to recovery" in app.state.status


def app_state():
    from grasp_core.apps.flowpose_request_ik_app import RuntimeState

    return RuntimeState()


class FakePublisher:
    publish_rate_hz = 100.0

    def uses_trajectory_command(self, hand: str) -> bool:
        return False

    def hold_target(self, hand, position, orientation, duration_sec):
        return 1


def test_grip_failed_min_limit_alias_raises_without_confirming(monkeypatch) -> None:
    confirmed = []
    args = Namespace(grip_settle_sec=0.0, grip_post_confirm_hold_sec=0.0)
    monkeypatch.setattr(
        "grasp_core.tasks.grasp_request_ik.gripper_receiver_args",
        lambda args, hand: [
            ("left", Namespace(grip_signal_port=55551, gripper_server="mock"))
        ],
    )
    monkeypatch.setattr(
        "grasp_core.tasks.grasp_request_ik.send_gripper_signal",
        lambda command, args, hand: "OK GRIP_FAILED_MIN_LIMIT grip done exit_code=2",
    )

    with pytest.raises(GripFailedMinLimit):
        execute_grip_at_pose(
            FakePublisher(),
            "left",
            np.array([0.1, 0.2, 0.3]),
            (0.0, 0.0, 0.0, 1.0),
            args,
            on_grip_confirmed=lambda hand, status: confirmed.append((hand, status)),
        )

    assert confirmed == []


def test_ribbon_accepts_min_limit_as_success(monkeypatch) -> None:
    confirmed = []
    args = Namespace(grip_settle_sec=0.0, grip_post_confirm_hold_sec=0.0)
    monkeypatch.setattr(
        "grasp_core.tasks.grasp_request_ik.gripper_receiver_args",
        lambda args, hand: [
            ("left", Namespace(grip_signal_port=55551, gripper_server="mock"))
        ],
    )
    monkeypatch.setattr(
        "grasp_core.tasks.grasp_request_ik.send_gripper_signal",
        lambda command, args, hand: "OK GRIP_FAILED_MIN_LIMIT grip done exit_code=2",
    )

    execute_grip_at_pose(
        FakePublisher(),
        "left",
        np.array([0.1, 0.2, 0.3]),
        (0.0, 0.0, 0.0, 1.0),
        args,
        assume_success=True,
        on_grip_confirmed=lambda hand, status: confirmed.append((hand, status)),
    )

    assert confirmed == [
        ("left", "OK GRIP_FAILED_MIN_LIMIT grip done exit_code=2")
    ]


def test_manual_ribbon_grip_skips_failure_recovery() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = Namespace(ik_hand="auto")
    app.state = app_state()
    identity = np.eye(4)
    app.state.base_targets = [
        TargetObjectPose("ribbon_1", "ribbon_1", identity, identity)
    ]
    app.gripper_future = Future()
    app.gripper_future.set_result(
        "OK GRASP_FAILED_MIN_LIMIT grip done exit_code=2 hand=left"
    )
    app.gripper_future_command = "grip"
    app.gripper_future_hand = "left"
    app.auto_put_after_confirmed_grasp = lambda: None
    app.start_grip_failure_recovery = lambda *args, **kwargs: pytest.fail(
        "ribbon must not enter failure recovery"
    )

    app._collect_gripper_result()

    assert app.state.grasp_confirmed
    assert app.state.grasp_confirmed_hand == "left"
    assert app.state.grasp_confirmed_label == "ribbon_1"
