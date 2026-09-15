from argparse import Namespace
from concurrent.futures import Future

import numpy as np
import pytest

from grasp_core.tools.flowpose_request_ik_app import GraspDemoApp, RetryStage
from grasp_core.execution.skills.grasp import GripFailedMinLimit, execute_grip_at_pose
from grasp_core.core.types.robot_target_pose import TargetObjectPose


def test_grasp_path_artifacts_are_not_saved_before_template_grip() -> None:
    """Diagnostics must stay out of the target-arrival -> close critical path."""
    import inspect
    from grasp_core.execution.skills import grasp as skill_grasp

    source = inspect.getsource(skill_grasp.execute_grasp)
    callback_index = source.index("count += execute_grip_at_pose(")
    final_artifact_index = source.index(
        "grasp_path_artifacts or save_request_ik_grasp_path_artifacts"
    )

    assert "save_request_ik_grasp_path_artifacts" not in source[:callback_index]
    assert callback_index < final_artifact_index


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

    def publish_failure_recovery(self, hand: str, **kwargs) -> str:
        self.calls.append(("recovery", hand))
        return f"{hand} recovery ok"

    def send_gripper(self, command: str, hand: str | None = None) -> str:
        self.calls.append((command, hand))
        return f"{hand} {command} ok"


class FakeIkPublisher:
    def __init__(self) -> None:
        self.generation = 0
        self.stopped = False

    def request_stop(self) -> int:
        self.generation += 1
        self.stopped = True
        return self.generation

    def stop_generation(self) -> int:
        return self.generation


def test_left_grip_failure_returns_home_even_when_retry_loop_disabled() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = Namespace(
        ik_hand="auto",
        grip_retry_loop=False,
        grip_retry_max_attempts=3,
    )
    app.state = app_state()
    app.robot_actions = FakeRobotActions()
    app.ik_publisher = FakeIkPublisher()
    app.action_executor = ImmediateExecutor()

    app.start_grip_failure_recovery(
        "GRIP_FAILED_MIN_LIMIT hand=left: reached min",
        failed_hand="left",
    )

    assert app.state.retry_stage is RetryStage.RECOVERY
    assert app.state.retry_will_regrasp is False
    assert app.robot_actions.calls == [("release", "left"), ("recovery", "left")]
    assert "releasing left" in app.state.status
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
    app.ik_publisher = FakeIkPublisher()
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
    assert app.robot_actions.calls == [("release", "right"), ("recovery", "right")]
    assert "releasing right" in app.state.status


def test_failure_recovery_continues_with_fresh_grasp_pipeline() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = Namespace(
        ik_hand="auto",
        continuous_grasp_after_place=True,
        grip_retry_loop=True,
        grip_retry_max_attempts=3,
    )
    app.state = app_state()
    app.robot_actions = FakeRobotActions()
    app.ik_publisher = FakeIkPublisher()
    app.action_executor = ImmediateExecutor()
    submitted = []
    app.submit_sam3 = lambda bundle, retry=False: submitted.append((bundle, retry))

    app.start_grip_failure_recovery(
        "GRIP_FAILED_MIN_LIMIT hand=right: reached min",
        failed_hand="right",
    )
    app.update_recovery()
    frame = object()
    app.advance_replan_state(frame)

    assert app.state.retry_stage is RetryStage.SAM3
    assert submitted == [(frame, True)]


def test_continuous_place_switch_does_not_limit_failure_regrasp() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = Namespace(
        ik_hand="auto",
        continuous_grasp_after_place=False,
        grip_retry_loop=True,
        grip_retry_max_attempts=3,
    )
    app.state = app_state()
    app.robot_actions = FakeRobotActions()
    app.ik_publisher = FakeIkPublisher()
    app.action_executor = ImmediateExecutor()

    app.start_grip_failure_recovery(
        "GRIP_FAILED_MIN_LIMIT hand=left: reached min",
        failed_hand="left",
    )

    # Failed grasps retry indefinitely; only the explicit retry-loop switch
    # disables the recovery loop.
    assert app.state.retry_will_regrasp is True


def app_state():
    from grasp_core.tools.flowpose_request_ik_app import RuntimeState

    return RuntimeState()


class FakePublisher:
    publish_rate_hz = 100.0

    def hold_target(self, hand, position, orientation, duration_sec):
        return 1


def test_grip_failed_min_limit_alias_raises_without_confirming(monkeypatch) -> None:
    confirmed = []
    args = Namespace(grip_settle_sec=0.0, grip_post_confirm_hold_sec=0.0)
    monkeypatch.setattr(
        "grasp_core.execution.skills.grasp.gripper_receiver_args",
        lambda args, hand: [
            ("left", Namespace(grip_signal_port=55551, gripper_server="mock"))
        ],
    )
    monkeypatch.setattr(
        "grasp_core.execution.skills.grasp.send_gripper_signal",
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
        "grasp_core.execution.skills.grasp.gripper_receiver_args",
        lambda args, hand: [
            ("left", Namespace(grip_signal_port=55551, gripper_server="mock"))
        ],
    )
    monkeypatch.setattr(
        "grasp_core.execution.skills.grasp.send_gripper_signal",
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
    app.auto_place_after_confirmed_grasp = lambda: None
    app.start_grip_failure_recovery = lambda *args, **kwargs: pytest.fail(
        "ribbon must not enter failure recovery"
    )

    app._collect_gripper_result()

    assert app.state.grasp_confirmed
    assert app.state.grasp_confirmed_hand == "left"
    assert app.state.grasp_confirmed_label == "ribbon_1"


def test_manual_gripper_only_result_does_not_trigger_followup() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = app_state()
    app.state.grasp_confirmed = True
    app.state.grasp_confirmed_hand = "right"
    app.gripper_future = Future()
    app.gripper_future.set_result(
        "OK GRASP_FAILED_MIN_LIMIT grip done exit_code=2 hand=right"
    )
    app.gripper_future_command = "grip"
    app.gripper_future_hand = "right"
    app.gripper_future_followup = False
    app.gripper_interrupted = False
    app.auto_place_after_confirmed_grasp = lambda: pytest.fail(
        "manual gripper control must not trigger place"
    )
    app.start_grip_failure_recovery = lambda *args, **kwargs: pytest.fail(
        "manual gripper control must not trigger recovery"
    )

    app._collect_gripper_result()

    assert app.gripper_future is None
    assert app.state.grasp_confirmed
    assert app.state.grasp_confirmed_hand == "right"
