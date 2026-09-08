from types import SimpleNamespace
from unittest.mock import Mock
from concurrent.futures import Future

import numpy as np

import pytest

from grasp_core.apps.flowpose_request_ik_app import GraspDemoApp, RuntimeState


@pytest.mark.parametrize("continuous", [True, False])
def test_successful_put_optionally_restarts_a_key_pipeline(continuous):
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = SimpleNamespace(
        enable_put_after_grasp=True,
        continuous_grasp_after_put=continuous,
    )
    app.state = RuntimeState(
        grasp_confirmed=True,
        grasp_confirmed_hand="right",
        grasp_confirmed_label="cube",
    )
    app.robot_actions = Mock()
    app.robot_actions.publish_put.return_value = SimpleNamespace(
        ok=True,
        status="put complete",
        grasp_hand="right",
    )
    app.start_pipeline = Mock(return_value=True)

    app.publish_put()

    assert app.state.grasp_confirmed is False
    assert app.state.grasp_confirmed_hand is None
    assert app.state.grasp_confirmed_label is None
    assert app.state.parked_after_put_hand == "right"
    if continuous:
        app.start_pipeline.assert_called_once_with(grasp_on_done=True)
        assert app.state.continuous_grasp_pending is False
    else:
        app.start_pipeline.assert_not_called()


def test_successful_put_restarts_pipeline_by_default():
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = SimpleNamespace(enable_put_after_grasp=True)
    app.state = RuntimeState(
        grasp_confirmed=True,
        grasp_confirmed_hand="left",
        grasp_confirmed_label="cube",
    )
    app.robot_actions = Mock()
    app.robot_actions.publish_put.return_value = SimpleNamespace(
        ok=True,
        status="put complete",
        grasp_hand="left",
    )
    app.start_pipeline = Mock(return_value=True)

    app.publish_put()

    app.start_pipeline.assert_called_once_with(grasp_on_done=True)


def test_successful_put_retries_pipeline_when_first_start_is_blocked():
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = SimpleNamespace(enable_put_after_grasp=True)
    app.state = RuntimeState(
        grasp_confirmed=True,
        grasp_confirmed_hand="right",
        grasp_confirmed_label="cube",
    )
    app.robot_actions = Mock()
    app.robot_actions.publish_put.return_value = SimpleNamespace(
        ok=True,
        status="put complete",
        grasp_hand="right",
    )
    app.start_pipeline = Mock(side_effect=[False, True])

    app.publish_put()

    assert app.state.continuous_grasp_pending is True
    app.advance_continuous_grasp()
    assert app.state.continuous_grasp_pending is False
    assert app.start_pipeline.call_count == 2


def test_switching_hands_returns_previous_put_hand_home_in_parallel():
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = SimpleNamespace(ik_target_index=0, ik_hand="auto")
    app.state = RuntimeState(
        base_targets=[SimpleNamespace(base_xyz=np.array([0.4, 0.2, 0.7]))],
        parked_after_put_hand="right",
    )
    app.robot_actions = Mock()
    app.action_executor = Mock()
    home_future = Future()
    grasp_future = Future()
    app.action_executor.submit.side_effect = [home_future, grasp_future]
    app.grasp_future = None
    app.auto_home_future = None
    app.auto_home_hand = None
    app.ik_publisher = None

    app.publish_grasp()

    assert app.action_executor.submit.call_count == 2
    home_call, grasp_call = app.action_executor.submit.call_args_list
    assert home_call.args == (app.robot_actions.publish_home, "right")
    assert home_call.kwargs == {"fresh_measured_start": False}
    assert grasp_call.args == (
        app.robot_actions.publish_grasp,
        app.state.base_targets,
    )
    assert app.auto_home_future is home_future
    assert app.grasp_future is grasp_future
    assert app.state.parked_after_put_hand is None


def test_same_hand_keeps_put_pose_and_does_not_start_home():
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = SimpleNamespace(ik_target_index=0, ik_hand="auto")
    app.state = RuntimeState(
        base_targets=[SimpleNamespace(base_xyz=np.array([0.4, -0.2, 0.7]))],
        parked_after_put_hand="right",
    )
    app.robot_actions = Mock()
    app.action_executor = Mock()
    grasp_future = Future()
    app.action_executor.submit.return_value = grasp_future
    app.grasp_future = None
    app.auto_home_future = None
    app.auto_home_hand = None
    app.ik_publisher = None

    app.publish_grasp()

    app.action_executor.submit.assert_called_once_with(
        app.robot_actions.publish_grasp,
        app.state.base_targets,
    )
    assert app.state.parked_after_put_hand == "right"
    assert app.auto_home_future is None
