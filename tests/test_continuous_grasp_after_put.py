from types import SimpleNamespace
from unittest.mock import Mock

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
    app.start_pipeline = Mock()

    app.publish_put()

    assert app.state.grasp_confirmed is False
    assert app.state.grasp_confirmed_hand is None
    assert app.state.grasp_confirmed_label is None
    if continuous:
        app.start_pipeline.assert_called_once_with(grasp_on_done=True)
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
    app.start_pipeline = Mock()

    app.publish_put()

    app.start_pipeline.assert_called_once_with(grasp_on_done=True)
