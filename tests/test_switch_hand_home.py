from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from grasp_core.tools.flowpose_request_ik_app import GraspDemoApp, RuntimeState


def make_app(parked_hand: str, target_y: float):
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = SimpleNamespace(ik_target_index=0, ik_hand="auto")
    app.state = RuntimeState(
        base_targets=[SimpleNamespace(base_xyz=np.array([0.4, target_y, 0.7]))],
        parked_after_place_hand=parked_hand,
    )
    app.robot_actions = Mock()
    app.action_executor = Mock()
    app.action_executor.submit.side_effect = [Future(), Future()]
    app.grasp_future = None
    app.place_future = None
    app.ik_publisher = None
    app.auto_home_future = None
    app.auto_home_hand = None
    return app


def test_switching_hands_homes_parked_hand_before_next_grasp():
    app = make_app("right", target_y=0.2)  # next target is assigned to left

    app.publish_grasp()

    home_call, grasp_call = app.action_executor.submit.call_args_list
    assert home_call.args == (app.robot_actions.publish_home, "right")
    assert home_call.kwargs == {"fresh_measured_start": False}
    assert grasp_call.args == (app.robot_actions.publish_grasp, app.state.base_targets)
    assert app.auto_home_hand == "right"
    assert app.state.status == "Grasp action running; right HOME queued before grasp"
    assert app.state.parked_after_place_hand is None


def test_same_hand_does_not_home_parked_hand():
    app = make_app("right", target_y=-0.2)  # next target stays on right

    app.action_executor.submit.side_effect = [Future()]
    app.publish_grasp()

    app.action_executor.submit.assert_called_once_with(
        app.robot_actions.publish_grasp,
        app.state.base_targets,
    )
    assert app.auto_home_future is None
