from queue import SimpleQueue
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import add.web as web_module
from add.settings import VOICE_ENABLED, WEB_ENABLED
from add.voice import VoiceGraspInput
from add.web import (
    TabletCommand,
    TabletTaskLoopBridge,
    TabletWebService,
    format_base_link_targets,
)
from grasp_core.tools.flowpose_request_ik_app import GraspDemoApp, RuntimeState


class _VoiceInput:
    def __init__(self) -> None:
        self.busy_values: list[bool] = []

    def request(self, *, robot_busy: bool) -> str:
        self.busy_values.append(robot_busy)
        return "voice requested"


class _TabletBridge:
    def __init__(self, *commands: TabletCommand) -> None:
        self.commands: SimpleQueue[TabletCommand] = SimpleQueue()
        for command in commands:
            self.commands.put(command)

    def drain_commands(self) -> list[TabletCommand]:
        commands = []
        while not self.commands.empty():
            commands.append(self.commands.get_nowait())
        return commands


def _app() -> GraspDemoApp:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = SimpleNamespace(paused=False, status="", base_targets=[])
    app.voice_input = _VoiceInput()
    app.robot_busy = lambda: False
    app.manual_home_future = None
    return app


def test_v_key_requests_voice() -> None:
    app = _app()

    assert app.handle_key(ord("v"), None)

    assert app.voice_input.busy_values == [False]
    assert app.state.status == "voice requested"


def test_tablet_voice_button_uses_same_voice_request() -> None:
    app = _app()
    app.tablet_bridge = _TabletBridge(TabletCommand.VOICE)

    app.handle_tablet_commands(None)

    assert app.voice_input.busy_values == [False]
    assert app.state.status == "voice requested"


def test_web_and_voice_are_enabled_by_default() -> None:
    assert WEB_ENABLED is True
    assert VOICE_ENABLED is True
    assert VoiceGraspInput().enabled is True


def test_disabled_web_service_does_not_build_interface(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("disabled web service must not build Gradio UI")

    monkeypatch.setattr(web_module, "create_tablet_interface", fail_if_called)
    service = TabletWebService(
        TabletTaskLoopBridge(),
        host="127.0.0.1",
        port=7860,
        enabled=False,
    )

    service.start()

    assert service.demo is None


def test_base_link_targets_use_three_decimal_places() -> None:
    targets = [
        SimpleNamespace(
            label="toy_0",
            base_xyz=[0.573452943007408, -0.11985761159706018, 0.66],
        )
    ]

    assert format_base_link_targets(targets) == (
        "0: toy_0 xyz=[0.573, -0.120, 0.660]"
    )


@pytest.mark.parametrize("voice_task", [True, False])
@pytest.mark.parametrize("target_count", [1, 3])
def test_voice_place_returns_home_without_starting_another_grasp(
    voice_task, target_count,
) -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState(
        base_targets=[SimpleNamespace(label="pen")] * target_count,
        grasp_confirmed=True,
        grasp_confirmed_hand="right",
        grasp_confirmed_label="pen",
        continuous_grasp_pending=True,
    )
    app.voice_input = VoiceGraspInput()
    if voice_task:
        app.voice_input.activate("pen")
    app.place_future = None
    app.robot_actions = Mock()
    app.action_executor = Mock()
    future = Future()
    app.action_executor.submit.return_value = future
    app.restart_grasp_pipeline_after_place = Mock()

    app.start_place()

    # Place uses count <= 1 to release then HOME, and > 1 to park.
    app.action_executor.submit.assert_called_once_with(
        app.robot_actions.publish_place,
        grasp_confirmed=True,
        hand="right",
        object_label="pen",
        target_count=1 if voice_task else target_count,
    )
    app.collect_place_result()
    assert app.voice_input.task_active is voice_task
    app.restart_grasp_pipeline_after_place.assert_not_called()

    future.set_result(SimpleNamespace(
        ok=True, status="place and HOME complete", grasp_hand="right",
    ))
    app.collect_place_result()

    assert app.voice_input.task_active is False
    assert app.state.grasp_confirmed is False
    assert app.place_future is None
    if voice_task:
        app.restart_grasp_pipeline_after_place.assert_not_called()
        assert app.state.continuous_grasp_pending is False
        assert app.state.parked_after_place_hand is None
    else:
        app.restart_grasp_pipeline_after_place.assert_called_once_with()
        assert app.state.parked_after_place_hand == (
            "right" if target_count > 1 else None
        )
