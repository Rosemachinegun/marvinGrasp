from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from grasp_core.apps.voice_grasp import VoiceGraspInput, parse_voice_target
from grasp_core.ui.tablet_ui import (
    TabletCommand,
    TabletTaskLoopBridge,
    format_base_target_text,
)
from grasp_core.tools.flowpose_request_ik_app import GraspDemoApp, RuntimeState


@dataclass
class _SpeechResult:
    text: str


class _Recognizer:
    def listen_once(self, duration: float | None = None) -> _SpeechResult:
        assert duration == 0.01
        return _SpeechResult("请抓取黄色方块")


def test_voice_parser_rejects_negation_and_recognizes_targets() -> None:
    assert parse_voice_target("请抓取黄色方块") == "yellow_cube"
    assert parse_voice_target("拿螺丝刀") == "screwdriver_handle"
    assert parse_voice_target("不要抓笔") is None


def test_voice_worker_only_queues_parsed_command() -> None:
    voice = VoiceGraspInput(
        enabled=True,
        record_seconds=0.01,
        recognizer_factory=_Recognizer,
    )
    try:
        assert voice.request(robot_busy=False) is not None
        assert voice._thread is not None
        voice._thread.join(timeout=1)
        command = voice.poll()
        assert command is not None
        assert command.target == "yellow_cube"
    finally:
        voice.close()


def test_tablet_bridge_converts_frames_and_drains_commands() -> None:
    bridge = TabletTaskLoopBridge()
    bridge.request(TabletCommand.GRASP)
    bridge.request(TabletCommand.HOME_LEFT)
    assert bridge.drain_commands() == [
        TabletCommand.GRASP,
        TabletCommand.HOME_LEFT,
    ]

    bgr = np.array([[[1, 2, 3]]], dtype=np.uint8)
    bridge.publish(bgr, "ready", sam_bgr=bgr)
    snapshot = bridge.snapshot()
    assert snapshot.image_rgb.tolist() == [[[3, 2, 1]]]
    assert snapshot.sam_rgb.tolist() == [[[3, 2, 1]]]


def test_tablet_target_formatting_is_type_agnostic() -> None:
    target = type(
        "Target",
        (),
        {"frame_id": "cube", "base_xyz": np.array([1.0, 2.0, 3.0])},
    )()
    assert format_base_target_text([target]) == (
        "cube: x=1.000 y=2.000 z=3.000 m"
    )


def test_tablet_commands_are_dispatched_by_the_application_thread() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.tablet_bridge = TabletTaskLoopBridge()
    app.state = RuntimeState()
    app.manual_home_future = None
    app.start_pipeline = Mock(return_value=True)
    app.publish_manual_home = Mock()

    app.tablet_bridge.request(TabletCommand.GRASP)
    app.tablet_bridge.request(TabletCommand.HOME_LEFT)
    app.handle_tablet_commands(bundle=object())

    app.start_pipeline.assert_called_once_with(grasp_on_done=True)
    app.publish_manual_home.assert_called_once_with(("left",))


def test_voice_grasp_is_one_shot_after_successful_place() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = SimpleNamespace(continuous_grasp_after_place=True)
    app.state = RuntimeState()
    app.voice = Mock(task_active=True, active_target="yellow_cube")
    app.start_pipeline = Mock(return_value=True)

    app.restart_grasp_pipeline_after_place()

    app.voice.finish_task.assert_called_once_with()
    app.start_pipeline.assert_not_called()
