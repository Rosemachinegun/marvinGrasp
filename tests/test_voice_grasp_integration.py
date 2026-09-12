from concurrent.futures import Future
from queue import Queue
from threading import Event
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from grasp_core.apps.flowpose_request_ik_app import (
    GraspDemoApp,
    KEY_PAUSE,
    KEY_VOICE,
    RuntimeState,
)
from grasp_core.apps.voice_target_parser import parse_voice_target
from grasp_core.perception.flowpose_pipeline import (
    best_prompt_prediction,
    make_labels,
    split_prompts,
)
from grasp_core.planning.tool_pick_templates import (
    load_tool_pick_templates,
    pick_template_for_target,
)
from grasp_core.tasks.screwdriver_handle_grasp_policy import (
    long_object_grasp_policy_for_object,
)


@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("夹笔", "pen"),
        ("帮我夹黄色方块。", "yellow_cube"),
        ("夹螺丝刀", "screwdriver_handle"),
        ("抓取玩具", "toy"),
        ("抓一下黄色的积木", "yellow_cube"),
        ("夹一个未知物体", None),
        ("黄色方块", None),
        ("不要夹笔", None),
    ],
)
def test_voice_target_parser(command, target):
    assert parse_voice_target(command) == target


def make_app():
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.args = SimpleNamespace(
        prompts="toy,pen",
        enable_put_after_grasp=True,
        continuous_grasp_after_put=True,
        voice_input=False,
    )
    app.state = RuntimeState()
    app.voice_queue = Queue(maxsize=1)
    app.voice_stop = Event()
    app.voice_cancelled = Event()
    app.voice_thread = None
    app.voice_recognizer = None
    app.grasp_future = None
    app.gripper_future = None
    app.manual_home_future = None
    app.auto_home_future = None
    app.ik_publisher = None
    app.gripper_interrupted = False
    return app


def test_v_is_inert_when_voice_input_is_disabled():
    app = make_app()

    with patch("grasp_core.apps.flowpose_request_ik_app.Thread") as thread:
        assert app.handle_key(KEY_VOICE, None)
    thread.assert_not_called()
    assert app.state.voice_target is None
    assert app.state.status == "Ready"


def test_v_starts_one_recording_only_when_enabled_and_idle():
    app = make_app()
    app.args.voice_input = True
    fake_thread = Mock()
    fake_thread.is_alive.return_value = True

    with patch("grasp_core.apps.flowpose_request_ik_app.Thread", return_value=fake_thread) as thread:
        app.handle_key(KEY_VOICE, None)
        app.handle_key(KEY_VOICE, None)

    thread.assert_called_once()
    fake_thread.start.assert_called_once()
    assert app.state.status == "Voice recognition already running"


def test_voice_worker_records_exactly_four_seconds_once():
    app = make_app()
    app.voice_recognizer = Mock()
    app.voice_recognizer.listen_once.return_value = SimpleNamespace(
        empty=False, text="夹笔",
    )

    with patch.dict("sys.modules", {"voice_input": SimpleNamespace(VoiceToText=Mock())}):
        app._voice_worker()

    app.voice_recognizer.listen_once.assert_called_once_with(duration=4.0)
    assert app.voice_queue.get_nowait() == "夹笔"


def test_pause_cancels_pending_voice_recording():
    app = make_app()
    app.args.voice_input = True
    app.voice_thread = Mock()
    app.voice_thread.is_alive.return_value = True

    app.handle_key(KEY_PAUSE, None)

    assert app.voice_cancelled.is_set()
    assert app.state.paused is True


def test_silence_does_not_start_grasp():
    app = make_app()
    app.start_pipeline = Mock()
    app.voice_queue.put_nowait("")

    app.handle_voice_commands()

    assert app.state.status == "Voice: no speech detected"
    app.start_pipeline.assert_not_called()


@pytest.mark.parametrize(
    ("command", "target"),
    [("夹笔", "pen"), ("抓取玩具", "toy")],
)
def test_voice_queue_starts_only_on_main_loop_and_discards_busy_command(command, target):
    app = make_app()
    app.start_pipeline = Mock(return_value=True)
    app.voice_queue.put_nowait(command)

    assert app.state.voice_target is None
    app.handle_voice_commands()
    assert app.state.voice_target == target
    app.start_pipeline.assert_called_once_with(grasp_on_done=True)

    app.voice_queue.put_nowait("夹黄色方块")
    app.handle_voice_commands()
    assert app.state.voice_target == target
    app.start_pipeline.assert_called_once()


def test_voice_command_is_discarded_during_existing_robot_action():
    app = make_app()
    app.grasp_future = Future()
    app.start_pipeline = Mock(return_value=True)
    app.voice_queue.put_nowait("夹笔")

    app.handle_voice_commands()

    assert app.state.voice_target is None
    app.start_pipeline.assert_not_called()


def test_sam3_receives_voice_prompt_and_retry_reuses_it():
    app = make_app()
    app.state.voice_target = "yellow_cube"
    app.capture_dir = "unused"
    app.inference_executor = Mock()
    app.sam_cache = {"runner": None}
    app.sam_kwargs = {}
    bundle = SimpleNamespace(frame_id=7)
    app.inference_executor.submit.return_value = Future()

    with patch(
        "grasp_core.apps.flowpose_request_ik_app.save_capture",
        return_value=("capture.json", {}),
    ), patch(
        "grasp_core.apps.flowpose_request_ik_app.freeze_bundle",
        return_value=bundle,
    ):
        app.submit_sam3(bundle, retry=True)

    assert app.inference_executor.submit.call_args.args[4] == "yellow_cube"
    assert app.args.prompts == "toy,pen"


def test_sam3_uses_fixed_prompts_without_voice_target():
    app = make_app()
    app.capture_dir = "unused"
    app.inference_executor = Mock()
    app.sam_cache = {"runner": None}
    app.sam_kwargs = {}
    bundle = SimpleNamespace(frame_id=8)
    app.inference_executor.submit.return_value = Future()

    with patch(
        "grasp_core.apps.flowpose_request_ik_app.save_capture",
        return_value=("capture.json", {}),
    ), patch(
        "grasp_core.apps.flowpose_request_ik_app.freeze_bundle",
        return_value=bundle,
    ):
        app.submit_sam3(bundle)

    assert app.inference_executor.submit.call_args.args[4] == "toy,pen"


def test_voice_prompts_resolve_to_existing_tool_yaml_pick_templates():
    app = make_app()
    app.pick_templates = load_tool_pick_templates(SimpleNamespace(
        use_tool_pick_template=True,
        tool_template_path=Path(__file__).resolve().parents[1] / "config" / "tool.yaml",
    ))

    for target in ("pen", "yellow_cube", "screwdriver_handle", "toy"):
        app.state.voice_target = target
        prompts = split_prompts(app.sam_prompt_for_current_task())
        assert prompts
        if target == "screwdriver_handle":
            assert len(prompts) > 1
        for prompt in prompts:
            label = make_labels(prompt, 1)[0]
            for hand in ("left", "right"):
                assert pick_template_for_target(
                    SimpleNamespace(label=label), hand, app.pick_templates,
                ) is not None, (target, prompt, hand)


def test_generic_screwdriver_keeps_selected_yaml_label_through_policy_lookup():
    app = make_app()
    app.pick_templates = load_tool_pick_templates(SimpleNamespace(
        use_tool_pick_template=True,
        tool_template_path=Path(__file__).resolve().parents[1] / "config" / "tool.yaml",
    ))
    app.state.voice_target = "screwdriver_handle"

    runner = Mock()
    runner.infer.side_effect = lambda _image, prompt: {
        "prompt": prompt,
        "scores": np.array([0.9 if prompt == "yellow_screwdriver_handle" else 0.1]),
    }
    prediction = best_prompt_prediction(
        runner, np.zeros((1, 1, 3), dtype=np.uint8),
        app.sam_prompt_for_current_task(),
    )
    flowpose_label = make_labels(prediction["prompt"], 1)[0]

    assert flowpose_label == "yellow_screwdriver_handle_1"
    assert pick_template_for_target(
        SimpleNamespace(label=flowpose_label), "right", app.pick_templates,
    ) is not None
    assert long_object_grasp_policy_for_object(flowpose_label) is not None


def test_successful_voice_put_is_one_shot_even_with_continuous_enabled():
    app = make_app()
    app.state.voice_target = "pen"
    app.state.grasp_confirmed = True
    app.state.grasp_confirmed_hand = "right"
    app.state.grasp_confirmed_label = "pen_1"
    app.robot_actions = Mock()
    app.robot_actions.publish_put.return_value = SimpleNamespace(
        ok=True, status="put complete", grasp_hand="right",
    )
    app.start_pipeline = Mock(return_value=True)

    app.publish_put()

    assert app.state.voice_target is None
    assert app.state.continuous_grasp_pending is False
    app.start_pipeline.assert_not_called()


def test_voice_retry_does_not_depend_on_continuous_grasp_switch():
    app = make_app()
    app.args.continuous_grasp_after_put = False
    app.args.grip_retry_loop = True
    app.args.grip_retry_max_attempts = 1
    app.args.ik_hand = "right"
    app.state.voice_target = "pen"
    app.robot_actions = Mock()
    app.action_executor = Mock()
    app.action_executor.submit.side_effect = [Future(), Future()]

    app.start_grip_failure_recovery("grip failed", failed_hand="right")

    assert app.state.retry_will_regrasp is True
    assert app.state.voice_target == "pen"
