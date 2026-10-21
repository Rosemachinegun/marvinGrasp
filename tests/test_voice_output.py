from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock

from add.voice_output import (
    PROMPT_ONLY,
    START_RECOGNITION,
    VoiceOutput,
    VoicePlaybackResult,
)
from grasp_core.tools.flowpose_request_ik_app import GraspDemoApp
from grasp_core.tools.flowpose_request_ik_app import PipelineStage, RuntimeState
from add.voice import VoiceGraspInput


class _FirstChoice:
    def choice(self, items):
        return items[0]


def test_welcome_selects_greeting_then_first_question(tmp_path: Path) -> None:
    greeting = tmp_path / "01_greeting"
    question = tmp_path / "02_1_first_gift_question"
    greeting.mkdir()
    question.mkdir()
    (greeting / "01_hello.mp3").touch()
    (question / "01_question.mp3").touch()

    output = VoiceOutput(tmp_path, player="/bin/true", rng=_FirstChoice())
    selected = output._select_files(
        ("01_greeting", "02_1_first_gift_question")
    )

    assert selected == (
        greeting / "01_hello.mp3",
        question / "01_question.mp3",
    )


class _VoiceOutput:
    def __init__(self, result=None) -> None:
        self.result = result
        self.play_calls = 0
        self.accepted_calls = 0
        self.failed_calls = 0

    def play_welcome(self) -> str:
        self.play_calls += 1
        return "prompt playing"

    def play_task_accepted(self) -> str:
        self.accepted_calls += 1
        return "accepted playing"

    def play_task_failed(self) -> str:
        self.failed_calls += 1
        return "failed playing"

    def poll(self):
        result, self.result = self.result, None
        return result


class _VoiceInput:
    enabled = True
    task_active = False
    recognition_running = False

    def __init__(self) -> None:
        self.request_calls = 0

    def request(self, *, robot_busy: bool) -> str:
        assert robot_busy is False
        self.request_calls += 1
        return "recognition started"


def _app(result=None) -> GraspDemoApp:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = SimpleNamespace(status="")
    app.voice_input = _VoiceInput()
    app.voice_output = _VoiceOutput(result)
    app.robot_busy = lambda: False
    return app


def test_v_request_plays_prompts_before_starting_recognition() -> None:
    app = _app()

    app.request_voice()

    assert app.voice_output.play_calls == 1
    assert app.voice_input.request_calls == 0
    assert app.state.status == "prompt playing"


def test_recognition_starts_after_prompt_sequence_completes() -> None:
    result = VoicePlaybackResult(
        files=(Path("greeting.mp3"), Path("question.mp3")),
        purpose=START_RECOGNITION,
    )
    app = _app(result)

    app.collect_voice_output()

    assert app.voice_input.request_calls == 1
    assert app.state.status == "recognition started"


def test_recognition_does_not_start_if_app_is_paused_during_prompt() -> None:
    result = VoicePlaybackResult(
        files=(Path("greeting.mp3"), Path("question.mp3")),
        purpose=START_RECOGNITION,
    )
    app = _app(result)
    app.state.paused = True

    app.collect_voice_output()

    assert app.voice_input.request_calls == 0
    assert app.state.status == "Voice recognition cancelled: application paused"


def test_other_prompt_completion_does_not_restart_recognition() -> None:
    app = _app(VoicePlaybackResult(files=(Path("accepted.mp3"),)))

    app.collect_voice_output()

    assert app.voice_input.request_calls == 0


def test_valid_recognized_gift_plays_task_accepted_prompt() -> None:
    app = _app()
    app.state.paused = False
    app.voice_input = Mock()
    app.voice_input.poll.return_value = SimpleNamespace(
        target="toy", handover=True, status="recognized"
    )
    app.start_pipeline = Mock(return_value=True)

    app.collect_voice_result()

    assert app.voice_output.accepted_calls == 1
    app.voice_input.activate.assert_called_once_with("toy", handover=True)
    app.start_pipeline.assert_called_once_with(grasp_on_done=True)


def test_valid_voice_with_failed_capture_plays_only_failure() -> None:
    app = _app()
    app.state.paused = False
    app.voice_input = Mock()
    app.voice_input.poll.return_value = SimpleNamespace(
        target="toy", handover=True, status="recognized"
    )
    app.start_pipeline = Mock(return_value=False)

    app.collect_voice_result()

    assert app.voice_output.accepted_calls == 0
    assert app.voice_output.failed_calls == 1
    app.voice_input.finish_task.assert_called_once_with()


def test_unrecognized_voice_plays_task_failed_prompt() -> None:
    app = _app()
    app.state.paused = False
    app.voice_input = Mock()
    app.voice_input.poll.return_value = SimpleNamespace(
        target=None, handover=False, status="not recognized"
    )

    app.collect_voice_result()

    assert app.voice_output.failed_calls == 1
    app.voice_input.finish_task.assert_called_once_with()


def test_missing_voice_gift_plays_failure_and_clears_target() -> None:
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.state = RuntimeState(pipeline_stage=PipelineStage.FLOWPOSE)
    app.voice_input = VoiceGraspInput()
    app.voice_input.activate("toy", handover=True)
    app.voice_output = _VoiceOutput()
    app.home_both_when_no_targets = Mock()

    app.advance_pipeline()

    assert app.voice_output.failed_calls == 1
    assert app.voice_input.task_active is False
    app.home_both_when_no_targets.assert_called_once_with()


def test_followup_prompts_are_queued_in_workflow_order(tmp_path: Path) -> None:
    directories = (
        "03_1_task_failed",
        "03_2_task_accepted",
        "05_1_reply_thanks",
        "06_1_goodbye",
    )
    for directory_name in directories:
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "01_prompt.mp3").touch()

    output = VoiceOutput(tmp_path, player="/bin/true", rng=_FirstChoice())
    try:
        output.play_task_failed()
        output.play_task_accepted()
        output.play_reply_thanks()
        output.play_goodbye()
        results = []
        deadline = time.monotonic() + 1.0
        while len(results) < len(directories) and time.monotonic() < deadline:
            result = output.poll()
            if result is None:
                time.sleep(0.01)
            else:
                results.append(result)
    finally:
        output.close()

    assert [result.files[0].parent.name for result in results] == list(directories)
    assert all(result.purpose == PROMPT_ONLY for result in results)


def test_no_handover_prompt_waits_until_playback_finishes(tmp_path: Path) -> None:
    directory = tmp_path / "05_2_no_handover"
    directory.mkdir()
    (directory / "01_no_handover.mp3").touch()
    output = VoiceOutput(tmp_path, player="/bin/true", rng=_FirstChoice())
    try:
        status = output.play_no_handover_and_wait()
        result = output.poll()
    finally:
        output.close()

    assert "05_2_no_handover" in status
    assert result is not None
    assert result.files == (directory / "01_no_handover.mp3",)
