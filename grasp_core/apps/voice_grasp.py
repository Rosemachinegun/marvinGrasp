#!/usr/bin/env python3
"""Voice-command boundary for the grasp demo.

This module owns the V-key recording lifecycle, STT worker thread, thread-safe
queue, Chinese target parsing, active one-shot target, and dynamic SAM3 prompt
selection. It never calls perception, IK, the robot, or the gripper. Optional
audio and Whisper dependencies are imported only when recording starts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Callable, Protocol


VOICE_RECORD_SECONDS = 4.0


class SpeechResult(Protocol):
    text: str


class SpeechRecognizer(Protocol):
    def listen_once(self, duration: float | None = None) -> SpeechResult: ...


@dataclass(frozen=True, slots=True)
class WhisperSpeechResult:
    """Minimal result returned by the built-in microphone recognizer."""

    text: str


class WhisperSpeechRecognizer:
    """Lazy microphone + faster-whisper adapter owned by the voice module."""

    def __init__(self) -> None:
        import os

        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Voice input requires sounddevice and faster-whisper"
            ) from exc

        self._sd = sd
        self._sample_rate = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
        self._device = os.getenv("VOICE_MICROPHONE_DEVICE") or None
        self._model = WhisperModel(
            os.getenv("VOICE_WHISPER_MODEL", "small"),
            device=os.getenv("VOICE_WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("VOICE_WHISPER_COMPUTE_TYPE", "int8"),
        )

    def listen_once(self, duration: float | None = None) -> WhisperSpeechResult:
        import numpy as np

        seconds = VOICE_RECORD_SECONDS if duration is None else float(duration)
        if seconds <= 0:
            raise ValueError("Recording duration must be greater than zero")
        frames = int(round(seconds * self._sample_rate))
        audio = self._sd.rec(
            frames=frames,
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            device=self._device,
            blocking=True,
        )
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not waveform.size:
            return WhisperSpeechResult(text="")
        rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
        if rms < 0.003:
            return WhisperSpeechResult(text="")
        segments, _ = self._model.transcribe(
            waveform,
            language="zh",
            task="transcribe",
            beam_size=3,
            condition_on_previous_text=False,
            vad_filter=True,
            hotwords="机械臂 抓取 夹取 笔 螺丝刀 把手 黄色方块 方块 玩具",
        )
        return WhisperSpeechResult(
            text="".join(segment.text.strip() for segment in segments).strip()
        )


@dataclass(frozen=True, slots=True)
class ParsedVoiceCommand:
    """One STT result parsed on the application main thread."""

    text: str
    target: str | None
    status: str


def parse_voice_target(text: str) -> str | None:
    """Map a short Chinese grasp instruction to a canonical object label."""
    command = re.sub(r"[\s，。！？、,!.?]", "", text).lower()
    if (
        not command
        or any(word in command for word in ("不要", "别", "取消", "停止"))
        or not any(verb in command for verb in ("夹", "抓", "拿", "取"))
    ):
        return None

    if "螺丝刀" in command or "改锥" in command:
        return "screwdriver_handle"
    if ("黄色" in command or "黄" in command) and any(
        word in command for word in ("方块", "积木", "立方体")
    ):
        return "yellow_cube"
    if "笔" in command:
        return "pen"
    if "玩具" in command:
        return "toy"
    return None


def sam_prompt_for_target(
    target: str | None,
    default_prompts: str,
    pick_templates: dict[str, object] | None,
) -> str:
    """Return prompts whose resulting labels can reuse existing YAML policies."""
    if target is None:
        return default_prompts
    if target == "screwdriver_handle":
        variants = sorted(
            name for name in (pick_templates or {})
            if name == target or name.endswith("_screwdriver_handle")
        )
        if variants:
            return ",".join(variants)
    return target


class VoiceGraspInput:
    """Manage one-shot speech recognition without controlling the robot."""

    def __init__(
        self,
        *,
        enabled: bool,
        record_seconds: float = VOICE_RECORD_SECONDS,
        recognizer_factory: Callable[[], SpeechRecognizer] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.record_seconds = float(record_seconds)
        self._recognizer_factory = recognizer_factory
        self._recognizer: SpeechRecognizer | None = None
        self._queue: Queue[str] = Queue(maxsize=1)
        self._stop = Event()
        self._cancelled = Event()
        self._thread: Thread | None = None
        self.active_target: str | None = None

    @property
    def recognition_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def task_active(self) -> bool:
        return self.active_target is not None

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def request(self, *, robot_busy: bool) -> str | None:
        """Start one background recording; return the UI status to display."""
        if not self.enabled:
            return None
        if robot_busy or self.task_active:
            return "Voice command ignored: robot busy"
        if self.recognition_running or not self._queue.empty():
            return "Voice recognition already running"

        self._cancelled.clear()
        self._thread = Thread(
            target=self._recognize_once,
            name="voice-stt",
            daemon=True,
        )
        self._thread.start()
        return f"Voice recognition starting ({self.record_seconds:g} s recording)"

    def _build_recognizer(self) -> SpeechRecognizer:
        if self._recognizer_factory is not None:
            return self._recognizer_factory()
        return WhisperSpeechRecognizer()

    def _recognize_once(self) -> None:
        """Run only microphone/STT work and enqueue its text result."""
        try:
            if self._recognizer is None:
                self._recognizer = self._build_recognizer()
        except Exception as exc:  # noqa: BLE001
            print(f"[voice] STT unavailable: {exc}", flush=True)
            return

        if self._stop.is_set() or self._cancelled.is_set():
            return
        try:
            result = self._recognizer.listen_once(duration=self.record_seconds)
        except Exception as exc:  # noqa: BLE001
            print(f"[voice] recognition failed: {exc}", flush=True)
            return
        if self._stop.is_set() or self._cancelled.is_set():
            return
        self.submit_transcript(str(result.text))

    def submit_transcript(self, text: str) -> bool:
        """Queue STT text without exposing queue internals; useful for adapters/tests."""
        try:
            self._queue.put_nowait(str(text))
        except Full:
            print(f"[voice] command discarded (queue full): {text}", flush=True)
            return False
        return True

    def poll(self) -> ParsedVoiceCommand | None:
        """Parse at most one queued STT result on the caller's main thread."""
        try:
            text = self._queue.get_nowait()
        except Empty:
            return None

        text = text.strip()
        if not text:
            return ParsedVoiceCommand(text="", target=None, status="Voice: no speech detected")
        target = parse_voice_target(text)
        if target is None:
            return ParsedVoiceCommand(
                text=text,
                target=None,
                status=f"Voice target not recognized: {text}",
            )
        return ParsedVoiceCommand(
            text=text,
            target=target,
            status=f"Voice target recognized: {target}",
        )

    def activate(self, target: str) -> None:
        self.active_target = str(target)

    def finish_task(self) -> None:
        self.active_target = None

    def sam_prompt(
        self,
        default_prompts: str,
        pick_templates: dict[str, object] | None,
    ) -> str:
        return sam_prompt_for_target(
            self.active_target,
            default_prompts,
            pick_templates,
        )

    def cancel(self) -> None:
        """Discard a recording result and clear the active one-shot target."""
        self._cancelled.set()
        self.finish_task()
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return

    def close(self) -> None:
        self._stop.set()
        self.cancel()
        if self._thread is not None:
            self._thread.join(timeout=self.record_seconds + 1.0)
            self._thread = None


def main() -> int:
    """Small standalone STT/parser smoke test using the same voice flow."""
    voice = VoiceGraspInput(enabled=True)
    print("Press ENTER to record a 4-second grasp command, or Q to quit.")
    try:
        while input("> ").strip().lower() != "q":
            print(voice.request(robot_busy=False))
            if voice._thread is not None:
                voice._thread.join()
            parsed = voice.poll()
            print(parsed.status if parsed is not None else "No voice result")
    finally:
        voice.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
