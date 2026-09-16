#!/usr/bin/env python3
"""Voice-command boundary for the grasp demo.

This module owns the V-key recording lifecycle, STT worker thread, thread-safe
queue, Chinese target parsing, active one-shot target, and dynamic SAM3 prompt
selection. It never calls perception, IK, the robot, or the gripper.

The optional microphone/STT dependencies are imported lazily so the rest of
the application can still start with voice disabled or unconfigured.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Callable, Protocol

from add.settings import VOICE_ENABLED


VOICE_RECORD_SECONDS = 2.0
VOICE_SAMPLE_RATE = 16_000
VOICE_MIN_RMS = float(os.getenv("VOICE_MIN_RMS", "0.001"))
VOICE_GAIN = float(os.getenv("VOICE_GAIN", "1.0"))
VOICE_AUTO_GAIN = float(os.getenv("VOICE_AUTO_GAIN", "8.0"))
VOICE_HOTWORDS = (
    "Marvin 机械臂 抓取 夹取 笔 钢笔 螺丝刀 把手 黄色方块 方块 U盘 硬盘"
)


class SpeechResult(Protocol):
    text: str


class SpeechRecognizer(Protocol):
    def listen_once(self, duration: float | None = None) -> SpeechResult: ...


@dataclass(frozen=True, slots=True)
class _TextResult:
    text: str


class VoiceToText:
    """Record from the default microphone and transcribe with Faster-Whisper.

    All optional imports are delayed until the user presses ``V``.  The model
    can be a Hugging Face model name or a local CTranslate2 model directory.
    Environment variables are intentionally simple so this module also works
    when launched outside the full application:

    ``FASTER_WHISPER_MODEL`` (default: ``small``)
    ``FASTER_WHISPER_DEVICE`` (``cuda`` or ``cpu``; default: auto)
    ``FASTER_WHISPER_COMPUTE_TYPE`` (default: auto)
    ``VOICE_INPUT_DEVICE`` (optional sounddevice input-device index/name)
    """

    def __init__(self) -> None:
        self.model_name = os.getenv("FASTER_WHISPER_MODEL", "small")
        self.device = os.getenv("FASTER_WHISPER_DEVICE", "auto")
        self.compute_type = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "auto")
        self.input_device = os.getenv("VOICE_INPUT_DEVICE")
        self._model = None

    def _load_model(self):
        from faster_whisper import WhisperModel

        device = self.device
        compute_type = self.compute_type
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        print(
            f"[voice] loading Faster-Whisper model={self.model_name} "
            f"device={device} compute_type={compute_type}",
            flush=True,
        )
        self._model = WhisperModel(
            self.model_name,
            device=device,
            compute_type=compute_type,
        )
        return self._model

    def listen_once(self, duration: float | None = None) -> SpeechResult:
        import sounddevice as sd
        import numpy as np

        seconds = float(duration or VOICE_RECORD_SECONDS)
        device = self.input_device
        if device is not None and device.isdigit():
            device = int(device)
        if device is None:
            device = sd.default.device[0]
        device_info = sd.query_devices(device, "input")
        native_rate = int(round(float(device_info["default_samplerate"])))
        max_channels = int(device_info["max_input_channels"])
        if max_channels < 1:
            raise RuntimeError(f"Audio device has no input channels: {device_info['name']}")
        print(f"[voice] recording for {seconds:g} seconds...", flush=True)
        audio = sd.rec(
            int(seconds * native_rate),
            samplerate=native_rate,
            channels=1,
            dtype="float32",
            device=device,
            blocking=True,
        )
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        # Most USB/ALSA microphones expose 44.1/48 kHz, while Whisper expects
        # a 16 kHz waveform.  Recording at the hardware rate avoids
        # paInvalidSampleRate, then linear interpolation performs the small
        # downsampling without adding another audio dependency.
        if native_rate != VOICE_SAMPLE_RATE:
            source_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
            target_length = int(round(len(audio) * VOICE_SAMPLE_RATE / native_rate))
            target_x = np.linspace(0.0, 1.0, target_length, endpoint=False)
            audio = np.interp(target_x, source_x, audio).astype(np.float32)
        if VOICE_GAIN != 1.0:
            audio = np.clip(audio * VOICE_GAIN, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
        if rms < VOICE_MIN_RMS:
            print(
                f"[voice] no speech detected (RMS={rms:.6f}, "
                f"peak={float(np.max(np.abs(audio))) if audio.size else 0.0:.6f})",
                flush=True,
            )
            return _TextResult("")
        # USB microphones can expose a very small float amplitude.  Boost
        # quiet recordings before Whisper while keeping a hard gain limit.
        if 0.0 < VOICE_AUTO_GAIN and rms < 0.02:
            auto_gain = min(VOICE_AUTO_GAIN, 0.02 / rms)
            audio = np.clip(audio * auto_gain, -1.0, 1.0)
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        print(
            f"[voice] audio level RMS={rms:.6f} "
            f"peak={float(np.max(np.abs(audio))):.6f}",
            flush=True,
        )
        model = self._model or self._load_model()
        segments, _info = model.transcribe(
            audio,
            language="zh",
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
            hotwords=VOICE_HOTWORDS,
        )
        text = "".join(segment.text for segment in segments).strip()
        if not text:
            # Some microphones produce speech below the Silero VAD threshold.
            # Retry once without VAD; this is still bounded to one recording.
            segments, _info = model.transcribe(
                audio,
                language="zh",
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=False,
                hotwords=VOICE_HOTWORDS,
            )
            text = "".join(segment.text for segment in segments).strip()
        print(f"[voice] recognized: {text or '<no speech>'}", flush=True)
        return _TextResult(text)


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
        enabled: bool = VOICE_ENABLED,
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
        return VoiceToText()

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
    print("Press ENTER to record a voice grasp command, or Q to quit.")
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
