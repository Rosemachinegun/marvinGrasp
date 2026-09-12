#!/usr/bin/env python3
"""Microphone speech-to-text module for MarvinGrasp.

Responsibilities:
    microphone -> audio waveform -> speech recognition -> text

This module deliberately knows nothing about:
    - SAM3
    - FlowPose
    - IK
    - robot motion
    - gripper control

Typical usage:
    recognizer = VoiceToText()
    text = recognizer.listen_once()
    print(text)
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    """Configuration for microphone recording and Whisper inference."""

    # Whisper
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "zh"

    # Microphone
    sample_rate: int = 16000
    channels: int = 1
    record_seconds: float = 4.0
    microphone_device: int | str | None = None

    # Recognition
    beam_size: int = 3
    use_vad: bool = True

    # Reject recordings that are basically silent.
    min_rms: float = 0.003

    # Domain words can improve recognition of short robot commands.
    hotwords: str = (
        "Marvin "
        "机械臂 "
        "抓取 "
        "夹取 "
        "笔 "
        "钢笔 "
        "螺丝刀 "
        "把手 "
        "黄色方块 "
        "方块 "
        "U盘 "
        "硬盘"
    )


@dataclass(frozen=True, slots=True)
class VoiceResult:
    """Speech recognition result."""

    text: str
    language: str | None
    language_probability: float | None
    duration_seconds: float
    rms: float

    @property
    def empty(self) -> bool:
        return not bool(self.text.strip())


class VoiceToText:
    """Simple microphone -> text interface."""

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self.config = config or VoiceConfig()

        LOGGER.info(
            "Loading Whisper model: model=%s device=%s compute=%s",
            self.config.model_size,
            self.config.device,
            self.config.compute_type,
        )

        self._model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

        LOGGER.info("Whisper model loaded.")

    # ------------------------------------------------------------------
    # Microphone
    # ------------------------------------------------------------------

    @staticmethod
    def list_microphones() -> None:
        """Print all available audio devices."""

        devices = sd.query_devices()

        print("\nAvailable audio devices:")
        print("-" * 80)

        for index, device in enumerate(devices):
            if device["max_input_channels"] <= 0:
                continue

            print(
                f"[{index:2d}] "
                f"{device['name']} | "
                f"inputs={device['max_input_channels']} | "
                f"default_sr={device['default_samplerate']:.0f}"
            )

        print("-" * 80)

    def check_microphone(self) -> None:
        """Validate microphone settings before recording."""

        sd.check_input_settings(
            device=self.config.microphone_device,
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            dtype="float32",
        )

    def record(
        self,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """Record one audio command from the configured microphone."""

        seconds = (
            self.config.record_seconds
            if duration is None
            else float(duration)
        )

        if seconds <= 0:
            raise ValueError("Recording duration must be greater than zero.")

        self.check_microphone()

        frame_count = int(
            round(seconds * self.config.sample_rate)
        )

        LOGGER.info(
            "Recording %.2f seconds from microphone %r",
            seconds,
            self.config.microphone_device,
        )

        audio = sd.rec(
            frames=frame_count,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="float32",
            device=self.config.microphone_device,
            blocking=True,
        )

        # Whisper expects mono waveform.
        if audio.ndim == 2:
            audio = audio[:, 0]

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        return np.ascontiguousarray(audio)

    # ------------------------------------------------------------------
    # Audio validation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_rms(audio: np.ndarray) -> float:
        """Calculate root-mean-square volume."""

        if audio.size == 0:
            return 0.0

        audio64 = audio.astype(
            np.float64,
            copy=False,
        )

        return float(
            np.sqrt(
                np.mean(audio64 * audio64)
            )
        )

    # ------------------------------------------------------------------
    # Whisper
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
    ) -> VoiceResult:
        """Convert a mono 16 kHz audio waveform into text."""

        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).reshape(-1)

        if audio.size == 0:
            return VoiceResult(
                text="",
                language=None,
                language_probability=None,
                duration_seconds=0.0,
                rms=0.0,
            )

        duration_seconds = (
            len(audio) / self.config.sample_rate
        )

        rms = self.calculate_rms(audio)

        LOGGER.debug(
            "Audio duration=%.2fs RMS=%.6f",
            duration_seconds,
            rms,
        )

        # Avoid Whisper hallucinating words from near-silent recordings.
        if rms < self.config.min_rms:
            LOGGER.info(
                "Recording rejected as silence: RMS %.6f < %.6f",
                rms,
                self.config.min_rms,
            )

            return VoiceResult(
                text="",
                language=None,
                language_probability=None,
                duration_seconds=duration_seconds,
                rms=rms,
            )

        segments, info = self._model.transcribe(
            audio,
            language=self.config.language,
            task="transcribe",
            beam_size=self.config.beam_size,

            # Each robot command is independent.
            condition_on_previous_text=False,

            # Useful for removing silence inside the recording.
            vad_filter=self.config.use_vad,

            # Vocabulary hints for MarvinGrasp.
            hotwords=self.config.hotwords,
        )

        # segments is lazy/generator-like, so iterate over it here.
        text_parts: list[str] = []

        for segment in segments:
            segment_text = segment.text.strip()

            if segment_text:
                text_parts.append(segment_text)

        text = "".join(text_parts).strip()

        return VoiceResult(
            text=text,
            language=getattr(
                info,
                "language",
                None,
            ),
            language_probability=getattr(
                info,
                "language_probability",
                None,
            ),
            duration_seconds=duration_seconds,
            rms=rms,
        )

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def listen_once(
        self,
        duration: Optional[float] = None,
    ) -> VoiceResult:
        """Record one command and immediately transcribe it."""

        audio = self.record(
            duration=duration,
        )

        return self.transcribe(audio)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(name)s: "
            "%(message)s"
        ),
    )

    config = VoiceConfig(
        model_size="small",
        device="cpu",
        compute_type="int8",
        language="zh",
        record_seconds=4.0,

        # None = system default input device.
        microphone_device=None,
    )

    recognizer = VoiceToText(config)

    print()
    print("=" * 60)
    print("MarvinGrasp Voice Recognition Test")
    print("=" * 60)

    try:
        while True:
            command = input(
                "\nPress ENTER to speak, "
                "'d' to list microphones, "
                "'q' to quit: "
            ).strip().lower()

            if command == "q":
                break

            if command == "d":
                recognizer.list_microphones()
                continue

            print(
                f"\nListening for "
                f"{config.record_seconds:.1f} seconds..."
            )

            try:
                result = recognizer.listen_once()

            except Exception as exc:
                LOGGER.exception(
                    "Microphone/ASR failure: %s",
                    exc,
                )
                continue

            print()

            if result.empty:
                print("[VOICE] No speech detected.")
                print(
                    f"[VOICE] RMS: "
                    f"{result.rms:.6f}"
                )
                continue

            print(
                f"[VOICE] Text: "
                f"{result.text}"
            )

            print(
                f"[VOICE] Language: "
                f"{result.language}"
            )

            if result.language_probability is not None:
                print(
                    "[VOICE] Language confidence: "
                    f"{result.language_probability:.3f}"
                )

            print(
                f"[VOICE] Audio: "
                f"{result.duration_seconds:.2f}s"
            )

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()