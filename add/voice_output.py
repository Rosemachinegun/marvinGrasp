"""Asynchronous voice prompt playback for the grasping demo.

The player deliberately runs outside the camera/UI loop.  Call ``poll`` from
that loop to learn when the complete prompt sequence has finished; microphone
recording can then start without capturing Marvin's own voice.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Sequence


DEFAULT_VOICE_DIR = Path("/home/kewei/marvin_voice")
WELCOME_SEQUENCE = ("01_greeting", "02_1_first_gift_question")
TASK_FAILED_SEQUENCE = ("03_1_task_failed",)
TASK_ACCEPTED_SEQUENCE = ("03_2_task_accepted",)
REPLY_THANKS_SEQUENCE = ("05_1_reply_thanks",)
NO_HANDOVER_SEQUENCE = ("05_2_no_handover",)
GOODBYE_SEQUENCE = ("06_1_goodbye",)
START_RECOGNITION = "start_recognition"
PROMPT_ONLY = "prompt_only"


@dataclass(frozen=True)
class _PlaybackRequest:
    files: tuple[Path, ...]
    purpose: str
    generation: int
    done: Event = field(default_factory=Event, compare=False)


@dataclass(frozen=True)
class VoicePlaybackResult:
    """Result produced after every file in a prompt sequence is played."""

    files: tuple[Path, ...]
    purpose: str = PROMPT_ONLY
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class VoiceOutput:
    """Select and play one MP3 from each requested prompt directory."""

    def __init__(
        self,
        voice_dir: str | Path = DEFAULT_VOICE_DIR,
        *,
        player: str | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.voice_dir = Path(voice_dir).expanduser()
        self.player = player or self._find_player()
        self._rng = rng or random.Random()
        self._requests: Queue[_PlaybackRequest | None] = Queue()
        self._results: Queue[VoicePlaybackResult] = Queue()
        self._active = Event()
        self._closed = Event()
        self._generation = 0
        self._process: subprocess.Popen[bytes] | None = None
        self._state_lock = Lock()
        self._thread = Thread(
            target=self._playback_worker,
            name="voice-output",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _find_player() -> str:
        for name in ("ffplay", "mpg123", "cvlc"):
            executable = shutil.which(name)
            if executable:
                return executable
        raise RuntimeError("No MP3 player found (tried ffplay, mpg123, and cvlc)")

    @property
    def running(self) -> bool:
        return self._active.is_set() or not self._requests.empty()

    def _select_files(self, directories: Sequence[str]) -> tuple[Path, ...]:
        selected: list[Path] = []
        for directory_name in directories:
            directory = self.voice_dir / directory_name
            candidates = sorted(directory.glob("*.mp3"))
            if not candidates:
                raise FileNotFoundError(f"No MP3 files found in {directory}")
            selected.append(self._rng.choice(candidates))
        return tuple(selected)

    def play_welcome(self) -> str:
        """Play greeting -> first gift question without blocking the UI."""
        return self.play_sequence(WELCOME_SEQUENCE, purpose=START_RECOGNITION)

    def play_task_accepted(self) -> str:
        """Acknowledge a valid recognized gift request."""
        return self.play_sequence(TASK_ACCEPTED_SEQUENCE)

    def play_task_failed(self) -> str:
        """Tell the user that speech or the requested gift was not found."""
        return self.play_sequence(TASK_FAILED_SEQUENCE)

    def play_reply_thanks(self) -> str:
        """Reply to the recipient while the confirmed handover completes."""
        return self.play_sequence(REPLY_THANKS_SEQUENCE)

    def play_goodbye(self) -> str:
        """Queue a goodbye after handover HOME has completed."""
        return self.play_sequence(GOODBYE_SEQUENCE)

    def play_no_handover_and_wait(self) -> str:
        """Play the no-handover response fully before fallback Place starts."""
        status, request = self._queue_sequence(NO_HANDOVER_SEQUENCE, PROMPT_ONLY)
        if request is not None:
            request.done.wait()
        return status

    def play_sequence(
        self,
        directories: Sequence[str],
        *,
        purpose: str = PROMPT_ONLY,
    ) -> str:
        """Queue one prompt sequence and return a user-facing status."""
        status, _request = self._queue_sequence(directories, purpose)
        return status

    def _queue_sequence(
        self,
        directories: Sequence[str],
        purpose: str,
    ) -> tuple[str, _PlaybackRequest | None]:
        if self._closed.is_set():
            return "Voice prompt unavailable: player is closed", None
        try:
            files = self._select_files(directories)
        except (OSError, RuntimeError) as exc:
            return f"Voice prompt unavailable: {exc}", None

        was_running = self.running
        with self._state_lock:
            generation = self._generation
        request = _PlaybackRequest(
            files=files,
            purpose=purpose,
            generation=generation,
        )
        self._requests.put(request)
        names = " -> ".join(path.parent.name for path in files)
        action = "queued" if was_running else "playing"
        return f"Voice prompt {action}: {names}", request

    def _command(self, path: Path) -> list[str]:
        name = Path(self.player).name
        if name == "ffplay":
            return [self.player, "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
        if name == "cvlc":
            return [self.player, "--play-and-exit", "--intf", "dummy", str(path)]
        return [self.player, "-q", str(path)]

    def _is_current(self, generation: int) -> bool:
        with self._state_lock:
            return generation == self._generation and not self._closed.is_set()

    def _playback_worker(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            if not self._is_current(request.generation):
                request.done.set()
                continue
            self._active.set()
            self._play_files(request)
            self._active.clear()

    def _play_files(self, request: _PlaybackRequest) -> None:
        error: str | None = None
        try:
            for path in request.files:
                if not self._is_current(request.generation):
                    return
                print(f"[voice-output] playing {path}", flush=True)
                process = subprocess.Popen(self._command(path))
                with self._state_lock:
                    self._process = process
                    cancelled = request.generation != self._generation
                if cancelled:
                    process.terminate()
                return_code = process.wait()
                with self._state_lock:
                    self._process = None
                if not self._is_current(request.generation):
                    return
                if return_code != 0:
                    raise RuntimeError(
                        f"player exited with status {return_code}: {path.name}"
                    )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            print(f"[voice-output] playback failed: {error}", flush=True)
        finally:
            with self._state_lock:
                self._process = None
            request.done.set()

        if self._is_current(request.generation):
            self._results.put(
                VoicePlaybackResult(
                    files=request.files,
                    purpose=request.purpose,
                    error=error,
                )
            )

    def poll(self) -> VoicePlaybackResult | None:
        """Return one completed sequence result without blocking."""
        try:
            return self._results.get_nowait()
        except Empty:
            return None

    def cancel(self) -> None:
        """Stop playback and discard any unconsumed completion."""
        with self._state_lock:
            self._generation += 1
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        while True:
            try:
                request = self._requests.get_nowait()
            except Empty:
                break
            if request is not None:
                request.done.set()
        while True:
            try:
                self._results.get_nowait()
            except Empty:
                break

    def close(self) -> None:
        if self._closed.is_set():
            return
        self.cancel()
        self._closed.set()
        self._requests.put(None)
        self._thread.join(timeout=2.0)
