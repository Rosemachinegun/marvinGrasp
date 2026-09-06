"""夹持后至放置前，利用夹爪闭合位移检测物体掉落。"""

from __future__ import annotations

import argparse
import re
import threading
import time
from dataclasses import dataclass

from grasp_core.communication.gripper_signal import send_gripper_signal


_POSITION_RE = re.compile(r"\bpos=(-?\d+)\b")


def feedback_position(status: object) -> int | None:
    """Extract the logical gripper position from a receiver feedback reply."""
    match = _POSITION_RE.search(str(status))
    return int(match.group(1)) if match else None


@dataclass
class GraspDropMonitor:
    """Poll position while compliant close force remains active."""

    args: argparse.Namespace
    hand: str
    baseline_position: int
    publisher: object

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = False
        self.detected_position: int | None = None

    @property
    def threshold(self) -> int:
        return max(int(getattr(self.args, "grip_drop_close_delta", 30)), 0)

    def start(self) -> None:
        if not bool(getattr(self.args, "grip_drop_detection", True)):
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self.hand}-grasp-drop-monitor",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        interval = max(float(getattr(self.args, "grip_drop_poll_interval", 0.05)), 0.02)
        trip_position = self.baseline_position - self.threshold
        print(
            "[grip_drop] monitoring "
            f"hand={self.hand} baseline={self.baseline_position} "
            f"delta={self.threshold} trip_pos={trip_position}",
            flush=True,
        )
        while not self._stop.wait(interval):
            position = feedback_position(
                send_gripper_signal("feedback", self.args, hand=self.hand)
            )
            if self._stop.is_set():
                return
            if position is None or position == -1:
                continue
            # Logical position decreases in the closing direction.
            if self.baseline_position - position > self.threshold:
                self.dropped = True
                self.detected_position = position
                print(
                    "[grip_drop] DROP DETECTED; stopping put trajectory "
                    f"hand={self.hand} baseline={self.baseline_position} "
                    f"pos={position} close_delta={self.baseline_position - position} "
                    f"target={self.threshold}",
                    flush=True,
                )
                request_stop = getattr(self.publisher, "request_stop", None)
                if callable(request_stop):
                    request_stop()
                return


def read_grasp_baseline(args: argparse.Namespace, hand: str) -> int | None:
    """Record the successful-grasp position immediately before transport."""
    return feedback_position(send_gripper_signal("feedback", args, hand=hand))
