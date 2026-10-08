#!/usr/bin/env python3
"""
Online force/current tuner for RIGHT Daimon gripper.

Right gripper:
    192.168.10.11:55551

Right tactile sensors:
    Tactile 1 -> 192.168.10.11:50052
    Tactile 2 -> 192.168.10.11:50051

Keyboard:
    L       grasp with current target
    P       release
    +       stronger, increase current target
    -       softer, decrease current target
    [       fine step = 1
    ]       coarse step = 10
    R       reset tactile sensors
    Q/ESC   quit

IMPORTANT:
    current_threshold is NOT force in Newtons.
    It is used as a motor-load/current proxy for grasp force.
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from daimon_gripper.dm_gripper_tac_py.tac import (
    SensorManager,
    disable_proxy_for_host,
    make_dashboard as make_tactile_dashboard,
    make_options as make_tactile_options,
)

from grasp_core.communication.gripper_signal import (
    DEFAULT_RECEIVER_PATH,
    send_gripper_signal,
    start_gripper_signal_receiver,
    stop_gripper_signal_receiver,
)


# ============================================================
# Basic configuration
# ============================================================

WINDOW_NAME = "Right Gripper Force Tuner"

RIGHT_GRIPPER_SERVER = "192.168.10.11:55551"

KEY_GRIP = ord("l")
KEY_RELEASE = ord("p")
KEY_RESET_TACTILE = ord("r")

QUIT_KEYS = {
    ord("q"),
    27,  # ESC
}

# "+" is usually Shift + "="
KEY_FORCE_UP = {
    ord("+"),
    ord("="),
}

KEY_FORCE_DOWN = {
    ord("-"),
    ord("_"),
}

KEY_STEP_SMALL = ord("[")
KEY_STEP_LARGE = ord("]")

STATUS_HEIGHT = 150
TACTILE_TILE_SIZE = (480, 360)


# ============================================================
# Right-hand tactile configuration
# ============================================================

@dataclass(frozen=True)
class TactileSpec:
    sensor_id: int
    remote_addr: str
    dev_id: str
    pc_host: str
    pc_port: int


RIGHT_TACTILE_SPECS = [
    TactileSpec(
        sensor_id=1,
        remote_addr="192.168.10.11:50052",
        dev_id="2",
        pc_host="192.168.10.123",
        pc_port=60031,
    ),

    TactileSpec(
        sensor_id=2,
        remote_addr="192.168.10.11:50051",
        dev_id="0",
        pc_host="192.168.10.123",
        pc_port=60030,
    ),
]


# ============================================================
# Utility
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_key(key: int) -> int:
    key &= 0xFF

    if ord("A") <= key <= ord("Z"):
        return key + 32

    return key


def make_blank(
    size: tuple[int, int],
    text: str,
) -> np.ndarray:

    width, height = size

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        image,
        text,
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return image


def draw_label(
    image: np.ndarray,
    text: str,
) -> np.ndarray:

    canvas = image.copy()

    cv2.rectangle(
        canvas,
        (0, 0),
        (canvas.shape[1], 32),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        canvas,
        text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    return canvas


# ============================================================
# Build arguments expected by your existing gripper code
# ============================================================

def build_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Online current-threshold tuner "
            "for right Daimon gripper"
        )
    )

    parser.add_argument(
        "--current-threshold",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--min-threshold",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-threshold",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--force-step",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--grip-speed",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--grip-torque",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--hold-torque",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--backend",
        default="Flux",
    )

    parser.add_argument(
        "--max-fps",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    cli = parser.parse_args()

    if cli.min_threshold > cli.max_threshold:
        raise SystemExit(
            "--min-threshold must be <= --max-threshold"
        )

    args = argparse.Namespace()

    # --------------------------------------------------------
    # RIGHT gripper only
    # --------------------------------------------------------

    args.gripper_server = RIGHT_GRIPPER_SERVER
    args.right_gripper_server = RIGHT_GRIPPER_SERVER

    # API compatibility only
    args.left_gripper_server = "192.168.14.11:55551"

    args.dual_gripper = False
    args.manual_gripper_hand = "right"

    # --------------------------------------------------------
    # Existing mechanical parameters
    # --------------------------------------------------------

    args.gripper_clamp_pos = -52525
    args.gripper_open_pos = -142525

    args.gripper_max_itinerary = 90000
    args.gripper_speed_coe = 3600

    args.gripper_calibration_tolerance = 150

    args.gripper_connect_attempts = 3
    args.gripper_connect_timeout_sec = 5.0
    args.gripper_connect_retry_delay_sec = 0.5

    args.gripper_allow_homing_fallback = True

    # --------------------------------------------------------
    # Force/current control
    # --------------------------------------------------------

    args.gripper_min_pos = 300
    args.gripper_max_pos = 900

    args.gripper_grip_speed = int(
        clamp(
            cli.grip_speed,
            10,
            100,
        )
    )

    args.gripper_grip_torque = int(
        clamp(
            cli.grip_torque,
            10,
            100,
        )
    )

    args.gripper_hold_torque = int(
        clamp(
            cli.hold_torque,
            10,
            100,
        )
    )

    args.gripper_current_threshold = int(
        clamp(
            cli.current_threshold,
            cli.min_threshold,
            cli.max_threshold,
        )
    )

    args.gripper_poll_interval = 0.05

    args.gripper_contact_grace = 0.4

    args.gripper_progress_epsilon = 2

    args.gripper_stall_samples = 5

    args.gripper_timeout = 20.0

    # --------------------------------------------------------
    # Release
    # --------------------------------------------------------

    args.gripper_release_target = 1000

    args.gripper_release_speed = 40

    args.gripper_release_torque = 20

    args.gripper_release_wait = 0.5

    # --------------------------------------------------------
    # Local signal receiver
    # --------------------------------------------------------

    args.grip_signal_host = "127.0.0.1"

    args.grip_signal_port = 55660

    args.grip_signal_timeout_sec = 1.0

    args.grip_signal_command_timeout_sec = 60.0

    args.grip_signal_auto_start = True

    args.grip_signal_receiver_path = str(
        DEFAULT_RECEIVER_PATH
    )

    args.grip_signal_token = None

    # --------------------------------------------------------
    # Tactile
    # --------------------------------------------------------

    args.backend = (
        "Flux"
        if str(cli.backend).strip().lower() == "flux"
        else cli.backend
    )

    args.max_fps = cli.max_fps

    args.force = cli.force

    # --------------------------------------------------------
    # Online tuner
    # --------------------------------------------------------

    args.tune_min_threshold = int(
        cli.min_threshold
    )

    args.tune_max_threshold = int(
        cli.max_threshold
    )

    args.tune_step = max(
        1,
        int(cli.force_step),
    )

    return args


# ============================================================
# Tactile worker
# ============================================================

def make_tactile_args(
    spec: TactileSpec,
    args: argparse.Namespace,
) -> argparse.Namespace:

    return argparse.Namespace(
        dev_id=spec.dev_id,
        backend=args.backend,
        remote_addr=spec.remote_addr,
        pc_host=spec.pc_host,
        pc_port=spec.pc_port,
        max_fps=args.max_fps,
        force=args.force,
    )


class TactileWorker:

    def __init__(
        self,
        spec: TactileSpec,
        args: argparse.Namespace,
    ):

        self.spec = spec
        self.args = args

        self.sensor: SensorManager | None = None

        self.lock = threading.Lock()

        self.stop_event = threading.Event()

        self.thread: threading.Thread | None = None

        self.latest_data: dict | None = None

        self.error: str | None = None

        self.opening = False

    def open(self):

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):
            return

        self.opening = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()

    def _run(self):

        try:

            disable_proxy_for_host(
                self.args.remote_addr
            )

            self.sensor = SensorManager(
                make_tactile_options(
                    self.args
                )
            )

            with self.lock:

                self.opening = False
                self.error = None

        except Exception as exc:

            with self.lock:

                self.opening = False
                self.error = str(exc)

            return

        while not self.stop_event.is_set():

            try:

                if (
                    str(
                        self.args.backend
                    ).strip().lower()
                    == "flux"
                ):
                    self.sensor.sensor.getEvents()

                if not self.sensor.update():
                    continue

                data = self.sensor.read()

                with self.lock:

                    self.latest_data = data
                    self.error = None

            except Exception as exc:

                with self.lock:
                    self.error = str(exc)

                time.sleep(0.1)

    def dashboard(self) -> np.ndarray:

        with self.lock:

            data = (
                dict(self.latest_data)
                if self.latest_data is not None
                else None
            )

            error = self.error

            opening = self.opening

        if data is None:

            if opening:

                text = (
                    f"Tactile "
                    f"{self.spec.sensor_id}: opening"
                )

            elif error:

                text = (
                    f"Tactile "
                    f"{self.spec.sensor_id}: "
                    f"{error}"
                )

            else:

                text = (
                    f"Tactile "
                    f"{self.spec.sensor_id}: waiting"
                )

            return make_blank(
                TACTILE_TILE_SIZE,
                text,
            )

        return make_tactile_dashboard(
            data
        )

    def reset(self) -> str:

        if self.sensor is None:

            return (
                f"Tactile "
                f"{self.spec.sensor_id} "
                f"not ready"
            )

        try:

            self.sensor.reset()

            return (
                f"T{self.spec.sensor_id} reset"
            )

        except Exception as exc:

            return (
                f"T{self.spec.sensor_id} "
                f"reset failed: {exc}"
            )

    def close(self):

        self.stop_event.set()

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):

            self.thread.join(
                timeout=1.0
            )

        if self.sensor is not None:

            self.sensor.close()


# ============================================================
# Main force tuner
# ============================================================

class ForceTunerApp:

    def __init__(
        self,
        args: argparse.Namespace,
    ):

        self.args = args

        # Current target selected by user
        self.target_threshold = int(
            args.gripper_current_threshold
        )

        self.step = int(
            args.tune_step
        )

        # ----------------------------------------------------
        # Right tactile sensors
        # ----------------------------------------------------

        self.tactile_workers = [

            TactileWorker(
                spec,
                make_tactile_args(
                    spec,
                    args,
                ),
            )

            for spec
            in RIGHT_TACTILE_SPECS
        ]

        # ----------------------------------------------------
        # Gripper communication
        # ----------------------------------------------------

        self.receiver = None

        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=(
                "right-gripper-command"
            ),
        )

        self.future: Future | None = None

        self.last_command = "none"

        self.status = "Ready"

        # If '+' is pressed while a command is running,
        # remember that a stronger grasp is requested.
        self.pending_strengthen = False

        # '-' cannot physically unload the existing grasp
        # with the currently exposed API.
        self.lower_target_pending = False

    # --------------------------------------------------------
    # Startup / shutdown
    # --------------------------------------------------------

    def open(self):

        for worker in self.tactile_workers:

            worker.open()

            time.sleep(0.2)

        self.receiver = (
            start_gripper_signal_receiver(
                self.args
            )
        )

        self.status = (
            "Ready | "
            "L grip | "
            "P release | "
            "+/- threshold | "
            "[/] step | "
            "R reset | "
            "Q quit"
        )

    def close(self):

        for worker in self.tactile_workers:

            worker.close()

        if self.receiver is not None:

            stop_gripper_signal_receiver(
                self.receiver
            )

        self.executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

        cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Gripper state
    # --------------------------------------------------------

    def busy(self) -> bool:

        return (
            self.future is not None
            and not self.future.done()
        )

    def send_command(
        self,
        command: str,
    ):

        if self.busy():

            self.status = (
                f"Busy: {self.last_command}"
            )

            return

        # Synchronize selected target into
        # the normal configuration object.
        self.args.gripper_current_threshold = int(
            self.target_threshold
        )

        self.last_command = command

        self.status = (
            f"{command.upper()} running | "
            f"target current="
            f"{self.target_threshold}"
        )

        self.future = (
            self.executor.submit(
                send_gripper_signal,
                command,
                self.args,
                "right",
            )
        )

    # --------------------------------------------------------
    # Grasp / release
    # --------------------------------------------------------

    def grip(self):

        self.lower_target_pending = False

        self.send_command(
            "grip"
        )

    def release(self):

        self.pending_strengthen = False

        self.lower_target_pending = False

        self.send_command(
            "release"
        )

    # --------------------------------------------------------
    # Increase force
    # --------------------------------------------------------

    def increase_force(self):

        old = self.target_threshold

        self.target_threshold = min(
            self.target_threshold
            + self.step,

            int(
                self.args.tune_max_threshold
            ),
        )

        if self.target_threshold == old:

            self.status = (
                f"Already at max "
                f"{old}"
            )

            return

        self.args.gripper_current_threshold = int(
            self.target_threshold
        )

        if self.busy():

            self.pending_strengthen = True

            self.status = (
                f"Target +: "
                f"{old} -> "
                f"{self.target_threshold} | "
                f"queued"
            )

        else:

            self.status = (
                f"Target +: "
                f"{old} -> "
                f"{self.target_threshold}"
            )

            # Try another grip command with
            # the new requested setting.
            self.send_command(
                "grip"
            )

    # --------------------------------------------------------
    # Decrease force
    # --------------------------------------------------------

    def decrease_force(self):

        old = self.target_threshold

        self.target_threshold = max(
            self.target_threshold
            - self.step,

            int(
                self.args.tune_min_threshold
            ),
        )

        if self.target_threshold == old:

            self.status = (
                f"Already at min "
                f"{old}"
            )

            return

        self.args.gripper_current_threshold = int(
            self.target_threshold
        )

        self.lower_target_pending = True

        # IMPORTANT:
        #
        # The code supplied by you exposes only:
        #
        #     grip
        #     release
        #
        # It does not expose a "small open step".
        #
        # Therefore we MUST NOT pretend that lowering the
        # software threshold physically reduces the force
        # already applied to the object.

        self.status = (
            f"Target -: "
            f"{old} -> "
            f"{self.target_threshold} | "
            f"P then L to physically "
            f"reduce force"
        )

    # --------------------------------------------------------
    # Adjustment resolution
    # --------------------------------------------------------

    def set_small_step(self):

        self.step = 1

        self.status = (
            "Fine adjustment: step=1"
        )

    def set_large_step(self):

        self.step = 10

        self.status = (
            "Coarse adjustment: step=10"
        )

    # --------------------------------------------------------
    # Collect async gripper result
    # --------------------------------------------------------

    def collect_result(self):

        if (
            self.future is None
            or not self.future.done()
        ):
            return

        try:

            result = (
                self.future.result()
            )

            self.status = str(
                result
            )

        except Exception as exc:

            self.status = (
                f"Command failed: "
                f"{exc}"
            )

        finished_command = (
            self.last_command
        )

        self.future = None

        # User may hold '+' while the grip
        # command is still running.
        #
        # Once finished, use the newest target.
        if (
            finished_command == "grip"
            and self.pending_strengthen
            and not self.lower_target_pending
        ):

            self.pending_strengthen = False

            self.send_command(
                "grip"
            )

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------

    def render(self) -> np.ndarray:

        tiles = []

        for worker in self.tactile_workers:

            dashboard = (
                worker.dashboard()
            )

            dashboard = cv2.resize(
                dashboard,
                TACTILE_TILE_SIZE,
                interpolation=cv2.INTER_AREA,
            )

            dashboard = draw_label(
                dashboard,
                (
                    f"Right tactile "
                    f"{worker.spec.sensor_id}"
                ),
            )

            tiles.append(
                dashboard
            )

        tactile_panel = np.hstack(
            tiles
        )

        width = (
            tactile_panel.shape[1]
        )

        status_panel = np.zeros(
            (
                STATUS_HEIGHT,
                width,
                3,
            ),
            dtype=np.uint8,
        )

        minimum = int(
            self.args.tune_min_threshold
        )

        maximum = int(
            self.args.tune_max_threshold
        )

        span = max(
            1,
            maximum - minimum,
        )

        ratio = (
            self.target_threshold
            - minimum
        ) / span

        # ----------------------------------------------------
        # Force/current bar
        # ----------------------------------------------------

        bar_x0 = 20

        bar_x1 = (
            width - 20
        )

        bar_y0 = 62
        bar_y1 = 88

        fill_x = int(
            bar_x0
            + ratio
            * (
                bar_x1
                - bar_x0
            )
        )

        cv2.rectangle(
            status_panel,
            (bar_x0, bar_y0),
            (bar_x1, bar_y1),
            (80, 80, 80),
            1,
        )

        cv2.rectangle(
            status_panel,
            (bar_x0, bar_y0),
            (fill_x, bar_y1),
            (220, 220, 220),
            -1,
        )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        line1 = (
            "RIGHT 192.168.10.11"
            f" | target_current="
            f"{self.target_threshold}"
            f" | range="
            f"{minimum}-{maximum}"
            f" | step={self.step}"
        )

        line2 = (
            "L grip | "
            "P release | "
            "+ stronger | "
            "- lower target | "
            "[ fine | "
            "] coarse | "
            "R reset | "
            "Q quit"
        )

        line3 = (
            self.status[:150]
        )

        cv2.putText(
            status_panel,
            line1,
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            status_panel,
            line2,
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            status_panel,
            line3,
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

        return np.vstack(
            [
                tactile_panel,
                status_panel,
            ]
        )

    # --------------------------------------------------------
    # Keyboard
    # --------------------------------------------------------

    def handle_key(
        self,
        key: int,
    ) -> bool:

        key = normalize_key(
            key
        )

        if key in QUIT_KEYS:

            return False

        if key == KEY_GRIP:

            self.grip()

        elif key == KEY_RELEASE:

            self.release()

        elif key in KEY_FORCE_UP:

            self.increase_force()

        elif key in KEY_FORCE_DOWN:

            self.decrease_force()

        elif key == KEY_STEP_SMALL:

            self.set_small_step()

        elif key == KEY_STEP_LARGE:

            self.set_large_step()

        elif key == KEY_RESET_TACTILE:

            results = [
                worker.reset()
                for worker
                in self.tactile_workers
            ]

            self.status = (
                "; ".join(results)
            )

        return True

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    def run(self):

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        while True:

            self.collect_result()

            cv2.imshow(
                WINDOW_NAME,
                self.render(),
            )

            # Holding a key normally produces repeated
            # keyboard events at OS keyboard-repeat rate.
            key = cv2.waitKey(10)

            if key == -1:
                continue

            if not self.handle_key(key):
                break


# ============================================================
# Main
# ============================================================

def main() -> int:

    args = build_args()

    app = ForceTunerApp(
        args
    )

    try:

        app.open()

        app.run()

    finally:

        app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )