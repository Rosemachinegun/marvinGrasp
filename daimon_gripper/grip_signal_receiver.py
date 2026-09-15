#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''夹爪信号接收器：监听 TCP 信号并调用夹爪 runner 执行夹取/释放动作。'''
from __future__ import annotations

import argparse
import socketserver
import sys
import threading
import time

try:
    from .grip_signal_calibration import apply_cached_calibration
    from .grip_signal_config import (
        ACTION_COMMANDS,
        COMMAND_ERROR,
        add_gripper_args,
        gripper_config_summary,
        normalize_args,
    )
    from .grip_signal_device import (
        LingkongGrip,
        get_init_error,
        init_calibrated_gripper,
        init_gripper,
        recover_motor_fault,
    )
    from .grip_signal_runner import GripperRunner, run_continuous_grasp, run_grip
    from .grip_signal_utils import read_feedback_sample
except ImportError:
    from grip_signal_calibration import apply_cached_calibration
    from grip_signal_config import (
        ACTION_COMMANDS,
        COMMAND_ERROR,
        add_gripper_args,
        gripper_config_summary,
        normalize_args,
    )
    from grip_signal_device import (
        LingkongGrip,
        get_init_error,
        init_calibrated_gripper,
        init_gripper,
        recover_motor_fault,
    )
    from grip_signal_runner import GripperRunner, run_continuous_grasp, run_grip
    from grip_signal_utils import read_feedback_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lingkong gripper TCP receiver and direct command tool."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55660)
    parser.add_argument("--token")
    parser.add_argument(
        "--command",
        choices=("serve", "grip", "release", "status", "check", "feedback"),
        default="serve",
        help=(
            "serve listens for TCP commands; grip/release run once and exit; "
            "check initializes and reads status without commanding motion; "
            "feedback reads one gripper feedback sample."
        ),
    )
    add_gripper_args(parser)
    args = parser.parse_args()
    apply_cached_calibration(args)
    normalize_args(args)
    return args


class GripSignalReceiver:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.runner = GripperRunner(args)
        self.token = args.token
        self.lock = threading.Lock()
        self.running = False
        self.last_result = "idle"
        self.feedback_grip: LingkongGrip | None = None

    def close_feedback_grip(self) -> None:
        grip = self.feedback_grip
        self.feedback_grip = None
        if grip is None:
            return
        try:
            grip.close(reset_torque=False)
        except Exception as exc:  # noqa: BLE001
            print(f"关闭 feedback 夹爪连接失败: {exc}", flush=True)

    def ensure_feedback_grip(self) -> tuple[LingkongGrip | None, str]:
        if self.feedback_grip is not None:
            return self.feedback_grip, "cached"
        grip = init_gripper(self.args, command="check")
        if grip is None:
            return None, "failed"
        self.feedback_grip = grip
        return grip, "new"

    def run_cached_motion(self, command: str) -> int:
        grip, _source = self.ensure_feedback_grip()
        if grip is None:
            return 1
        if command == "grip":
            return self.runner.grip(grip)
        if command == "release":
            return self.runner.release(grip)
        raise ValueError(f"unsupported cached motion command: {command}")

    def run_feedback(self) -> tuple[int, str]:
        if bool(getattr(self.args, "calibrate_each_command", False)):
            return self.runner.feedback()

        grip, source = self.ensure_feedback_grip()
        if grip is None:
            detail = get_init_error(self.args)
            detail_suffix = f" detail={detail}" if detail else ""
            return 1, f"feedback source={source} failed{detail_suffix}"

        message = read_feedback_sample(grip, source=source)
        return 0, message

    def command_result(
        self, command: str, code: int, detail_message: str = ""
    ) -> tuple[bool, str]:
        if command == "grip" and code == 2:
            return True, "GRASP_FAILED_MIN_LIMIT grip done exit_code=2"
        if command == "feedback" and code == 0:
            return True, detail_message
        if code == 0:
            return True, f"{command} done exit_code={code}"
        return False, f"{command} failed exit_code={code}"

    def run_command(self, command: str) -> tuple[bool, str]:
        with self.lock:
            if self.running:
                return False, "BUSY command already running"
            self.running = True
            self.last_result = f"{command} running"

        if command != "feedback":
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] receive {command} signal",
                flush=True,
            )
        try:
            detail_message = ""
            if command in {"grip", "release"}:
                code = self.run_cached_motion(command)
            elif command == "check":
                self.close_feedback_grip()
                code = self.runner.check()
            elif command == "feedback":
                code, detail_message = self.run_feedback()
            else:
                return False, COMMAND_ERROR

            ok, message = self.command_result(
                command, code, detail_message if command == "feedback" else ""
            )
            if not ok:
                detail = get_init_error(self.args)
                if detail:
                    message += f" detail={detail}"
            if command != "feedback" or not ok:
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}",
                    flush=True,
                )
            with self.lock:
                self.last_result = message
            return ok, f"{'OK' if ok else 'ERR'} {message}"
        except Exception as exc:
            self.close_feedback_grip()
            message = f"ERROR {exc}"
            print(message, flush=True)
            with self.lock:
                self.last_result = message
            return False, f"ERR {message}"
        finally:
            with self.lock:
                self.running = False

    def status_reply(self) -> str:
        with self.lock:
            running = self.running
            last_result = self.last_result
        return (
            f"OK running={running} {gripper_config_summary(self.args)} "
            f"last={last_result}"
        )

    def handle_message(self, message: str) -> str:
        parts = message.split()
        if self.token:
            if len(parts) != 2 or parts[0] != self.token:
                return "ERR bad token or command"
            command = parts[1].lower()
        else:
            command = parts[0].lower() if parts else ""

        if command == "status":
            return self.status_reply()
        _ok, reply = self.run_command(command)
        return reply

    def run_stdin_console(self) -> None:
        for line in sys.stdin:
            command = line.strip().lower()
            if not command:
                continue
            if command == "status":
                print(self.status_reply(), flush=True)
                continue
            if command in ACTION_COMMANDS:
                _ok, reply = self.run_command(command)
                print(reply, flush=True)
                continue
            print(COMMAND_ERROR, flush=True)

    def prewarm_feedback_grip(self) -> None:
        if bool(getattr(self.args, "calibrate_each_command", False)):
            print("[gripper] startup calibration/prewarm will run once", flush=True)
            self.args.calibrate_each_command = False
            grip = init_calibrated_gripper(self.args)
            if grip is None:
                print(
                    "[gripper] startup calibration failed; commands will retry known init",
                    flush=True,
                )
            else:
                self.feedback_grip = grip
                print("[gripper] startup calibration ready source=calibrated", flush=True)
            return

        grip, source = self.ensure_feedback_grip()
        if grip is None:
            print("[gripper] startup prewarm failed; commands will retry", flush=True)
        else:
            print(f"[gripper] startup prewarm ready source={source}", flush=True)

    def serve(self) -> int:
        self.prewarm_feedback_grip()
        with ThreadedTCPServer((self.args.host, self.args.port), GripRequestHandler) as server:
            server.receiver = self
            print(
                f"grip signal receiver listening on {self.args.host}:{self.args.port}",
                flush=True,
            )
            print(f"[gripper] config {gripper_config_summary(self.args)}", flush=True)
            if self.args.token:
                print("token enabled: send '<token> grip' or '<token> release'", flush=True)
            if bool(getattr(self.args, "stdin_control", False)):
                print("[gripper] stdin text control enabled", flush=True)
                print(
                    "stdin commands: 'grip', 'release', 'feedback', 'status'",
                    flush=True,
                )
                threading.Thread(target=self.run_stdin_console, daemon=True).start()
            try:
                server.serve_forever()
            finally:
                self.close_feedback_grip()
        return 0


class GripRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(1024)
        if not raw:
            return

        message = raw.decode("utf-8", errors="replace").strip()
        parts = message.split()
        if not parts or parts[-1].lower() != "feedback":
            print(f"signal from {self.client_address}: {message!r}", flush=True)
        self.wfile.write((self.server.receiver.handle_message(message) + "\n").encode())


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(args: argparse.Namespace) -> int:
    return GripSignalReceiver(args).serve()


def main() -> int:
    args = parse_args()
    if args.command == "serve":
        return serve(args)
    return GripperRunner(args).direct_command()


if __name__ == "__main__":
    raise SystemExit(main())
