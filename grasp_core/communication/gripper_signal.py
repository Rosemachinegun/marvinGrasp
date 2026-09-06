#!/usr/bin/env python3
"""通信层：负责夹爪 receiver 的启动/停止，以及 grip/release socket 命令发送。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

DEFAULT_DAIMON_STUFF_DIR = PROJECT_ROOT / "daimon_stuff"
DEFAULT_RECEIVER_PATH = DEFAULT_DAIMON_STUFF_DIR / "grip_signal_receiver.py"

_RECEIVER_OPTION_MAP = {
    "gripper_hand": "--hand",
    "gripper_server": "--server",
    "gripper_clamp_pos": "--clamp-pos",
    "gripper_open_pos": "--open-pos",
    "gripper_max_itinerary": "--max-itinerary",
    "gripper_speed_coe": "--speed-coe",
    "gripper_calibration_tolerance": "--calibration-tolerance",
    "gripper_connect_attempts": "--connect-attempts",
    "gripper_connect_timeout_sec": "--connect-timeout-sec",
    "gripper_connect_retry_delay_sec": "--connect-retry-delay-sec",
    "gripper_allow_homing_fallback": "--allow-homing-fallback",
    "gripper_min_pos": "--min-pos",
    "gripper_max_pos": "--max-pos",
    "gripper_grip_speed": "--grip-speed",
    "gripper_grip_torque": "--grip-torque",
    "gripper_hold_torque": "--hold-torque",
    "gripper_current_threshold": "--current-threshold",
    "gripper_poll_interval": "--poll-interval",
    "gripper_contact_grace": "--contact-grace",
    "gripper_progress_epsilon": "--progress-epsilon",
    "gripper_stall_samples": "--stall-samples",
    "gripper_empty_grip_margin": "--empty-grip-margin",
    "gripper_target_pos_tolerance": "--target-pos-tolerance",
    "gripper_timeout": "--timeout",
    "gripper_grip_done_wait": "--grip-done-wait",
    "gripper_release_target": "--release-target",
    "gripper_release_speed": "--release-speed",
    "gripper_release_torque": "--release-torque",
    "gripper_release_wait": "--release-wait",
}


def _expected_receiver_status_tokens(args: argparse.Namespace) -> list[str]:
    tokens = []
    expected_attrs = {
        "gripper_hand": "hand",
        "gripper_server": "server",
        "gripper_clamp_pos": "clamp_pos",
        "gripper_open_pos": "open_pos",
        "gripper_max_itinerary": "max_itinerary",
        "gripper_speed_coe": "speed_coe",
        "gripper_min_pos": "min_pos",
        "gripper_max_pos": "max_pos",
        "gripper_release_target": "release_target",
        "gripper_empty_grip_margin": "empty_grip_margin",
        "gripper_target_pos_tolerance": "target_pos_tolerance",
        "gripper_grip_done_wait": "grip_done_wait",
        "gripper_connect_attempts": "connect_attempts",
        "gripper_allow_homing_fallback": "allow_homing_fallback",
    }
    for attr, name in expected_attrs.items():
        if hasattr(args, attr):
            tokens.append(f"{name}={getattr(args, attr)}")
    if hasattr(args, "gripper_connect_timeout_sec"):
        tokens.append(
            f"connect_timeout_sec={float(args.gripper_connect_timeout_sec):.2f}"
        )
    if hasattr(args, "gripper_connect_retry_delay_sec"):
        tokens.append(
            "connect_retry_delay_sec="
            f"{float(args.gripper_connect_retry_delay_sec):.2f}"
        )
    return tokens


def _warn_receiver_config_mismatch(reply: str, args: argparse.Namespace) -> None:
    if "server=" not in reply:
        print(
            "[gripper] existing receiver status does not include config; "
            "restart the receiver if gripper behavior differs between apps",
            flush=True,
        )
        return
    missing = [
        token for token in _expected_receiver_status_tokens(args) if token not in reply
    ]
    if missing:
        print(
            "[gripper] existing receiver config differs from requested args; "
            f"missing/status-mismatch tokens: {', '.join(missing)}",
            flush=True,
        )


def _receiver_message(command: str, args: argparse.Namespace) -> str:
    token = str(getattr(args, "grip_signal_token", "") or "").strip()
    if token:
        return f"{token} {command}\n"
    return f"{command}\n"


def _disable_proxy_for_host(addr: str) -> None:
    host = urlsplit(addr if "://" in addr else "//" + addr).hostname
    if not host:
        return
    for key in ("NO_PROXY", "no_proxy"):
        values = os.environ.get(key, "")
        hosts = [item.strip() for item in values.split(",") if item.strip()]
        if host not in hosts:
            hosts.append(host)
        os.environ[key] = ",".join(hosts)


def _query_gripper_receiver(
    args: argparse.Namespace,
    command: str = "status",
    *,
    reply_timeout_sec: float | None = None,
) -> str:
    host = str(args.grip_signal_host)
    port = int(args.grip_signal_port)
    timeout_sec = max(float(args.grip_signal_timeout_sec), 0.01)
    with socket.create_connection((host, port), timeout=timeout_sec) as sock:
        sock.sendall(_receiver_message(command, args).encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        sock.settimeout(max(float(reply_timeout_sec or timeout_sec), 0.01))
        return sock.recv(4096).decode("utf-8", errors="replace").strip()


def _dual_gripper_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "dual_gripper", False))


def _receiver_args(
    args: argparse.Namespace,
    *,
    hand: str,
    port: int,
    server: str,
) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.gripper_hand = hand
    copied.grip_signal_port = port
    copied.gripper_server = server
    if hand == "left":
        for name in (
            "clamp_pos",
            "open_pos",
            "max_itinerary",
            "speed_coe",
            "min_pos",
            "max_pos",
            "grip_speed",
            "grip_torque",
            "hold_torque",
            "current_threshold",
            "poll_interval",
            "contact_grace",
            "progress_epsilon",
            "stall_samples",
            "empty_grip_margin",
            "target_pos_tolerance",
            "timeout",
            "grip_done_wait",
            "release_target",
            "release_speed",
            "release_torque",
            "release_wait",
        ):
            left_attr = f"left_gripper_{name}"
            if hasattr(copied, left_attr):
                setattr(copied, f"gripper_{name}", getattr(copied, left_attr))
    return copied


def gripper_receiver_args(
    args: argparse.Namespace,
    hand: str | None = None,
) -> list[tuple[str, argparse.Namespace]]:
    hand = str(hand or "").strip().lower()
    if hand and hand not in {"left", "right"}:
        raise ValueError(f"hand must be 'left', 'right', or None; got {hand!r}")

    if not _dual_gripper_enabled(args):
        return [("gripper", args)]

    base_port = int(args.grip_signal_port)
    endpoints = [
        (
            "left",
            _receiver_args(
                args,
                hand="left",
                port=base_port,
                server=str(
                    getattr(
                        args,
                        "left_gripper_server",
                        getattr(args, "gripper_server", ""),
                    )
                ),
            ),
        ),
        (
            "right",
            _receiver_args(
                args,
                hand="right",
                port=base_port + 1,
                server=str(
                    getattr(
                        args,
                        "right_gripper_server",
                        getattr(args, "gripper_server", ""),
                    )
                ),
            ),
        ),
    ]
    if hand:
        endpoints = [endpoint for endpoint in endpoints if endpoint[0] == hand]
    return endpoints


def _start_single_gripper_signal_receiver(
    args: argparse.Namespace,
    *,
    label: str,
) -> subprocess.Popen | None:
    if not bool(getattr(args, "grip_signal_auto_start", True)):
        print(
            f"[gripper:{label}] auto-start disabled; using existing receiver if any",
            flush=True,
        )
        return None

    if hasattr(args, "gripper_server"):
        _disable_proxy_for_host(str(args.gripper_server))

    host = str(args.grip_signal_host)
    port = int(args.grip_signal_port)
    try:
        reply = _query_gripper_receiver(args)
    except OSError:
        reply = ""
    else:
        print(
            f"[gripper:{label}] receiver already listening on {host}:{port}: {reply}",
            flush=True,
        )
        _warn_receiver_config_mismatch(reply, args)
        _check_single_gripper_signal_receiver(args, label=label)
        return None

    receiver_path = Path(
        getattr(args, "grip_signal_receiver_path", DEFAULT_RECEIVER_PATH)
    ).expanduser()
    if not receiver_path.is_absolute():
        receiver_path = PROJECT_ROOT / receiver_path
    receiver_path = receiver_path.resolve()
    if not receiver_path.exists():
        print(f"[gripper:{label}] receiver not found: {receiver_path}", flush=True)
        return None

    command = [
        sys.executable,
        str(receiver_path),
        "--host",
        host,
        "--port",
        str(port),
    ]
    token = str(getattr(args, "grip_signal_token", "") or "").strip()
    if token:
        command.extend(["--token", token])
    for attr, option in _RECEIVER_OPTION_MAP.items():
        if not hasattr(args, attr):
            continue
        value = getattr(args, attr)
        if value is None:
            continue
        if attr == "gripper_allow_homing_fallback":
            command.append(option if bool(value) else "--no-allow-homing-fallback")
            continue
        command.extend([option, str(value)])

    print(
        f"[gripper:{label}] starting receiver on {host}:{port}: {receiver_path}",
        flush=True,
    )
    process = subprocess.Popen(command, cwd=str(receiver_path.parent))
    deadline = time.monotonic() + max(float(args.grip_signal_timeout_sec), 0.5)
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(
                f"[gripper:{label}] receiver exited early with code {process.returncode}",
                flush=True,
            )
            return None
        try:
            reply = _query_gripper_receiver(args)
        except OSError as exc:
            last_error = str(exc)
            time.sleep(0.05)
            continue
        print(f"[gripper:{label}] receiver ready: {reply}", flush=True)
        _check_single_gripper_signal_receiver(args, label=label)
        return process

    print(f"[gripper:{label}] receiver start timed out: {last_error}", flush=True)
    return process


def _check_single_gripper_signal_receiver(
    args: argparse.Namespace,
    *,
    label: str,
) -> str:
    command_timeout_sec = max(
        float(getattr(args, "grip_signal_command_timeout_sec", 60.0)),
        0.01,
    )
    try:
        reply = _query_gripper_receiver(
            args,
            command="check",
            reply_timeout_sec=command_timeout_sec,
        )
    except OSError as exc:
        status = f"ERR startup check failed: {exc}"
        print(f"[gripper:{label}] {status}", flush=True)
        return status
    action = "startup check OK" if reply.startswith("OK") else "startup check failed"
    print(f"[gripper:{label}] {action}: {reply}", flush=True)
    if "command must be" in reply and "check" not in reply:
        print(
            f"[gripper:{label}] existing receiver is from an older version; "
            "restart flowpose_request_ik_tester.py so startup checks are available",
            flush=True,
        )
    return reply


def start_gripper_signal_receiver(
    args: argparse.Namespace,
) -> subprocess.Popen | list[subprocess.Popen] | None:
    receivers = [
        _start_single_gripper_signal_receiver(receiver_args, label=label)
        for label, receiver_args in gripper_receiver_args(args)
    ]
    receivers = [process for process in receivers if process is not None]
    if _dual_gripper_enabled(args):
        return receivers
    return receivers[0] if receivers else None


def stop_gripper_signal_receiver(
    process: subprocess.Popen | list[subprocess.Popen] | tuple[subprocess.Popen, ...] | None,
) -> None:
    if isinstance(process, (list, tuple)):
        for item in process:
            stop_gripper_signal_receiver(item)
        return
    if process is None or process.poll() is not None:
        return
    print("[gripper] stopping receiver", flush=True)
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)

def _send_single_gripper_signal(
    command: str,
    args: argparse.Namespace,
    *,
    label: str,
) -> str:
    command = str(command).strip()
    if command not in {"grip", "release", "check", "feedback"}:
        status = f"Invalid gripper command: {command!r}"
        print(f"[gripper:{label}] {status}", flush=True)
        return status

    host = str(args.grip_signal_host)
    port = int(args.grip_signal_port)
    timeout_sec = max(float(args.grip_signal_timeout_sec), 0.01)
    try:
        with socket.create_connection((host, port), timeout=timeout_sec) as sock:
            sock.sendall(_receiver_message(command, args).encode("utf-8"))
            sock.shutdown(socket.SHUT_WR)
            command_timeout_sec = max(
                float(getattr(args, "grip_signal_command_timeout_sec", 60.0)), 0.01
            )
            sock.settimeout(command_timeout_sec)
            reply = sock.recv(4096).decode("utf-8", errors="replace").strip()
    except OSError as exc:
        status = f"Failed to send gripper {command!r} to {host}:{port}: {exc}"
        print(f"[gripper:{label}] {status}", flush=True)
        return status

    ok = not reply.startswith("ERR")
    action = "Sent" if ok else "Gripper command failed"
    status = f"{action} {label} gripper {command} to {host}:{port}"
    if reply:
        status += f": {reply}"
    print(f"[gripper:{label}] {status}", flush=True)
    return status


def _gripper_status_failed(status: str) -> bool:
    return (
        "Invalid gripper command" in status
        or "Failed to send" in status
        or "Gripper command failed" in status
        or "ERR " in status
        or "failed exit_code=" in status
    )


def send_gripper_signal(
    command: str,
    args: argparse.Namespace,
    hand: str | None = None,
) -> str:
    try:
        endpoints = gripper_receiver_args(args, hand=hand)
    except ValueError as exc:
        status = f"Invalid gripper target: {exc}"
        print(f"[gripper] {status}", flush=True)
        return status

    if not endpoints:
        status = f"No gripper endpoint for hand={hand!r}"
        print(f"[gripper] {status}", flush=True)
        return status

    if len(endpoints) == 1:
        label, receiver_args = endpoints[0]
        return _send_single_gripper_signal(command, receiver_args, label=label)

    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
        futures = [
            executor.submit(
                _send_single_gripper_signal,
                command,
                receiver_args,
                label=label,
            )
            for label, receiver_args in endpoints
        ]
        results = [
            f"{label}: {future.result()}"
            for (label, _receiver_args), future in zip(endpoints, futures, strict=True)
        ]

    failed = any(_gripper_status_failed(result) for result in results)
    prefix = "ERR dual gripper" if failed else "Sent dual gripper"
    status = f"{prefix} {str(command).strip()}: " + " | ".join(results)
    print(f"[gripper] {status}", flush=True)
    return status
