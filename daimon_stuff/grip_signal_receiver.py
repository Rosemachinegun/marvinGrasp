#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''夹爪信号接收器：监听 TCP 信号并直接调用 SDK 执行机械臂完成运动后的夹取/释放动作。'''
from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRIPPER_SDK_ROOT = PROJECT_ROOT / "daimon_stuff" / "dm_gripper_py"
if str(GRIPPER_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(GRIPPER_SDK_ROOT))

from dm_lingkong_grip_sdk import LingkongGrip


DEFAULT_SERVER = "192.168.10.11:55551"
DEFAULT_CLAMP_POS = -52525
DEFAULT_OPEN_POS = -142525
DEFAULT_MAX_ITINERARY = 90000
DEFAULT_SPEED_COE = 3600
DEFAULT_CALIBRATION_TOLERANCE = 150
DEFAULT_CONNECT_ATTEMPTS = 3
DEFAULT_CONNECT_TIMEOUT_SEC = 5.0
DEFAULT_CONNECT_RETRY_DELAY_SEC = 0.5
INIT_ERROR_ATTR = "_last_init_error"
CALIBRATION_SOURCE_ATTR = "_calibration_source"
CALIBRATION_INVALIDATED_ATTR = "_calibration_invalidated"
CALIBRATION_CACHE_PATH = PROJECT_ROOT / "daimon_stuff" / ".gripper_calibration_cache.json"


def clamp(value: int | float, low: int | float, high: int | float):
    return max(low, min(high, value))


def disable_proxy_for_host(addr: str) -> None:
    """Avoid grpc using HTTP proxy for the local gripper server."""
    host = urlsplit(addr if "://" in addr else "//" + addr).hostname
    if not host:
        return
    for key in ("NO_PROXY", "no_proxy"):
        values = os.environ.get(key, "")
        hosts = [item.strip() for item in values.split(",") if item.strip()]
        if host not in hosts:
            hosts.append(host)
        os.environ[key] = ",".join(hosts)


def add_gripper_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hand",
        choices=("right", "left"),
        default="right",
        help="Select independent right/left gripper grasp parameters.",
    )
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--clamp-pos", type=int, default=DEFAULT_CLAMP_POS)
    parser.add_argument("--open-pos", type=int, default=DEFAULT_OPEN_POS)
    parser.add_argument("--max-itinerary", type=int, default=DEFAULT_MAX_ITINERARY)
    parser.add_argument("--speed-coe", type=int, default=DEFAULT_SPEED_COE)
    parser.add_argument(
        "--calibration-tolerance", type=int, default=DEFAULT_CALIBRATION_TOLERANCE
    )
    parser.add_argument("--connect-attempts", type=int, default=DEFAULT_CONNECT_ATTEMPTS)
    parser.add_argument(
        "--connect-timeout-sec", type=float, default=DEFAULT_CONNECT_TIMEOUT_SEC
    )
    parser.add_argument(
        "--connect-retry-delay-sec",
        type=float,
        default=DEFAULT_CONNECT_RETRY_DELAY_SEC,
    )
    parser.add_argument(
        "--allow-homing-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in to SDK grip_init() if known-calibration init fails. "
            "Disabled by default because grip_init() performs a homing motion."
        ),
    )
    parser.add_argument("--min-pos", type=int, default=100)
    parser.add_argument("--max-pos", type=int, default=1000)
    parser.add_argument("--grip-speed", type=int, default=60)
    parser.add_argument("--grip-torque", type=int, default=30)
    parser.add_argument("--hold-torque", type=int, default=10)
    parser.add_argument("--current-threshold", type=int, default=120)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--contact-grace", type=float, default=0.4)
    parser.add_argument("--progress-epsilon", type=int, default=2)
    parser.add_argument("--stall-samples", type=int, default=5)
    parser.add_argument(
        "--empty-grip-margin",
        type=int,
        default=0,
        help=(
            "Treat a stalled grip at or below min_pos + this margin as an empty "
            "grip/min-limit failure instead of confirmed contact."
        ),
    )
    parser.add_argument(
        "--target-pos-tolerance",
        type=int,
        default=120,
        help="Warn when the measured SDK position is far from the commanded target.",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--grip-done-wait",
        type=float,
        default=0.05,
        help="Seconds to wait after switching to hold torque before replying to grip.",
    )
    parser.add_argument("--release-target", type=int, default=1000)
    parser.add_argument("--release-speed", type=int, default=40)
    parser.add_argument("--release-torque", type=int, default=20)
    parser.add_argument("--release-wait", type=float, default=0.5)


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


def normalize_args(args: argparse.Namespace) -> None:
    args.min_pos = int(clamp(args.min_pos, 0, 1000))
    args.max_pos = int(clamp(args.max_pos, 0, 1000))
    args.release_target = int(clamp(args.release_target, 0, 1000))
    args.grip_speed = int(clamp(args.grip_speed, 10, 100))
    args.release_speed = int(clamp(args.release_speed, 10, 100))
    args.grip_torque = int(clamp(args.grip_torque, 10, 100))
    args.release_torque = int(clamp(args.release_torque, 10, 100))
    args.hold_torque = int(clamp(args.hold_torque, 10, 100))
    args.poll_interval = max(float(args.poll_interval), 0.02)
    args.contact_grace = max(float(args.contact_grace), 0.0)
    args.progress_epsilon = max(int(args.progress_epsilon), 0)
    args.stall_samples = max(int(args.stall_samples), 1)
    args.empty_grip_margin = max(int(args.empty_grip_margin), 0)
    args.target_pos_tolerance = max(int(args.target_pos_tolerance), 0)
    args.timeout = max(float(args.timeout), 0.1)
    args.grip_done_wait = max(float(args.grip_done_wait), 0.0)
    args.release_wait = max(float(args.release_wait), 0.0)
    args.calibration_tolerance = max(int(args.calibration_tolerance), 0)
    args.connect_attempts = max(int(args.connect_attempts), 1)
    args.connect_timeout_sec = max(float(args.connect_timeout_sec), 0.1)
    args.connect_retry_delay_sec = max(float(args.connect_retry_delay_sec), 0.0)
    if args.min_pos > args.max_pos:
        raise SystemExit("--min-pos must be less than or equal to --max-pos")


def gripper_config_summary(args: argparse.Namespace) -> str:
    homing_fallback_effective = (
        bool(args.allow_homing_fallback)
        and getattr(args, CALIBRATION_SOURCE_ATTR, "") != "cache"
    )
    return (
        f"hand={args.hand} server={args.server} clamp_pos={args.clamp_pos} "
        f"open_pos={args.open_pos} max_itinerary={args.max_itinerary} "
        f"speed_coe={args.speed_coe} min_pos={args.min_pos} max_pos={args.max_pos} "
        f"empty_grip_margin={args.empty_grip_margin} "
        f"empty_limit={args.min_pos + args.empty_grip_margin} "
        f"target_pos_tolerance={args.target_pos_tolerance} "
        f"grip_done_wait={args.grip_done_wait:.2f} "
        f"release_target={args.release_target} connect_attempts={args.connect_attempts} "
        f"connect_timeout_sec={args.connect_timeout_sec:.2f} "
        f"connect_retry_delay_sec={args.connect_retry_delay_sec:.2f} "
        f"allow_homing_fallback={bool(args.allow_homing_fallback)} "
        f"homing_fallback_effective={homing_fallback_effective} "
        f"calibration_source={getattr(args, CALIBRATION_SOURCE_ATTR, 'default')}"
    )


def cache_key_for_server(server: str) -> str:
    return str(server).strip()


def load_calibration_cache() -> dict:
    try:
        with CALIBRATION_CACHE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"读取夹爪标定缓存失败: {exc}", flush=True)
        return {}
    return data if isinstance(data, dict) else {}


def apply_cached_calibration(args: argparse.Namespace) -> None:
    cache = load_calibration_cache()
    entry = cache.get(cache_key_for_server(args.server))
    if not isinstance(entry, dict):
        return
    fields = ("clamp_pos", "open_pos", "max_itinerary", "speed_coe")
    try:
        values = {field: int(entry[field]) for field in fields}
    except (KeyError, TypeError, ValueError):
        print(f"忽略无效夹爪标定缓存: server={args.server}", flush=True)
        return
    args.clamp_pos = values["clamp_pos"]
    args.open_pos = values["open_pos"]
    args.max_itinerary = values["max_itinerary"]
    args.speed_coe = values["speed_coe"]
    setattr(args, CALIBRATION_SOURCE_ATTR, "cache")
    print(
        "加载夹爪标定缓存: "
        f"hand={args.hand} server={args.server} clamp_pos={args.clamp_pos} "
        f"open_pos={args.open_pos} max_itinerary={args.max_itinerary} "
        f"speed_coe={args.speed_coe}",
        flush=True,
    )


def save_cached_calibration(args: argparse.Namespace) -> None:
    cache = load_calibration_cache()
    cache[cache_key_for_server(args.server)] = {
        "clamp_pos": int(args.clamp_pos),
        "open_pos": int(args.open_pos),
        "max_itinerary": int(args.max_itinerary),
        "speed_coe": int(args.speed_coe),
    }
    try:
        with CALIBRATION_CACHE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        print(f"保存夹爪标定缓存失败: {exc}", flush=True)
        return
    print(f"已保存夹爪标定缓存: {CALIBRATION_CACHE_PATH}", flush=True)


def invalidate_cached_calibration(args: argparse.Namespace, reason: str) -> None:
    if getattr(args, CALIBRATION_INVALIDATED_ATTR, False):
        return
    setattr(args, CALIBRATION_INVALIDATED_ATTR, True)
    server = cache_key_for_server(args.server)
    cache = load_calibration_cache()
    if server not in cache:
        return
    cache.pop(server, None)
    try:
        with CALIBRATION_CACHE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        print(
            f"删除失效夹爪标定缓存失败: server={server} reason={reason}: {exc}",
            flush=True,
        )
        return
    print(f"已删除失效夹爪标定缓存: server={server} reason={reason}", flush=True)


def set_init_error(args: argparse.Namespace, message: str) -> None:
    setattr(args, INIT_ERROR_ATTR, message)
    if message:
        print(message, flush=True)


def get_init_error(args: argparse.Namespace) -> str:
    return str(getattr(args, INIT_ERROR_ATTR, "") or "").strip()


def init_homing_fallback(
    args: argparse.Namespace,
    grip: LingkongGrip,
    *,
    command: str,
) -> LingkongGrip | None:
    if (
        getattr(args, CALIBRATION_SOURCE_ATTR, "") == "cache"
        and not getattr(args, CALIBRATION_INVALIDATED_ATTR, False)
    ):
        set_init_error(args, "cached calibration init failed; homing fallback disabled")
        return None
    if not bool(args.allow_homing_fallback):
        return None
    print(
        "已知标定初始化失败，fallback 到 SDK grip_init()；会执行找零动作",
        flush=True,
    )
    if not grip.grip_init():
        set_init_error(args, "SDK grip_init fallback failed")
        grip.close(reset_torque=True)
        return None
    learn_calibration_from_grip(args, grip)
    pos = grip.read_pos()
    if command == "grip":
        grip.set_speed(args.release_speed)
        grip.set_torque_limit(args.release_torque)
        print(
            "fallback 初始化完成；先打开夹爪再执行力控闭合 "
            f"target={args.release_target}",
            flush=True,
        )
        grip.move_to_pos(args.release_target)
        time.sleep(args.release_wait)
    set_init_error(args, f"known init failed; SDK grip_init fallback OK pos={pos}")
    return grip


def learn_calibration_from_grip(args: argparse.Namespace, grip: LingkongGrip) -> None:
    clamp_pos = getattr(grip, "_clamp_pos", None)
    open_pos = getattr(grip, "_open_pos", None)
    max_itinerary = getattr(grip, "_max_itinerary", None)
    speed_coe = getattr(grip, "_speed_coe", None)
    if None in {clamp_pos, open_pos, max_itinerary, speed_coe}:
        print("fallback 标定成功，但未能读取完整标定参数", flush=True)
        return
    args.clamp_pos = int(clamp_pos)
    args.open_pos = int(open_pos)
    args.max_itinerary = int(max_itinerary)
    args.speed_coe = int(speed_coe)
    print(
        "已共享/更新夹爪标定参数: "
        f"clamp_pos={args.clamp_pos} open_pos={args.open_pos} "
        f"max_itinerary={args.max_itinerary} speed_coe={args.speed_coe}",
        flush=True,
    )
    save_cached_calibration(args)



def init_known_gripper(args: argparse.Namespace, *, command: str) -> LingkongGrip | None:
    set_init_error(args, "")
    disable_proxy_for_host(args.server)
    grip = LingkongGrip(
        server_address=args.server,
        connect_attempts=args.connect_attempts,
        connect_timeout_sec=args.connect_timeout_sec,
        connect_retry_delay_sec=args.connect_retry_delay_sec,
    )
    print("使用已知标定参数初始化夹爪，不执行找零动作", flush=True)
    ok = grip.grip_init_with_known_limits(
        clamp_pos=args.clamp_pos,
        open_pos=args.open_pos,
        max_itinerary=args.max_itinerary,
        speed_coe=args.speed_coe,
    )
    if not ok:
        set_init_error(args, "known calibration init failed")
        if getattr(args, CALIBRATION_SOURCE_ATTR, "") == "cache":
            invalidate_cached_calibration(args, "known calibration init failed")
        if command == "check":
            grip.close(reset_torque=False)
            return None
        fallback = init_homing_fallback(args, grip, command=command)
        if fallback is not None:
            return fallback
        grip.close(reset_torque=True)
        return None

    pos = grip.read_pos()
    low = -args.calibration_tolerance
    high = 1000 + args.calibration_tolerance
    if pos == -1 or pos < low or pos > high:
        set_init_error(args, f"SDK position validation failed: pos={pos}, expected={low}..{high}")
        if getattr(args, CALIBRATION_SOURCE_ATTR, "") == "cache":
            invalidate_cached_calibration(args, f"position {pos} outside {low}..{high}")
        if command == "check":
            grip.close(reset_torque=False)
            return None
        fallback = init_homing_fallback(args, grip, command=command)
        if fallback is not None:
            return fallback
        grip.close(reset_torque=True)
        return None

    print(f"当前 SDK 位置验证通过: pos={pos}", flush=True)
    return grip


def wait_for_fresh_status(grip: LingkongGrip, timeout: float = 2.0) -> bool:
    start = time.monotonic()
    last_current = grip.read_cur_current()
    while time.monotonic() - start < timeout:
        time.sleep(0.05)
        current = grip.read_cur_current()
        pos = grip.read_pos()
        if current != last_current or pos != -1:
            return True
    return False


def measured_pos_far_from_target(pos: int, target: int, args: argparse.Namespace) -> bool:
    if pos == -1:
        return False
    return abs(int(pos) - int(target)) > int(args.target_pos_tolerance)


def run_grip(args: argparse.Namespace) -> int:
    grip = init_known_gripper(args, command="grip")
    if grip is None:
        return 1

    keep_hold_torque = False
    try:
        grip.set_speed(args.grip_speed)
        grip.set_torque_limit(args.grip_torque)
        wait_for_fresh_status(grip)

        print(
            "开始夹取: "
            f"target={args.min_pos}, range={args.min_pos}-{args.max_pos}, "
            f"empty_limit={args.min_pos + args.empty_grip_margin}, "
            f"threshold={args.current_threshold}, torque={args.grip_torque}, "
            f"hold_torque={args.hold_torque}",
            flush=True,
        )
        contact_pos, stop_reason, failed_min_limit = run_continuous_grasp(grip, args)
        if failed_min_limit:
            print(f"{stop_reason}，不进入保持夹持状态", flush=True)
            return 2

        finish_grasp(grip, args, contact_pos, stop_reason)
        keep_hold_torque = True
        time.sleep(args.grip_done_wait)
        return 0
    finally:
        grip.close(reset_torque=not keep_hold_torque)
        print("夹爪连接已关闭", flush=True)


def run_continuous_grasp(
    grip: LingkongGrip, args: argparse.Namespace
) -> tuple[int | None, str, bool]:
    start_time = time.monotonic()
    stall_count = 0
    contact_pos = None
    last_progress_pos = grip.read_pos()

    print(f"连续闭合到限位目标: {args.min_pos}", flush=True)
    grip.move_to_pos(args.min_pos)

    while True:
        now = time.monotonic()
        if now - start_time > args.timeout:
            return grip.read_pos(), "夹取超时，停止继续闭合", False

        time.sleep(args.poll_interval)
        current = grip.read_cur_current()
        pos = grip.read_pos()
        temp = grip.read_cur_tempture()
        moved = None if pos == -1 or last_progress_pos == -1 else last_progress_pos - pos
        print(
            f"target={args.min_pos}, pos={pos}, moved={moved}, "
            f"current={current}, temp={temp}",
            flush=True,
        )

        empty_limit = args.min_pos + args.empty_grip_margin
        if now - start_time < args.contact_grace or pos == -1:
            continue

        # Position has priority over current near the configured empty limit.
        # A high current at/below empty_limit is usually the fingers reaching
        # their end stop, not object contact.
        if pos <= empty_limit and current >= args.current_threshold:
            return (
                pos,
                "到达空夹限位且电流达到阈值，判定空夹失败 "
                f"pos={pos}, empty_limit={empty_limit}, current={current}",
                True,
            )

        if current >= args.current_threshold:
            return pos, f"电流达到阈值 current={current}", False
        if last_progress_pos == -1:
            last_progress_pos = pos
            continue

        moved = last_progress_pos - pos
        if moved >= args.progress_epsilon:
            last_progress_pos = pos
            stall_count = 0
        else:
            stall_count += 1
        if stall_count >= args.stall_samples:
            contact_pos = pos
            if contact_pos != -1 and contact_pos <= empty_limit:
                return (
                    contact_pos,
                    "位置停止但已接近最小限位，判定空夹失败 "
                    f"pos={contact_pos}, empty_limit={empty_limit}, "
                    f"moved={moved}, samples={stall_count}",
                    True,
                )
            return contact_pos, f"位置停止变化 moved={moved}, samples={stall_count}", False


def finish_grasp(
    grip: LingkongGrip, args: argparse.Namespace, contact_pos: int | None, stop_reason: str
) -> int:
    if contact_pos == -1 or contact_pos is None:
        contact_pos = grip.read_pos()
    if contact_pos == -1 or contact_pos is None:
        contact_pos = args.min_pos
    contact_pos = int(clamp(contact_pos, args.min_pos, args.max_pos))
    print(
        f"{stop_reason}，接触位置 {contact_pos}；"
        "切换为小力矩持续闭合保持，不锁死在接触位置",
        flush=True,
    )
    grip.set_torque_limit(args.hold_torque)
    time.sleep(0.05)
    grip.move_to_pos(args.min_pos)
    print(
        "已切换到 compliant hold: "
        f"hold_torque={args.hold_torque}, hold_target={args.min_pos}, "
        f"baseline_contact_pos={contact_pos}",
        flush=True,
    )
    return contact_pos


def run_release(args: argparse.Namespace) -> int:
    grip = init_known_gripper(args, command="release")
    if grip is None:
        return 1

    keep_torque = False
    try:
        grip.set_speed(args.release_speed)
        grip.set_torque_limit(args.release_torque)
        print(
            f"释放夹爪: target={args.release_target}, "
            f"speed={args.release_speed}, torque={args.release_torque}",
            flush=True,
        )
        grip.move_to_pos(args.release_target)
        time.sleep(args.release_wait)
        pos = grip.read_pos()
        if measured_pos_far_from_target(pos, args.release_target, args):
            print(
                "WARNING 夹爪位置标定可能偏差较大: "
                f"release target={args.release_target}, measured pos={pos}, "
                f"tolerance={args.target_pos_tolerance}. "
                "继续使用已知标定参数，不执行找零动作。",
                flush=True,
            )
        print(
            f"完成: pos={pos}, current={grip.read_cur_current()}, "
            f"temp={grip.read_cur_tempture()}",
            flush=True,
        )
        keep_torque = True
        return 0
    finally:
        grip.close(reset_torque=not keep_torque)
        print("夹爪连接已关闭", flush=True)


def run_check(args: argparse.Namespace) -> int:
    grip = init_known_gripper(args, command="check")
    if grip is None:
        return 1
    try:
        print(
            f"CHECK OK: pos={grip.read_pos()}, current={grip.read_cur_current()}, "
            f"temp={grip.read_cur_tempture()}",
            flush=True,
        )
        return 0
    finally:
        grip.close(reset_torque=False)
        print("夹爪连接已关闭", flush=True)


def run_feedback(args: argparse.Namespace) -> tuple[int, str]:
    grip = init_known_gripper(args, command="check")
    if grip is None:
        detail = get_init_error(args)
        return 1, f"feedback failed{f' detail={detail}' if detail else ''}"
    try:
        pos = grip.read_pos()
        current = grip.read_cur_current()
        temp = grip.read_cur_tempture()
        message = f"feedback source=direct pos={pos} current={current} temp={temp}"
        print(message, flush=True)
        return 0, message
    finally:
        grip.close(reset_torque=False)
        print("夹爪连接已关闭", flush=True)


class GripState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
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

    def get_feedback_grip(self) -> tuple[LingkongGrip | None, str]:
        if self.feedback_grip is not None:
            return self.feedback_grip, "cached"
        grip = init_known_gripper(self.args, command="check")
        if grip is None:
            return None, "failed"
        self.feedback_grip = grip
        return grip, "new"


def run_command(state: GripState, command: str) -> tuple[bool, str]:
    with state.lock:
        if state.running:
            return False, "BUSY command already running"
        state.running = True
        state.last_result = f"{command} running"

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{started}] receive {command} signal", flush=True)
    try:
        if command == "grip":
            state.close_feedback_grip()
            code = run_grip(state.args)
        elif command == "release":
            state.close_feedback_grip()
            code = run_release(state.args)
        elif command == "check":
            state.close_feedback_grip()
            code = run_check(state.args)
        elif command == "feedback":
            grip, source = state.get_feedback_grip()
            if grip is None:
                detail = get_init_error(state.args)
                code = 1
                detail_message = (
                    f"feedback source={source} failed"
                    f"{f' detail={detail}' if detail else ''}"
                )
            else:
                pos = grip.read_pos()
                current = grip.read_cur_current()
                temp = grip.read_cur_tempture()
                detail_message = (
                    f"feedback source={source} pos={pos} "
                    f"current={current} temp={temp}"
                )
                print(detail_message, flush=True)
                code = 0
        else:
            return False, "ERR command must be 'grip', 'release', 'check', 'feedback', or 'status'"

        if command == "grip" and code == 2:
            message = "GRASP_FAILED_MIN_LIMIT grip done exit_code=2"
            ok = True
        elif command == "feedback" and code == 0:
            message = detail_message
            ok = True
        elif code == 0:
            message = f"{command} done exit_code={code}"
            ok = True
        else:
            message = f"{command} failed exit_code={code}"
            detail = get_init_error(state.args)
            if detail:
                message += f" detail={detail}"
            ok = False
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)
        with state.lock:
            state.last_result = message
        return ok, f"{'OK' if ok else 'ERR'} {message}"
    except Exception as exc:
        if command == "feedback":
            state.close_feedback_grip()
        message = f"ERROR {exc}"
        print(message, flush=True)
        with state.lock:
            state.last_result = message
        return False, f"ERR {message}"
    finally:
        with state.lock:
            state.running = False


class GripRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(1024)
        if not raw:
            return

        message = raw.decode("utf-8", errors="replace").strip()
        state = self.server.state
        print(f"signal from {self.client_address}: {message!r}", flush=True)

        parts = message.split()
        if state.token:
            if len(parts) != 2 or parts[0] != state.token:
                self.wfile.write(b"ERR bad token or command\n")
                return
            command = parts[1].lower()
        else:
            command = parts[0].lower() if parts else ""

        if command == "status":
            with state.lock:
                running = state.running
                last_result = state.last_result
            self.wfile.write(
                f"OK running={running} {gripper_config_summary(state.args)} "
                f"last={last_result}\n".encode()
            )
            return

        _, reply = run_command(state, command)
        self.wfile.write((reply + "\n").encode())


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(args: argparse.Namespace) -> int:
    state = GripState(args)
    with ThreadedTCPServer((args.host, args.port), GripRequestHandler) as server:
        server.state = state
        print(f"grip signal receiver listening on {args.host}:{args.port}", flush=True)
        print(f"[gripper] config {gripper_config_summary(args)}", flush=True)
        print(
            "send 'grip' to grasp, 'release' to open, "
            "'feedback' to sample, 'status' to query",
            flush=True,
        )
        if args.token:
            print("token enabled: send '<token> grip' or '<token> release'", flush=True)
        server.serve_forever()
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "serve":
        return serve(args)
    if args.command == "grip":
        return run_grip(args)
    if args.command == "release":
        return run_release(args)
    if args.command == "check":
        return run_check(args)
    if args.command == "feedback":
        code, message = run_feedback(args)
        print(("OK " if code == 0 else "ERR ") + message)
        return code
    print("OK direct status: not serving")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
