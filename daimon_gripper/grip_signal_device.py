from __future__ import annotations

import argparse
import sys
import time

try:
    from .grip_signal_calibration import (
        invalidate_cached_calibration,
        save_cached_calibration,
    )
    from .grip_signal_config import (
        CALIBRATION_INVALIDATED_ATTR,
        CALIBRATION_SOURCE_ATTR,
        INIT_ERROR_ATTR,
        PROJECT_ROOT,
    )
    from .grip_signal_utils import disable_proxy_for_host, set_motion_profile
except ImportError:
    from grip_signal_calibration import (
        invalidate_cached_calibration,
        save_cached_calibration,
    )
    from grip_signal_config import (
        CALIBRATION_INVALIDATED_ATTR,
        CALIBRATION_SOURCE_ATTR,
        INIT_ERROR_ATTR,
        PROJECT_ROOT,
    )
    from grip_signal_utils import disable_proxy_for_host, set_motion_profile


GRIPPER_SDK_ROOT = PROJECT_ROOT / "daimon_gripper" / "dm_gripper_py"
if str(GRIPPER_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(GRIPPER_SDK_ROOT))

from dm_lingkong_grip_sdk import LingkongGrip


MOTOR_ERROR_DESCRIPTIONS = {
    1: "欠压/供电电压过低",
    8: "电机过温",
}


def set_init_error(args: argparse.Namespace, message: str) -> None:
    setattr(args, INIT_ERROR_ATTR, message)
    if message:
        print(message, flush=True)


def get_init_error(args: argparse.Namespace) -> str:
    return str(getattr(args, INIT_ERROR_ATTR, "") or "").strip()


def describe_motor_error(error_status: int) -> str:
    descriptions = [
        description
        for bit, description in MOTOR_ERROR_DESCRIPTIONS.items()
        if error_status & bit
    ]
    return ", ".join(descriptions) if descriptions else "未知电机故障"


def connect_gripper(args: argparse.Namespace) -> LingkongGrip:
    disable_proxy_for_host(args.server)
    return LingkongGrip(
        server_address=args.server,
        connect_attempts=args.connect_attempts,
        connect_timeout_sec=args.connect_timeout_sec,
        connect_retry_delay_sec=args.connect_retry_delay_sec,
    )


def open_gripper_after_calibration(args: argparse.Namespace, grip: LingkongGrip) -> None:
    print(
        f"夹爪校准完成: pos={grip.read_pos()}；"
        f"打开到最大位置 target={args.release_target}",
        flush=True,
    )
    set_motion_profile(grip, speed=args.release_speed, torque=args.release_torque)
    grip.move_to_pos(args.release_target)
    time.sleep(args.release_wait)
    print(f"校准后打开完成: pos={grip.read_pos()}", flush=True)


def recover_motor_fault(args: argparse.Namespace, grip: LingkongGrip) -> bool:
    read_error = getattr(grip, "read_error_status", None)
    clear_error = getattr(grip, "clear_motor_error", None)
    if not callable(read_error) or not callable(clear_error):
        return True

    error_status = int(read_error())
    if error_status == 0:
        return True

    detail = describe_motor_error(error_status)
    print(
        f"检测到电机故障: error_status={error_status} ({detail})；尝试清除锁存故障",
        flush=True,
    )
    if clear_error(timeout=1.0):
        print("电机故障已清除，重新验证夹爪状态", flush=True)
        return True

    error_status = int(read_error())
    detail = describe_motor_error(error_status)
    set_init_error(
        args,
        f"电机故障无法清除: error_status={error_status} ({detail})；"
        "请检查夹爪供电、急停、驱动器和机械卡阻，禁止继续找零",
    )
    return False


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
        set_motion_profile(grip, speed=args.release_speed, torque=args.release_torque)
        print(
            "fallback 初始化完成；先打开夹爪再执行力控闭合 "
            f"target={args.release_target}",
            flush=True,
        )
        grip.move_to_pos(args.release_target)
        time.sleep(args.release_wait)
    set_init_error(args, f"known init failed; SDK grip_init fallback OK pos={pos}")
    return grip


def handle_known_init_failure(
    args: argparse.Namespace,
    grip: LingkongGrip,
    *,
    command: str,
    message: str,
    cache_reason: str,
) -> LingkongGrip | None:
    set_init_error(args, message)
    if getattr(args, CALIBRATION_SOURCE_ATTR, "") == "cache":
        invalidate_cached_calibration(args, cache_reason)
    if command == "check" and not bool(args.allow_homing_fallback):
        grip.close(reset_torque=False)
        return None

    fallback = init_homing_fallback(args, grip, command=command)
    if fallback is not None:
        return fallback

    grip.close(reset_torque=True)
    return None


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
    grip = connect_gripper(args)
    print("使用已知标定参数初始化夹爪，不执行找零动作", flush=True)
    ok = grip.grip_init_with_known_limits(
        clamp_pos=args.clamp_pos,
        open_pos=args.open_pos,
        max_itinerary=args.max_itinerary,
        speed_coe=args.speed_coe,
    )
    if not ok:
        return handle_known_init_failure(
            args,
            grip,
            command=command,
            message="known calibration init failed",
            cache_reason="known calibration init failed",
        )

    pos = grip.read_pos()
    low = -args.calibration_tolerance
    high = 1000 + args.calibration_tolerance
    if pos == -1 or pos < low or pos > high:
        raw_pos = getattr(grip, "_cur_pos", None)
        if command == "release" and pos != -1 and not bool(args.allow_homing_fallback):
            print(
                "WARNING SDK 位置超出已知标定范围，但 release 将先尝试打开夹爪: "
                f"pos={pos}, raw_pos={raw_pos}, expected={low}..{high}. "
                "如未成功打开，请加 --allow-homing-fallback 重新标定。",
                flush=True,
            )
            return grip
        if pos == -1 and not recover_motor_fault(args, grip):
            grip.close(reset_torque=False)
            return None
        if pos == -1:
            time.sleep(0.1)
            pos = grip.read_pos()
            if low <= pos <= high:
                print(f"故障恢复后 SDK 位置验证通过: pos={pos}", flush=True)
                return grip
        validation_error = (
            f"SDK position validation failed: pos={pos}, expected={low}..{high}"
        )
        return handle_known_init_failure(
            args,
            grip,
            command=command,
            message=validation_error,
            cache_reason=f"position {pos} outside {low}..{high}",
        )

    print(f"当前 SDK 位置验证通过: pos={pos}", flush=True)
    return grip


def init_calibrated_gripper(args: argparse.Namespace) -> LingkongGrip | None:
    set_init_error(args, "")
    grip = connect_gripper(args)
    set_motion_profile(grip, speed=args.calibration_speed, torque=args.calibration_torque)
    print(
        "校准夹爪: 调用 SDK grip_init() 执行找零动作 "
        f"speed={args.calibration_speed}, torque={args.calibration_torque}",
        flush=True,
    )
    if not grip.grip_init():
        set_init_error(args, "SDK grip_init failed")
        grip.close(reset_torque=True)
        return None
    learn_calibration_from_grip(args, grip)
    open_gripper_after_calibration(args, grip)
    return grip


def init_gripper(args: argparse.Namespace, *, command: str) -> LingkongGrip | None:
    if bool(getattr(args, "calibrate_each_command", False)):
        return init_calibrated_gripper(args)
    return init_known_gripper(args, command=command)
