from __future__ import annotations

import argparse
import time

try:
    from .grip_signal_device import LingkongGrip, get_init_error, init_gripper
    from .grip_signal_utils import (
        clamp,
        read_check_sample,
        read_feedback_sample,
        set_motion_profile,
    )
except ImportError:
    from grip_signal_device import LingkongGrip, get_init_error, init_gripper
    from grip_signal_utils import (
        clamp,
        read_check_sample,
        read_feedback_sample,
        set_motion_profile,
    )


def wait_for_fresh_status(grip: LingkongGrip, timeout: float = 2.0) -> bool:
    # A pre-warmed receiver already has a valid asynchronous status sample; do
    # not add a fixed sleep on the critical target-arrival -> close path.
    if grip.read_pos() != -1:
        return True
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


class GripperRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def grip(self, grip: LingkongGrip | None = None) -> int:
        owns_grip = grip is None
        grip = grip or init_gripper(self.args, command="grip")
        if grip is None:
            return 1

        keep_hold_torque = False
        try:
            set_motion_profile(
                grip, speed=self.args.grip_speed, torque=self.args.grip_torque
            )
            wait_for_fresh_status(grip)

            print(
                "开始夹取: "
                f"target={self.args.min_pos}, range={self.args.min_pos}-{self.args.max_pos}, "
                f"empty_limit={self.args.min_pos + self.args.empty_grip_margin}, "
                f"threshold={self.args.current_threshold}, torque={self.args.grip_torque}, "
                f"hold_torque={self.args.hold_torque}",
                flush=True,
            )
            contact_pos, stop_reason, failed_min_limit = self.continuous_grasp(grip)
            if failed_min_limit:
                print(f"{stop_reason}，不进入保持夹持状态", flush=True)
                return 2

            finish_grasp(grip, self.args, contact_pos, stop_reason)
            keep_hold_torque = True
            time.sleep(self.args.grip_done_wait)
            return 0
        finally:
            if owns_grip:
                grip.close(reset_torque=not keep_hold_torque)
                print("夹爪连接已关闭", flush=True)

    def continuous_grasp(self, grip: LingkongGrip) -> tuple[int | None, str, bool]:
        start_time = time.monotonic()
        stall_count = 0
        last_progress_pos = grip.read_pos()
        contact_confirm_samples = max(
            int(getattr(self.args, "contact_confirm_samples", 1)), 1
        )
        empty_limit_confirm_samples = max(
            int(getattr(self.args, "empty_limit_confirm_samples", 5)), 1
        )
        contact_count = 0
        empty_limit_count = 0

        print(f"连续闭合到限位目标: {self.args.min_pos}", flush=True)
        grip.move_to_pos(self.args.min_pos)

        while True:
            now = time.monotonic()
            if now - start_time > self.args.timeout:
                return grip.read_pos(), "夹取超时，未获得连续电流确认", True

            time.sleep(self.args.poll_interval)
            current = grip.read_cur_current()
            pos = grip.read_pos()
            temp = grip.read_cur_tempture()
            moved = (
                None
                if pos == -1 or last_progress_pos == -1
                else last_progress_pos - pos
            )
            print(
                f"target={self.args.min_pos}, pos={pos}, moved={moved}, "
                f"current={current}, temp={temp}",
                flush=True,
            )

            empty_limit = self.args.min_pos + self.args.empty_grip_margin
            if now - start_time < self.args.contact_grace or pos == -1:
                continue

            # A confirmed contact current takes priority over the position band.
            # Objects can legitimately be grasped very close to min_pos.
            if current >= self.args.current_threshold:
                contact_count += 1
            else:
                contact_count = 0
            if contact_count >= contact_confirm_samples:
                return (
                    pos,
                    "电流连续达到阈值，判定夹到物体 "
                    f"current={current}, "
                    f"samples={contact_count}",
                    False,
                )

            # Without a sustained contact current, repeated samples in the
            # mechanical end-stop band are treated as an empty grasp.
            if pos <= empty_limit and current < self.args.current_threshold:
                empty_limit_count += 1
            else:
                empty_limit_count = 0
            if empty_limit_count >= empty_limit_confirm_samples:
                return (
                    pos,
                    "到达空夹限位但未获得连续接触电流，判定空夹失败 "
                    f"pos={pos}, empty_limit={empty_limit}, "
                    f"current={current}, samples={empty_limit_count}",
                    True,
                )
            if last_progress_pos == -1:
                last_progress_pos = pos
                continue

            moved = last_progress_pos - pos
            if moved >= self.args.progress_epsilon:
                last_progress_pos = pos
                stall_count = 0
            else:
                stall_count += 1
            if stall_count >= self.args.stall_samples:
                if pos != -1 and pos <= empty_limit:
                    return (
                        pos,
                        "位置停止但已接近最小限位，判定空夹失败 "
                        f"pos={pos}, empty_limit={empty_limit}, "
                        f"moved={moved}, samples={stall_count}",
                        True,
                    )
                return pos, f"位置停止变化 moved={moved}, samples={stall_count}", False

    def release(self, grip: LingkongGrip | None = None) -> int:
        owns_grip = grip is None
        grip = grip or init_gripper(self.args, command="release")
        if grip is None:
            return 1

        keep_torque = False
        try:
            set_motion_profile(
                grip, speed=self.args.release_speed, torque=self.args.release_torque
            )
            print(
                f"释放夹爪: target={self.args.release_target}, "
                f"speed={self.args.release_speed}, torque={self.args.release_torque}",
                flush=True,
            )
            grip.move_to_pos(self.args.release_target)
            time.sleep(self.args.release_wait)
            pos = grip.read_pos()
            if measured_pos_far_from_target(pos, self.args.release_target, self.args):
                print(
                    "WARNING 夹爪位置标定可能偏差较大: "
                    f"release target={self.args.release_target}, measured pos={pos}, "
                    f"tolerance={self.args.target_pos_tolerance}. "
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
            if owns_grip:
                grip.close(reset_torque=not keep_torque)
                print("夹爪连接已关闭", flush=True)

    def check(self) -> int:
        code, _message = self.status_sample(read_check_sample)
        return code

    def feedback(self) -> tuple[int, str]:
        return self.status_sample(read_feedback_sample)

    def status_sample(self, formatter) -> tuple[int, str]:
        grip = init_gripper(self.args, command="check")
        if grip is None:
            detail = get_init_error(self.args)
            message = f"feedback failed{f' detail={detail}' if detail else ''}"
            return 1, message
        try:
            message = formatter(grip)
            print(message, flush=True)
            return 0, message
        finally:
            grip.close(reset_torque=False)
            print("夹爪连接已关闭", flush=True)

    def direct_command(self) -> int:
        if self.args.command == "feedback":
            code, message = self.feedback()
            print(("OK " if code == 0 else "ERR ") + message)
            return code
        runner = {
            "grip": self.grip,
            "release": self.release,
            "check": self.check,
        }.get(self.args.command)
        if runner is not None:
            return runner()
        print("OK direct status: not serving")
        return 0


def run_grip(args: argparse.Namespace, grip: LingkongGrip | None = None) -> int:
    return GripperRunner(args).grip(grip)


def run_continuous_grasp(
    grip: LingkongGrip, args: argparse.Namespace
) -> tuple[int | None, str, bool]:
    return GripperRunner(args).continuous_grasp(grip)
