from argparse import Namespace

import pytest

from daimon_stuff import grip_signal_receiver


class FakeGrip:
    def __init__(self, *, positions, currents=None) -> None:
        self.positions = list(positions)
        self.currents = list(currents or [0] * len(self.positions))
        self.position_index = 0
        self.current_index = 0
        self.commands = []
        self.speed = None
        self.torque = None
        self.closed = False

    def move_to_pos(self, pos: int) -> None:
        self.commands.append(("move_to_pos", pos))

    def read_pos(self) -> int:
        index = min(self.position_index, len(self.positions) - 1)
        value = self.positions[index]
        self.position_index += 1
        return value

    def read_cur_current(self) -> int:
        index = min(self.current_index, len(self.currents) - 1)
        value = self.currents[index]
        self.current_index += 1
        return value

    def read_cur_tempture(self) -> int:
        return 25

    def set_speed(self, speed: int) -> None:
        self.speed = speed

    def set_torque_limit(self, torque: int) -> None:
        self.torque = torque

    def close(self, reset_torque=True) -> None:
        self.closed = True


def grip_args(**overrides) -> Namespace:
    values = {
        "min_pos": 100,
        "max_pos": 1000,
        "current_threshold": 120,
        "poll_interval": 0.02,
        "contact_grace": 0.0,
        "progress_epsilon": 2,
        "stall_samples": 3,
        "empty_grip_margin": 150,
        "target_pos_tolerance": 120,
        "timeout": 1.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_stall_near_min_limit_is_empty_grip_failure() -> None:
    grip = FakeGrip(positions=[280, 220, 200, 200, 200, 200])

    contact_pos, reason, failed = grip_signal_receiver.run_continuous_grasp(
        grip,
        grip_args(),
    )

    assert contact_pos == 200
    assert failed is True
    assert "空夹失败" in reason


def test_high_current_near_min_limit_is_empty_grip_failure() -> None:
    grip = FakeGrip(
        positions=[280, 219],
        currents=[0, 200],
    )

    contact_pos, reason, failed = grip_signal_receiver.run_continuous_grasp(
        grip,
        grip_args(),
    )

    assert contact_pos == 219
    assert failed is True
    assert "空夹失败" in reason


def test_stall_well_above_min_limit_is_contact_success() -> None:
    grip = FakeGrip(positions=[400, 320, 270, 270, 270, 270])

    contact_pos, reason, failed = grip_signal_receiver.run_continuous_grasp(
        grip,
        grip_args(),
    )

    assert contact_pos == 270
    assert failed is False
    assert "位置停止变化" in reason


def test_release_warns_when_known_calibration_position_is_far(monkeypatch, capsys) -> None:
    grip = FakeGrip(positions=[219])
    monkeypatch.setattr(
        grip_signal_receiver,
        "init_known_gripper",
        lambda args, command: grip,
    )

    result = grip_signal_receiver.run_release(
        Namespace(
            release_target=1000,
            release_speed=60,
            release_torque=20,
            release_wait=0.0,
            target_pos_tolerance=120,
        )
    )

    assert result == 0
    assert grip.commands == [("move_to_pos", 1000)]
    assert grip.closed
    output = capsys.readouterr().out
    assert "WARNING 夹爪位置标定可能偏差较大" in output
    assert "不执行找零动作" in output


def test_cached_gripper_starts_grip_without_reinitializing_or_closing(monkeypatch) -> None:
    grip = FakeGrip(positions=[600, 500, 400], currents=[0, 200])
    args = grip_args(
        grip_speed=80,
        grip_torque=40,
        hold_torque=20,
        grip_done_wait=0.0,
    )
    monkeypatch.setattr(
        grip_signal_receiver,
        "init_known_gripper",
        lambda *_args, **_kwargs: pytest.fail("cached grip must not reinitialize"),
    )
    monkeypatch.setattr(
        grip_signal_receiver,
        "finish_grasp",
        lambda *_args, **_kwargs: None,
    )

    result = grip_signal_receiver.run_grip(args, grip=grip)

    assert result == 0
    assert grip.commands[0] == ("move_to_pos", args.min_pos)
    assert grip.closed is False


class FaultGrip:
    def __init__(self, error_status: int, clear_result: bool) -> None:
        self.error_status = error_status
        self.clear_result = clear_result
        self.clear_calls = []

    def read_error_status(self) -> int:
        return self.error_status

    def clear_motor_error(self, timeout: float) -> bool:
        self.clear_calls.append(timeout)
        if self.clear_result:
            self.error_status = 0
        return self.clear_result


def test_motor_fault_is_cleared_before_motion(capsys) -> None:
    grip = FaultGrip(error_status=1, clear_result=True)
    args = Namespace()

    assert grip_signal_receiver.recover_motor_fault(args, grip) is True
    assert grip.clear_calls == [1.0]
    assert "电机故障已清除" in capsys.readouterr().out


def test_persistent_motor_fault_blocks_homing(capsys) -> None:
    grip = FaultGrip(error_status=1, clear_result=False)
    args = Namespace()

    assert grip_signal_receiver.recover_motor_fault(args, grip) is False
    assert "欠压/供电电压过低" in grip_signal_receiver.get_init_error(args)
    assert "禁止继续找零" in capsys.readouterr().out
