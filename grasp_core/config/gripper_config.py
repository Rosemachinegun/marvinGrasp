"""独立的夹爪通信、运动和闭合识别配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from grasp_core.config.resource_paths import PROJECT_ROOT


@dataclass(frozen=True)
class GripSignalDefaults:
    host: str = "127.0.0.1"
    port: int = 55660
    timeout_sec: float = 0.5
    command_timeout_sec: float = 5.0
    receiver_path: Path = PROJECT_ROOT / "daimon_gripper" / "grip_signal_receiver.py"
    settle_sec: float = 0.0
    post_confirm_hold_sec: float = 0.0
    lift_hold_sec: float = 0.0
    retry_max_attempts: int = 1
    drop_close_delta: int = 80
    drop_poll_interval: float = 0.05


@dataclass(frozen=True)
class GripperDefaults:
    """左右夹爪的硬件、闭合和识别默认值。"""

    left_server: str = "192.168.14.11:55551"
    right_server: str = "192.168.14.10:55551"
    dual: bool = True
    left_clamp_pos: int = -52525
    right_clamp_pos: int = -52525
    left_open_pos: int = -142525
    right_open_pos: int = -142525
    left_max_itinerary: int = 90000
    right_max_itinerary: int = 90000
    left_speed_coe: int = 3600
    right_speed_coe: int = 3600
    calibration_tolerance: int = 150
    connect_attempts: int = 3
    connect_timeout_sec: float = 2.0
    connect_retry_delay_sec: float = 0.2
    allow_homing_fallback: bool = False
    left_min_pos: int = 100
    right_min_pos: int = 100
    left_max_pos: int = 1000
    right_max_pos: int = 1000
    left_grip_speed: int = 50
    right_grip_speed: int = 50
    left_grip_torque: int = 100
    right_grip_torque: int = 100
    left_hold_torque: int = 100
    right_hold_torque: int = 100
    # The device feedback in this setup is typically around 20-30 at contact.
    left_current_threshold: int = 20
    right_current_threshold: int = 20
    left_poll_interval: float = 0.01
    right_poll_interval: float = 0.01
    left_contact_grace: float = 0.2
    right_contact_grace: float = 0.2
    left_progress_epsilon: int = 2
    right_progress_epsilon: int = 2
    left_stall_samples: int = 5
    right_stall_samples: int = 5
    # Keep the empty-limit band narrow; an object may legitimately stop close
    # to the mechanical minimum while still producing contact current.
    left_empty_grip_margin: int = 2
    right_empty_grip_margin: int = 2
    left_target_pos_tolerance: int = 120
    right_target_pos_tolerance: int = 120
    left_timeout: float = 5.0
    right_timeout: float = 5.0
    left_grip_done_wait: float = 0.0
    right_grip_done_wait: float = 0.0
    left_release_target: int = 1000
    right_release_target: int = 1000
    left_release_speed: int = 60
    right_release_speed: int = 60
    left_release_torque: int = 20
    right_release_torque: int = 20
    left_release_wait: float = 0.05
    right_release_wait: float = 0.05

    @property
    def server(self) -> str: return self.right_server
    @property
    def clamp_pos(self) -> int: return self.right_clamp_pos
    @property
    def open_pos(self) -> int: return self.right_open_pos
    @property
    def max_itinerary(self) -> int: return self.right_max_itinerary
    @property
    def speed_coe(self) -> int: return self.right_speed_coe
    @property
    def min_pos(self) -> int: return self.right_min_pos
    @property
    def max_pos(self) -> int: return self.right_max_pos
    @property
    def grip_speed(self) -> int: return self.right_grip_speed
    @property
    def grip_torque(self) -> int: return self.right_grip_torque
    @property
    def hold_torque(self) -> int: return self.right_hold_torque
    @property
    def current_threshold(self) -> int: return self.right_current_threshold
    @property
    def poll_interval(self) -> float: return self.right_poll_interval
    @property
    def contact_grace(self) -> float: return self.right_contact_grace
    @property
    def progress_epsilon(self) -> int: return self.right_progress_epsilon
    @property
    def stall_samples(self) -> int: return self.right_stall_samples
    @property
    def empty_grip_margin(self) -> int: return self.right_empty_grip_margin
    @property
    def target_pos_tolerance(self) -> int: return self.right_target_pos_tolerance
    @property
    def timeout(self) -> float: return self.right_timeout
    @property
    def grip_done_wait(self) -> float: return self.right_grip_done_wait
    @property
    def release_target(self) -> int: return self.right_release_target
    @property
    def release_speed(self) -> int: return self.right_release_speed
    @property
    def release_torque(self) -> int: return self.right_release_torque
    @property
    def release_wait(self) -> float: return self.right_release_wait


GRIP_SIGNAL_DEFAULTS = GripSignalDefaults()
GRIPPER_DEFAULTS = GripperDefaults()
