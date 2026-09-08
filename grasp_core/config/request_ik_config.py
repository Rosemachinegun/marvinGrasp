#!/usr/bin/env python3
"""配置层：集中管理命令行参数、默认值和 tool.yaml 默认抓取配置。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from dataclasses import dataclass

import numpy as np
import yaml

from grasp_core.perception.flowpose_pipeline import (
    DEFAULT_BBOX_CONTAINMENT_THRESHOLD,
    DEFAULT_CAPTURE_DIR,
    DEFAULT_CONTAINMENT_MIN_AREA_RATIO,
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_DEDUP_IOU_THRESHOLD,
    DEFAULT_DINO_CKPT_CANDIDATES,
    DEFAULT_DINO_REPO_CANDIDATES,
    DEFAULT_FLOW_MODEL_PATH,
    DEFAULT_SCALE_MODEL_PATH,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_SERIAL,
)


def _format_xyz(values: np.ndarray) -> str:
    return "(" + ", ".join(f"{float(value):.4f}" for value in values[:3]) + ")"


def clamp(value: int | float, low: int | float, high: int | float):
    return max(low, min(high, value))

DEFAULT_ROBOT_XACRO_PATH = PROJECT_ROOT / "config" / "stand_v3.urf.xacro"


DEFAULT_TOOL_TEMPLATE_PATH = PROJECT_ROOT / "config" / "tool.yaml"
DEFAULT_SAM3_ROI_XYXY = (161, 9, 504, 343)

@dataclass(frozen=True)
class GripSignalDefaults:
    host: str = "127.0.0.1"
    port: int = 55660
    timeout_sec: float = 0.5
    command_timeout_sec: float = 5.0
    receiver_path: Path = PROJECT_ROOT / "daimon_stuff" / "grip_signal_receiver.py"
    settle_sec: float = 0.0
    post_confirm_hold_sec: float = 0.0
    lift_hold_sec: float = 0.0
    retry_max_attempts: int = 1
    drop_close_delta: int = 30
    drop_poll_interval: float = 0.05


@dataclass(frozen=True)
class GripperDefaults:
    left_server: str = "192.168.14.11:55551"
    right_server: str = "192.168.10.11:55551"
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
    # Close briskly, then switch to the lower hold torque as soon as contact is
    # confirmed.  This shortens the only blocking section between the lowest
    # pick waypoint and the lift motion without lifting on an unconfirmed grip.
    left_grip_speed: int = 80
    right_grip_speed: int = 80
    left_grip_torque: int = 40
    right_grip_torque: int = 40
    left_hold_torque: int = 20
    right_hold_torque: int = 20
    left_current_threshold: int = 120
    right_current_threshold: int = 120
    left_poll_interval: float = 0.02
    right_poll_interval: float = 0.02
    left_contact_grace: float = 0.1
    right_contact_grace: float = 0.1
    left_progress_epsilon: int = 2
    right_progress_epsilon: int = 2
    left_stall_samples: int = 3
    right_stall_samples: int = 3
    left_empty_grip_margin: int = 50
    right_empty_grip_margin: int = 50
    left_target_pos_tolerance: int = 120
    right_target_pos_tolerance: int = 120
    left_timeout: float = 5.0
    right_timeout: float = 5.0
    left_grip_done_wait: float = 0.05
    right_grip_done_wait: float = 0.05
    left_release_target: int = 1000
    right_release_target: int = 1000
    left_release_speed: int = 60
    right_release_speed: int = 60
    left_release_torque: int = 20
    right_release_torque: int = 20
    left_release_wait: float = 0.05
    right_release_wait: float = 0.05

    @property
    def server(self) -> str:
        return self.right_server

    @property
    def clamp_pos(self) -> int:
        return self.right_clamp_pos

    @property
    def open_pos(self) -> int:
        return self.right_open_pos

    @property
    def max_itinerary(self) -> int:
        return self.right_max_itinerary

    @property
    def speed_coe(self) -> int:
        return self.right_speed_coe

    @property
    def min_pos(self) -> int:
        return self.right_min_pos

    @property
    def max_pos(self) -> int:
        return self.right_max_pos

    @property
    def grip_speed(self) -> int:
        return self.right_grip_speed

    @property
    def grip_torque(self) -> int:
        return self.right_grip_torque

    @property
    def hold_torque(self) -> int:
        return self.right_hold_torque

    @property
    def current_threshold(self) -> int:
        return self.right_current_threshold

    @property
    def poll_interval(self) -> float:
        return self.right_poll_interval

    @property
    def contact_grace(self) -> float:
        return self.right_contact_grace

    @property
    def progress_epsilon(self) -> int:
        return self.right_progress_epsilon

    @property
    def stall_samples(self) -> int:
        return self.right_stall_samples

    @property
    def empty_grip_margin(self) -> int:
        return self.right_empty_grip_margin

    @property
    def target_pos_tolerance(self) -> int:
        return self.right_target_pos_tolerance

    @property
    def timeout(self) -> float:
        return self.right_timeout

    @property
    def grip_done_wait(self) -> float:
        return self.right_grip_done_wait

    @property
    def release_target(self) -> int:
        return self.right_release_target

    @property
    def release_speed(self) -> int:
        return self.right_release_speed

    @property
    def release_torque(self) -> int:
        return self.right_release_torque

    @property
    def release_wait(self) -> float:
        return self.right_release_wait


@dataclass(frozen=True)
class HomeDefaults:
    right_xyz: tuple[float, float, float] = (0.25, -0.25, 0.81)
    left_xyz: tuple[float, float, float] = (0.25, 0.25, 0.81)
    safe_z_m: float = 0.95
    side_clearance_y_m: float = 0.28


@dataclass(frozen=True)
class PutDefaults:
    target_hold_sec: float = 0.0
    home_hold_sec: float = 0.05
    keep_put_pose: bool = True


@dataclass(frozen=True)
class SideApproachDefaults:
    enabled: bool = False
    offset_y_m: float = 0.0   #what the
    min_abs_y_m: float = 0.035
    lift_m: float = 0.06
    wrist_outward_bias_deg: float = 8.0
    max_wrist_deviation_deg: float = 25.0


@dataclass(frozen=True)
class TargetTrajectoryDefaults:
    publish_rate_hz: float = 80.0
    step_m: float = 0.005
    step_deg: float = 1.0
    min_steps: int = 15
    speed_mps: float = 0.15
    angular_speed_dps: float = 35.0
    plot_dir: Path = PROJECT_ROOT / "captures" / "request_ik_trajectories"


GRIP_SIGNAL_DEFAULTS = GripSignalDefaults()
GRIPPER_DEFAULTS = GripperDefaults()
HOME_DEFAULTS = HomeDefaults()
PUT_DEFAULTS = PutDefaults()
SIDE_APPROACH_DEFAULTS = SideApproachDefaults()
TARGET_TRAJECTORY_DEFAULTS = TargetTrajectoryDefaults()

# Compatibility exports used by other modules.
DEFAULT_GRIP_SETTLE_SEC = GRIP_SIGNAL_DEFAULTS.settle_sec
DEFAULT_TARGET_PUBLISH_RATE_HZ = TARGET_TRAJECTORY_DEFAULTS.publish_rate_hz
DEFAULT_TARGET_TRAJECTORY_STEP_M = TARGET_TRAJECTORY_DEFAULTS.step_m
DEFAULT_TARGET_TRAJECTORY_STEP_DEG = TARGET_TRAJECTORY_DEFAULTS.step_deg
DEFAULT_TARGET_TRAJECTORY_MIN_STEPS = TARGET_TRAJECTORY_DEFAULTS.min_steps
DEFAULT_TARGET_TRAJECTORY_SPEED_MPS = TARGET_TRAJECTORY_DEFAULTS.speed_mps
DEFAULT_TARGET_TRAJECTORY_ANGULAR_SPEED_DPS = (
    TARGET_TRAJECTORY_DEFAULTS.angular_speed_dps
)
DEFAULT_TRAJECTORY_PLOT_DIR = TARGET_TRAJECTORY_DEFAULTS.plot_dir
DEFAULT_JOINT_TRAJECTORY_CSV_DIR = (
    PROJECT_ROOT / "captures" / "request_ik_joint_trajectories"
)


@dataclass(frozen=True)
class GraspConfig:
    force_object_z: bool = True
    forced_object_z_m: float = 0.685
    pregrasp_distance_m: float = 0.05
    lift_distance_m: float = 0.08
    approach_axis: str = "z"
    approach_sign: float = -1.0
    use_flowpose_grasp_rotation: bool = False
    use_cube_z_symmetry_grasp_policy: bool = False
    ik_grasp_tcp_offset_m: tuple[float, float, float] = (-0.06, 0.0, -0.02)
    ik_pregrasp_extra_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ik_orientation_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    ik_downward_tilt_deg: float = 45.0
    ik_downward_tilt_left_deg: float | None = None
    ik_downward_tilt_right_deg: float | None = None
    ik_downward_tilt_axis: str = "y"
    ik_downward_tilt_y_deg: float = 0.0
    ik_downward_tilt_y_left_deg: float | None = None
    ik_downward_tilt_y_right_deg: float | None = None
    ik_downward_tilt_frame: str = "local"
    visualize_grasp_path: bool = True
    save_joint_trajectory_csv: bool = False


DEFAULT_GRASP_CONFIG = GraspConfig()

def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected TRUE/FALSE, got {value!r}")


def parse_optional_bool(value: str | bool | None) -> bool | None:
    if value is None:
        return None
    return parse_bool(value)


def load_tool_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw if isinstance(raw, dict) else {}


def parse_float_tuple(
    value: object,
    *,
    expected_len: int,
    fallback: tuple[float, ...],
    name: str,
) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        print(f"[grasp_config] invalid {name}={value!r}; using {fallback}", flush=True)
        return fallback
    if len(values) != expected_len or not all(np.isfinite(values)):
        print(f"[grasp_config] invalid {name}={value!r}; using {fallback}", flush=True)
        return fallback
    return values


def parse_config_bool(value: object, *, fallback: bool, name: str) -> bool:
    try:
        return parse_bool(value)  # type: ignore[arg-type]
    except argparse.ArgumentTypeError:
        print(f"[grasp_config] invalid {name}={value!r}; using {fallback}", flush=True)
        return fallback


def parse_config_float(value: object, *, fallback: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        print(f"[grasp_config] invalid {name}={value!r}; using {fallback}", flush=True)
        return fallback
    if not np.isfinite(result):
        print(f"[grasp_config] invalid {name}={value!r}; using {fallback}", flush=True)
        return fallback
    return result


def tool_grasp_defaults_from_yaml(path: Path) -> GraspConfig:
    try:
        raw = load_tool_yaml(path)
    except OSError as exc:
        print(
            f"[grasp_config] unable to read {path}: {exc}; using built-in defaults",
            flush=True,
        )
        return DEFAULT_GRASP_CONFIG
    except yaml.YAMLError as exc:
        print(
            f"[grasp_config] invalid YAML {path}: {exc}; using built-in defaults",
            flush=True,
        )
        return DEFAULT_GRASP_CONFIG

    defaults = raw.get("defaults")
    if not isinstance(defaults, dict):
        return DEFAULT_GRASP_CONFIG

    cfg = DEFAULT_GRASP_CONFIG
    axis = str(defaults.get("ik_downward_tilt_axis", cfg.ik_downward_tilt_axis)).lower()
    if axis not in {"x", "y", "z"}:
        print(
            f"[grasp_config] invalid ik_downward_tilt_axis={axis!r}; using {cfg.ik_downward_tilt_axis!r}",
            flush=True,
        )
        axis = cfg.ik_downward_tilt_axis
    frame = str(
        defaults.get("ik_downward_tilt_frame", cfg.ik_downward_tilt_frame)
    ).lower()
    if frame not in {"local", "base"}:
        print(
            f"[grasp_config] invalid ik_downward_tilt_frame={frame!r}; using {cfg.ik_downward_tilt_frame!r}",
            flush=True,
        )
        frame = cfg.ik_downward_tilt_frame
    approach_axis = str(defaults.get("approach_axis", cfg.approach_axis)).lower()
    if approach_axis not in {"x", "y", "z"}:
        print(
            f"[grasp_config] invalid approach_axis={approach_axis!r}; using {cfg.approach_axis!r}",
            flush=True,
        )
        approach_axis = cfg.approach_axis

    return GraspConfig(
        force_object_z=parse_config_bool(
            defaults.get("force_object_z", cfg.force_object_z),
            fallback=cfg.force_object_z,
            name="force_object_z",
        ),
        forced_object_z_m=parse_config_float(
            defaults.get("forced_object_z_m", cfg.forced_object_z_m),
            fallback=cfg.forced_object_z_m,
            name="forced_object_z_m",
        ),
        pregrasp_distance_m=parse_config_float(
            defaults.get("pregrasp_distance_m", cfg.pregrasp_distance_m),
            fallback=cfg.pregrasp_distance_m,
            name="pregrasp_distance_m",
        ),
        lift_distance_m=parse_config_float(
            defaults.get("lift_distance_m", cfg.lift_distance_m),
            fallback=cfg.lift_distance_m,
            name="lift_distance_m",
        ),
        approach_axis=approach_axis,
        approach_sign=parse_config_float(
            defaults.get("approach_sign", cfg.approach_sign),
            fallback=cfg.approach_sign,
            name="approach_sign",
        ),
        use_flowpose_grasp_rotation=parse_config_bool(
            defaults.get(
                "use_flowpose_grasp_rotation",
                cfg.use_flowpose_grasp_rotation,
            ),
            fallback=cfg.use_flowpose_grasp_rotation,
            name="use_flowpose_grasp_rotation",
        ),
        use_cube_z_symmetry_grasp_policy=parse_config_bool(
            defaults.get(
                "use_cube_z_symmetry_grasp_policy",
                cfg.use_cube_z_symmetry_grasp_policy,
            ),
            fallback=cfg.use_cube_z_symmetry_grasp_policy,
            name="use_cube_z_symmetry_grasp_policy",
        ),
        ik_grasp_tcp_offset_m=parse_float_tuple(
            defaults.get("ik_grasp_tcp_offset_m", cfg.ik_grasp_tcp_offset_m),
            expected_len=3,
            fallback=cfg.ik_grasp_tcp_offset_m,
            name="ik_grasp_tcp_offset_m",
        ),  # type: ignore[arg-type]
        ik_pregrasp_extra_offset_m=parse_float_tuple(
            defaults.get(
                "ik_pregrasp_extra_offset_m",
                cfg.ik_pregrasp_extra_offset_m,
            ),
            expected_len=3,
            fallback=cfg.ik_pregrasp_extra_offset_m,
            name="ik_pregrasp_extra_offset_m",
        ),  # type: ignore[arg-type]
        ik_orientation_quat=parse_float_tuple(
            defaults.get("ik_orientation_quat", cfg.ik_orientation_quat),
            expected_len=4,
            fallback=cfg.ik_orientation_quat,
            name="ik_orientation_quat",
        ),  # type: ignore[arg-type]
        ik_downward_tilt_deg=parse_config_float(
            defaults.get("ik_downward_tilt_deg", cfg.ik_downward_tilt_deg),
            fallback=cfg.ik_downward_tilt_deg,
            name="ik_downward_tilt_deg",
        ),
        ik_downward_tilt_left_deg=(
            None
            if defaults.get("ik_downward_tilt_left_deg") is None
            else parse_config_float(
                defaults.get("ik_downward_tilt_left_deg"),
                fallback=cfg.ik_downward_tilt_deg,
                name="ik_downward_tilt_left_deg",
            )
        ),
        ik_downward_tilt_right_deg=(
            None
            if defaults.get("ik_downward_tilt_right_deg") is None
            else parse_config_float(
                defaults.get("ik_downward_tilt_right_deg"),
                fallback=cfg.ik_downward_tilt_deg,
                name="ik_downward_tilt_right_deg",
            )
        ),
        ik_downward_tilt_axis=axis,
        ik_downward_tilt_y_deg=parse_config_float(
            defaults.get("ik_downward_tilt_y_deg", cfg.ik_downward_tilt_y_deg),
            fallback=cfg.ik_downward_tilt_y_deg,
            name="ik_downward_tilt_y_deg",
        ),
        ik_downward_tilt_y_left_deg=(
            None
            if defaults.get("ik_downward_tilt_y_left_deg") is None
            else parse_config_float(
                defaults.get("ik_downward_tilt_y_left_deg"),
                fallback=cfg.ik_downward_tilt_y_deg,
                name="ik_downward_tilt_y_left_deg",
            )
        ),
        ik_downward_tilt_y_right_deg=(
            None
            if defaults.get("ik_downward_tilt_y_right_deg") is None
            else parse_config_float(
                defaults.get("ik_downward_tilt_y_right_deg"),
                fallback=cfg.ik_downward_tilt_y_deg,
                name="ik_downward_tilt_y_right_deg",
            )
        ),
        ik_downward_tilt_frame=frame,
        visualize_grasp_path=parse_config_bool(
            defaults.get("visualize_grasp_path", cfg.visualize_grasp_path),
            fallback=cfg.visualize_grasp_path,
            name="visualize_grasp_path",
        ),
        save_joint_trajectory_csv=parse_config_bool(
            defaults.get(
                "save_joint_trajectory_csv",
                cfg.save_joint_trajectory_csv,
            ),
            fallback=cfg.save_joint_trajectory_csv,
            name="save_joint_trajectory_csv",
        ),
    )


def apply_grasp_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    cli_tilt_override = getattr(args, "ik_downward_tilt_deg", None) is not None
    cli_tilt_y_override = getattr(args, "ik_downward_tilt_y_deg", None) is not None
    config = tool_grasp_defaults_from_yaml(Path(args.tool_template_path).expanduser())
    for field_name, config_value in config.__dict__.items():
        if getattr(args, field_name, None) is None:
            setattr(args, field_name, config_value)
    if cli_tilt_override:
        args.ik_downward_tilt_left_deg = None
        args.ik_downward_tilt_right_deg = None
    if cli_tilt_y_override:
        args.ik_downward_tilt_y_left_deg = None
        args.ik_downward_tilt_y_right_deg = None
    if getattr(args, "target_trajectory_plot", None) is not None:
        args.visualize_grasp_path = bool(args.target_trajectory_plot)
    args.target_trajectory_plot = bool(args.visualize_grasp_path)
    print(
        "[grasp_config] defaults "
        f"source={args.tool_template_path} "
        f"force_z={bool(args.force_object_z)} "
        f"object_z={float(args.forced_object_z_m):.4f}m "
        f"pregrasp={float(args.pregrasp_distance_m):.3f}m "
        f"tcp_offset={_format_xyz(np.asarray(args.ik_grasp_tcp_offset_m, dtype=np.float64))} "
        f"flowpose_rotation={bool(args.use_flowpose_grasp_rotation)} "
        f"cube_z_symmetry_policy={bool(args.use_cube_z_symmetry_grasp_policy)} "
        f"tilt={float(args.ik_downward_tilt_deg):.2f}deg/"
        f"left={getattr(args, 'ik_downward_tilt_left_deg', None)} "
        f"right={getattr(args, 'ik_downward_tilt_right_deg', None)} "
        f"{args.ik_downward_tilt_axis}+y={float(args.ik_downward_tilt_y_deg):.2f}deg/"
        f"y_left={getattr(args, 'ik_downward_tilt_y_left_deg', None)} "
        f"y_right={getattr(args, 'ik_downward_tilt_y_right_deg', None)} "
        f"{args.ik_downward_tilt_frame} "
        f"side_approach_policy=disabled "
        f"visualize_grasp_path={bool(args.visualize_grasp_path)} "
        f"save_joint_trajectory_csv={bool(args.save_joint_trajectory_csv)}",
        flush=True,
    )
    return normalize_gripper_args(args)


def normalize_gripper_args(args: argparse.Namespace) -> argparse.Namespace:
    args.left_gripper_clamp_pos = int(args.left_gripper_clamp_pos)
    args.left_gripper_open_pos = int(args.left_gripper_open_pos)
    args.left_gripper_max_itinerary = max(int(args.left_gripper_max_itinerary), 1)
    args.left_gripper_speed_coe = max(int(args.left_gripper_speed_coe), 1)
    args.left_gripper_min_pos = int(clamp(args.left_gripper_min_pos, 0, 1000))
    args.left_gripper_max_pos = int(clamp(args.left_gripper_max_pos, 0, 1000))
    args.left_gripper_release_target = int(
        clamp(args.left_gripper_release_target, 0, 1000)
    )
    args.left_gripper_grip_speed = int(clamp(args.left_gripper_grip_speed, 10, 100))
    args.left_gripper_release_speed = int(
        clamp(args.left_gripper_release_speed, 10, 100)
    )
    args.left_gripper_grip_torque = int(clamp(args.left_gripper_grip_torque, 10, 100))
    args.left_gripper_hold_torque = int(clamp(args.left_gripper_hold_torque, 10, 100))
    args.left_gripper_release_torque = int(
        clamp(args.left_gripper_release_torque, 10, 100)
    )
    args.left_gripper_current_threshold = max(
        int(args.left_gripper_current_threshold), 0
    )
    args.left_gripper_poll_interval = max(float(args.left_gripper_poll_interval), 0.02)
    args.left_gripper_contact_grace = max(float(args.left_gripper_contact_grace), 0.0)
    args.left_gripper_progress_epsilon = max(
        int(args.left_gripper_progress_epsilon), 0
    )
    args.left_gripper_stall_samples = max(int(args.left_gripper_stall_samples), 1)
    args.left_gripper_empty_grip_margin = max(
        int(args.left_gripper_empty_grip_margin), 0
    )
    args.left_gripper_target_pos_tolerance = max(
        int(args.left_gripper_target_pos_tolerance), 0
    )
    args.left_gripper_timeout = max(float(args.left_gripper_timeout), 0.1)
    args.left_gripper_grip_done_wait = max(
        float(args.left_gripper_grip_done_wait), 0.0
    )
    args.left_gripper_release_wait = max(float(args.left_gripper_release_wait), 0.0)

    args.gripper_min_pos = int(clamp(args.gripper_min_pos, 0, 1000))
    args.gripper_max_pos = int(clamp(args.gripper_max_pos, 0, 1000))
    args.gripper_release_target = int(clamp(args.gripper_release_target, 0, 1000))
    args.gripper_grip_speed = int(clamp(args.gripper_grip_speed, 10, 100))
    args.gripper_release_speed = int(clamp(args.gripper_release_speed, 10, 100))
    args.gripper_grip_torque = int(clamp(args.gripper_grip_torque, 10, 100))
    args.gripper_hold_torque = int(clamp(args.gripper_hold_torque, 10, 100))
    args.gripper_release_torque = int(clamp(args.gripper_release_torque, 10, 100))
    args.gripper_poll_interval = max(float(args.gripper_poll_interval), 0.02)
    args.gripper_contact_grace = max(float(args.gripper_contact_grace), 0.0)
    args.gripper_progress_epsilon = max(int(args.gripper_progress_epsilon), 0)
    args.gripper_stall_samples = max(int(args.gripper_stall_samples), 1)
    args.gripper_empty_grip_margin = max(int(args.gripper_empty_grip_margin), 0)
    args.gripper_target_pos_tolerance = max(
        int(args.gripper_target_pos_tolerance), 0
    )
    args.gripper_timeout = max(float(args.gripper_timeout), 0.1)
    args.gripper_grip_done_wait = max(float(args.gripper_grip_done_wait), 0.0)
    args.gripper_release_wait = max(float(args.gripper_release_wait), 0.0)
    args.grip_settle_sec = max(float(args.grip_settle_sec), 0.0)
    args.grip_post_confirm_hold_sec = max(
        float(args.grip_post_confirm_hold_sec),
        0.0,
    )
    args.grip_lift_hold_sec = max(float(args.grip_lift_hold_sec), 0.0)
    args.grip_drop_close_delta = max(int(args.grip_drop_close_delta), 0)
    args.grip_drop_poll_interval = max(float(args.grip_drop_poll_interval), 0.02)
    args.put_target_hold_sec = max(float(args.put_target_hold_sec), 0.0)
    args.put_home_hold_sec = max(float(args.put_home_hold_sec), 0.0)
    args.gripper_calibration_tolerance = max(
        int(args.gripper_calibration_tolerance),
        0,
    )
    args.gripper_allow_homing_fallback = bool(args.gripper_allow_homing_fallback)
    args.dual_gripper = bool(args.dual_gripper)
    args.left_gripper_server = str(args.left_gripper_server)
    args.right_gripper_server = str(args.right_gripper_server)
    args.grip_signal_port = int(args.grip_signal_port)
    args.side_approach_offset_y_m = max(float(args.side_approach_offset_y_m), 0.0)
    args.side_approach_min_abs_y_m = max(
        float(args.side_approach_min_abs_y_m),
        0.0,
    )
    args.side_approach_lift_m = max(float(args.side_approach_lift_m), 0.0)
    args.side_approach_wrist_outward_bias_deg = max(
        float(args.side_approach_wrist_outward_bias_deg),
        0.0,
    )
    args.side_approach_max_wrist_deviation_deg = max(
        float(args.side_approach_max_wrist_deviation_deg),
        0.0,
    )
    args.gripper_connect_attempts = max(int(args.gripper_connect_attempts), 1)
    args.gripper_connect_timeout_sec = max(
        float(args.gripper_connect_timeout_sec),
        0.1,
    )
    args.gripper_connect_retry_delay_sec = max(
        float(args.gripper_connect_retry_delay_sec),
        0.0,
    )
    args.sam3_roi_xyxy = normalize_roi_xyxy(
        args.sam3_roi_xyxy,
        width=args.width,
        height=args.height,
    )
    if args.gripper_min_pos > args.gripper_max_pos:
        raise SystemExit("--gripper-min-pos must be <= --gripper-max-pos")
    if args.left_gripper_min_pos > args.left_gripper_max_pos:
        raise SystemExit("--left-gripper-min-pos must be <= --left-gripper-max-pos")
    return args


def normalize_roi_xyxy(
    roi_xyxy: tuple[int, int, int, int] | list[int] | None,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if roi_xyxy is None:
        return None
    x_min, y_min, x_max, y_max = (int(value) for value in roi_xyxy)
    x_min = int(clamp(x_min, 0, max(int(width) - 1, 0)))
    x_max = int(clamp(x_max, 0, int(width)))
    y_min = int(clamp(y_min, 0, max(int(height) - 1, 0)))
    y_max = int(clamp(y_max, 0, int(height)))
    if x_max <= x_min or y_max <= y_min:
        raise SystemExit(
            "--sam3-roi-xyxy must satisfy x_max > x_min and y_max > y_min"
        )
    return (x_min, y_min, x_max, y_max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RealSense + SAM3 + FlowPose 6D pose demo"
    )
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--prompts", default="toy,yellow_screwdriver_handle,pen,rectangular object,plastic part,ribbon")
    parser.add_argument("--sam3-checkpoint-path", default="/model/sam3.pt")
    parser.add_argument("--sam3-root", default=None)
    parser.add_argument(
        "--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD
    )
    parser.add_argument(
        "--dedup-iou-threshold", type=float, default=DEFAULT_DEDUP_IOU_THRESHOLD
    )
    parser.add_argument(
        "--suppress-contained-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--containment-threshold", type=float, default=DEFAULT_CONTAINMENT_THRESHOLD
    )
    parser.add_argument(
        "--bbox-containment-threshold",
        type=float,
        default=DEFAULT_BBOX_CONTAINMENT_THRESHOLD,
    )
    parser.add_argument(
        "--containment-min-area-ratio",
        type=float,
        default=DEFAULT_CONTAINMENT_MIN_AREA_RATIO,
    )
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument(
        "--sam3-roi-xyxy",
        nargs=4,
        type=int,
        default=DEFAULT_SAM3_ROI_XYXY,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help=(
            "Only keep SAM3 detections whose bbox center is inside this pixel ROI. "
            "Defaults to the ROI calibrated by test1.py."
        ),
    )
    parser.add_argument(
        "--sam3-roi-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter SAM3 detections by --sam3-roi-xyxy before FlowPose.",
    )
    parser.add_argument(
        "--sam3-device", default="auto", choices=["auto", "cuda", "cpu"]
    )
    parser.add_argument("--flowpose-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--flow-model-path", default=str(DEFAULT_FLOW_MODEL_PATH))
    parser.add_argument("--scale-model-path", default=str(DEFAULT_SCALE_MODEL_PATH))
    parser.add_argument("--dino-repo-path", default=None)
    parser.add_argument("--dino-ckpt-path", default=None)
    parser.add_argument("--capture-dir", default=str(DEFAULT_CAPTURE_DIR))
    parser.add_argument(
        "--ros2-publish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish FlowPose outputs to /tf and RViz markers.",
    )
    parser.add_argument("--ros2-parent-frame-id", default="camera_rgb_link")
    parser.add_argument("--ros2-base-frame-id", default="base_link")
    parser.add_argument("--ros2-tf-topic", default="/tf")
    parser.add_argument("--ros2-marker-topic", default="/flowpose/grasp_markers")
    parser.add_argument("--ros2-publish-rate-hz", type=float, default=5.0)
    parser.add_argument(
        "--pregrasp-distance-m",
        type=float,
        default=None,
        help="Override tool.yaml defaults.pregrasp_distance_m.",
    )
    parser.add_argument("--robot-xacro-path", default=str(DEFAULT_ROBOT_XACRO_PATH))
    parser.add_argument("--camera-joint", default="camera_joint")
    parser.add_argument(
        "--show-base-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show FlowPose targets converted into base_link coordinates.",
    )
    parser.add_argument(
        "--auto-pipeline-on-a",
        type=parse_bool,
        default=True,
        metavar="TRUE/FALSE",
        help=(
            "TRUE makes A run capture -> SAM3 -> FlowPose -> grasp automatically; "
            "FALSE restores manual A/B/C steps."
        ),
    )
    parser.add_argument(
        "--force-object-z",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override tool.yaml defaults.force_object_z.",
    )
    parser.add_argument(
        "--forced-object-z-m",
        type=float,
        default=None,
        help="Override tool.yaml defaults.forced_object_z_m.",
    )
    parser.add_argument(
        "--left-target-topic",
        default="/control/request_ik_tester/target_poseL",
        help="PoseStamped topic consumed by request_ik_tester for the left hand.",
    )
    parser.add_argument(
        "--right-target-topic",
        default="/control/request_ik_tester/target_poseR",
        help="PoseStamped topic consumed by request_ik_tester for the right hand.",
    )
    parser.add_argument(
        "--target-command-mode",
        choices=["auto", "pose_stream", "cartesian_trajectory"],
        default="auto",
        help=(
            "How to command request_ik_tester targets. auto publishes timestamped "
            "Cartesian trajectories when a trajectory subscriber exists, otherwise "
            "falls back to PoseStamped streaming."
        ),
    )
    parser.add_argument(
        "--left-trajectory-topic",
        default="/control/request_ik_tester/target_trajectoryL",
        help="MultiDOFJointTrajectory topic for left-hand Cartesian trajectories.",
    )
    parser.add_argument(
        "--right-trajectory-topic",
        default="/control/request_ik_tester/target_trajectoryR",
        help="MultiDOFJointTrajectory topic for right-hand Cartesian trajectories.",
    )
    parser.add_argument(
        "--left-trajectory-joint-name",
        default="left_tcp",
        help="Joint name stored in left-hand MultiDOFJointTrajectory messages.",
    )
    parser.add_argument(
        "--right-trajectory-joint-name",
        default="right_tcp",
        help="Joint name stored in right-hand MultiDOFJointTrajectory messages.",
    )
    parser.add_argument(
        "--target-publish-rate-hz",
        type=float,
        default=DEFAULT_TARGET_PUBLISH_RATE_HZ,
        help="Rate used for request_ik_tester target publishing and trajectory steps.",
    )
    parser.add_argument(
        "--target-publish-sec",
        type=float,
        default=0.5,
        help="Seconds to keep publishing the final target after a trajectory finishes.",
    )
    parser.add_argument(
        "--target-smooth-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish Cartesian waypoints to make request_ik_tester motion smoother.",
    )
    parser.add_argument(
        "--target-trajectory-step-m",
        type=float,
        default=DEFAULT_TARGET_TRAJECTORY_STEP_M,
        help="Maximum Cartesian distance between smooth target waypoints.",
    )
    parser.add_argument(
        "--target-trajectory-step-deg",
        type=float,
        default=DEFAULT_TARGET_TRAJECTORY_STEP_DEG,
        help="Maximum orientation angle between smooth target waypoints.",
    )
    parser.add_argument(
        "--target-trajectory-min-steps",
        type=int,
        default=TARGET_TRAJECTORY_DEFAULTS.min_steps,
        help="Minimum interpolation samples per non-zero path segment.",
    )
    parser.add_argument(
        "--target-trajectory-speed-mps",
        type=float,
        default=DEFAULT_TARGET_TRAJECTORY_SPEED_MPS,
        help=(
            "Approximate Cartesian target speed limit. Lower values slow motion "
            "and insert more interpolation samples."
        ),
    )
    parser.add_argument(
        "--target-trajectory-angular-speed-dps",
        type=float,
        default=DEFAULT_TARGET_TRAJECTORY_ANGULAR_SPEED_DPS,
        help=(
            "Approximate orientation target speed limit in degrees/sec. Lower values "
            "slow rotational motion and insert more interpolation samples."
        ),
    )
    parser.add_argument(
        "--target-trajectory-plot-dir",
        default=str(DEFAULT_TRAJECTORY_PLOT_DIR),
        help="Directory where grasp path visualization PNG files are saved.",
    )
    parser.add_argument(
        "--joint-trajectory-csv-dir",
        default=str(DEFAULT_JOINT_TRAJECTORY_CSV_DIR),
        help="Directory where timestamped joint trajectory CSV files are saved.",
    )
    parser.add_argument(
        "--target-trajectory-plot",
        type=parse_bool,
        default=None,
        metavar="TRUE/FALSE",
        help=(
            "Legacy alias for --visualize-grasp-path. TRUE saves grasp path "
            "visualizations; FALSE skips path recording/rendering."
        ),
    )
    parser.add_argument(
        "--visualize-grasp-path",
        type=parse_optional_bool,
        default=None,
        metavar="TRUE/FALSE",
        help=(
            "Override tool.yaml defaults.visualize_grasp_path. TRUE saves a grasp "
            "path PNG after publishing; FALSE skips visualization."
        ),
    )
    parser.add_argument(
        "--save-joint-trajectory-csv",
        type=parse_optional_bool,
        default=None,
        metavar="TRUE/FALSE",
        help=(
            "Override tool.yaml defaults.save_joint_trajectory_csv. TRUE saves "
            "one timestamped joint trajectory CSV row per trajectory frame."
        ),
    )
    parser.add_argument(
        "--tool-template-path",
        default=str(DEFAULT_TOOL_TEMPLATE_PATH),
        help="Path to pick path templates; only the pick entries are used.",
    )
    parser.add_argument(
        "--use-tool-pick-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use config/tool.yaml pick waypoints for visual grasp targets when available.",
    )
    parser.add_argument(
        "--use-side-approach-grasp-policy",
        type=parse_bool,
        default=SIDE_APPROACH_DEFAULTS.enabled,
        metavar="TRUE/FALSE",
        help=(
            "Deprecated compatibility option; side-first grasp adjustment is disabled."
        ),
    )
    parser.add_argument(
        "--side-approach-offset-y-m",
        type=float,
        default=SIDE_APPROACH_DEFAULTS.offset_y_m,
        help="Extra outward Y distance for the side-first grasp policy.",
    )
    parser.add_argument(
        "--side-approach-min-abs-y-m",
        type=float,
        default=SIDE_APPROACH_DEFAULTS.min_abs_y_m,
        help="Minimum |Y| clearance from base_link centerline for side-first grasp targets.",
    )
    parser.add_argument(
        "--side-approach-lift-m",
        type=float,
        default=SIDE_APPROACH_DEFAULTS.lift_m,
        help="Minimum Z lift for the inserted side approach waypoint.",
    )
    parser.add_argument(
        "--side-approach-wrist-outward-bias-deg",
        type=float,
        default=SIDE_APPROACH_DEFAULTS.wrist_outward_bias_deg,
        help=(
            "Small base-frame yaw bias toward the arm's outside when FlowPose "
            "orientation is too far from the natural wrist pose."
        ),
    )
    parser.add_argument(
        "--side-approach-max-wrist-deviation-deg",
        type=float,
        default=SIDE_APPROACH_DEFAULTS.max_wrist_deviation_deg,
        help=(
            "Maximum allowed wrist orientation deviation from natural pose before "
            "rejecting a FlowPose/reference orientation."
        ),
    )
    parser.add_argument("--ik-frame-id", default="base_link")
    parser.add_argument(
        "--ik-hand",
        choices=["auto", "left", "right"],
        default="auto",
        help="Hand controlled by IKRequest; auto selects by target y sign.",
    )
    parser.add_argument(
        "--ik-target-stage",
        choices=["pregrasp", "grasp"],
        default="pregrasp",
        help="Which computed gripper target S should publish.",
    )
    parser.add_argument("--ik-target-index", type=int, default=0)
    parser.add_argument(
        "--use-flowpose-grasp-rotation",
        type=parse_optional_bool,
        default=None,
        metavar="TRUE/FALSE",
        help="Override tool.yaml defaults.use_flowpose_grasp_rotation.",
    )
    parser.add_argument(
        "--use-cube-z-symmetry-grasp-policy",
        type=parse_optional_bool,
        default=None,
        metavar="TRUE/FALSE",
        help=(
            "Override tool.yaml defaults.use_cube_z_symmetry_grasp_policy. "
            "When TRUE, cube FlowPose poses may rotate around local Z by "
            "0/90/180/-90 degrees so local -X faces the selected gripper side."
        ),
    )
    parser.add_argument(
        "--ik-grasp-tcp-offset-m",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override tool.yaml defaults.ik_grasp_tcp_offset_m.",
    )
    parser.add_argument(
        "--ik-pregrasp-extra-offset-m",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override tool.yaml defaults.ik_pregrasp_extra_offset_m.",
    )
    parser.add_argument(
        "--ik-orientation-quat",
        nargs=4,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "W"),
        help="Override tool.yaml defaults.ik_orientation_quat.",
    )
    parser.add_argument(
        "--lift-distance-m",
        type=float,
        default=None,
        help="Override tool.yaml defaults.lift_distance_m.",
    )
    parser.add_argument(
        "--approach-axis",
        default=None,
        choices=["x", "y", "z"],
        help="Override tool.yaml defaults.approach_axis.",
    )
    parser.add_argument(
        "--approach-sign",
        type=float,
        default=None,
        help="Override tool.yaml defaults.approach_sign.",
    )
    parser.add_argument(
        "--ik-downward-tilt-deg",
        type=float,
        default=None,
        help="Override tool.yaml defaults.ik_downward_tilt_deg.",
    )
    parser.add_argument(
        "--ik-downward-tilt-axis",
        choices=["x", "y", "z"],
        default=None,
        help="Override tool.yaml defaults.ik_downward_tilt_axis.",
    )
    parser.add_argument(
        "--ik-downward-tilt-y-deg",
        type=float,
        default=None,
        help="Override tool.yaml defaults.ik_downward_tilt_y_deg.",
    )
    parser.add_argument(
        "--ik-downward-tilt-frame",
        choices=["local", "base"],
        default=None,
        help="Override tool.yaml defaults.ik_downward_tilt_frame.",
    )
    parser.add_argument(
        "--right-home-xyz",
        nargs=3,
        type=float,
        default=HOME_DEFAULTS.right_xyz,
        metavar=("X", "Y", "Z"),
        help="Right hand home target published by H.",
    )
    parser.add_argument(
        "--left-home-xyz",
        nargs=3,
        type=float,
        default=HOME_DEFAULTS.left_xyz,
        metavar=("X", "Y", "Z"),
        help="Left hand home target published by J.",
    )

    parser.add_argument(
        "--home-safe-z-m",
        type=float,
        default=HOME_DEFAULTS.safe_z_m,
        help="Safe z height used before moving sideways/back to home.",
    )
    parser.add_argument(
        "--home-side-clearance-y-m",
        type=float,
        default=HOME_DEFAULTS.side_clearance_y_m,
        help="Minimum absolute y used as the left/right side corridor during home return.",
    )
    parser.add_argument("--grip-signal-host", default=GRIP_SIGNAL_DEFAULTS.host)
    parser.add_argument(
        "--grip-signal-port", type=int, default=GRIP_SIGNAL_DEFAULTS.port
    )
    parser.add_argument(
        "--grip-signal-timeout-sec",
        type=float,
        default=GRIP_SIGNAL_DEFAULTS.timeout_sec,
    )
    parser.add_argument(
        "--grip-signal-command-timeout-sec",
        type=float,
        default=GRIP_SIGNAL_DEFAULTS.command_timeout_sec,
        help="Seconds to wait for a gripper grip/release command to finish.",
    )
    parser.add_argument(
        "--grip-signal-auto-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start daimon_stuff/grip_signal_receiver.py together with this demo.",
    )
    parser.add_argument(
        "--grip-signal-receiver-path",
        default=str(GRIP_SIGNAL_DEFAULTS.receiver_path),
        help="Path to grip_signal_receiver.py used when auto-start is enabled.",
    )
    parser.add_argument(
        "--grip-signal-token",
        default=None,
        help="Optional token for grip_signal_receiver.py; messages become '<token> <command>'.",
    )
    parser.add_argument(
        "--grip-drop-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect a dropped object from extra gripper closing during transport.",
    )
    parser.add_argument(
        "--grip-drop-close-delta",
        type=int,
        default=GRIP_SIGNAL_DEFAULTS.drop_close_delta,
        help="Closing-position change beyond the successful-grasp baseline that means drop.",
    )
    parser.add_argument(
        "--grip-drop-poll-interval",
        type=float,
        default=GRIP_SIGNAL_DEFAULTS.drop_poll_interval,
        help="Seconds between gripper position samples during transport to put.",
    )
    parser.add_argument(
        "--gripper-server",
        default=GRIPPER_DEFAULTS.server,
        help="Remote CAN gRPC server used when --no-dual-gripper is set.",
    )
    parser.add_argument(
        "--dual-gripper",
        action=argparse.BooleanOptionalAction,
        default=GRIPPER_DEFAULTS.dual,
        help=(
            "When enabled, L/P and grasp tasks command both grippers in parallel. "
            "Left uses --grip-signal-port, right uses port+1."
        ),
    )
    parser.add_argument(
        "--left-gripper-server",
        default=GRIPPER_DEFAULTS.left_server,
        help="Remote CAN gRPC server for the left gripper.",
    )
    parser.add_argument(
        "--left-gripper-clamp-pos",
        type=int,
        default=GRIPPER_DEFAULTS.left_clamp_pos,
    )
    parser.add_argument(
        "--left-gripper-open-pos",
        type=int,
        default=GRIPPER_DEFAULTS.left_open_pos,
    )
    parser.add_argument(
        "--left-gripper-max-itinerary",
        type=int,
        default=GRIPPER_DEFAULTS.left_max_itinerary,
    )
    parser.add_argument(
        "--left-gripper-speed-coe",
        type=int,
        default=GRIPPER_DEFAULTS.left_speed_coe,
    )
    parser.add_argument(
        "--left-gripper-min-pos",
        type=int,
        default=GRIPPER_DEFAULTS.left_min_pos,
    )
    parser.add_argument(
        "--left-gripper-max-pos",
        type=int,
        default=GRIPPER_DEFAULTS.left_max_pos,
    )
    parser.add_argument(
        "--left-gripper-grip-speed",
        type=int,
        default=GRIPPER_DEFAULTS.left_grip_speed,
    )
    parser.add_argument(
        "--left-gripper-grip-torque",
        type=int,
        default=GRIPPER_DEFAULTS.left_grip_torque,
    )
    parser.add_argument(
        "--left-gripper-hold-torque",
        type=int,
        default=GRIPPER_DEFAULTS.left_hold_torque,
    )
    parser.add_argument(
        "--left-gripper-current-threshold",
        type=int,
        default=GRIPPER_DEFAULTS.left_current_threshold,
    )
    parser.add_argument(
        "--left-gripper-poll-interval",
        type=float,
        default=GRIPPER_DEFAULTS.left_poll_interval,
    )
    parser.add_argument(
        "--left-gripper-contact-grace",
        type=float,
        default=GRIPPER_DEFAULTS.left_contact_grace,
    )
    parser.add_argument(
        "--left-gripper-progress-epsilon",
        type=int,
        default=GRIPPER_DEFAULTS.left_progress_epsilon,
    )
    parser.add_argument(
        "--left-gripper-stall-samples",
        type=int,
        default=GRIPPER_DEFAULTS.left_stall_samples,
    )
    parser.add_argument(
        "--left-gripper-empty-grip-margin",
        type=int,
        default=GRIPPER_DEFAULTS.left_empty_grip_margin,
    )
    parser.add_argument(
        "--left-gripper-target-pos-tolerance",
        type=int,
        default=GRIPPER_DEFAULTS.left_target_pos_tolerance,
    )
    parser.add_argument(
        "--left-gripper-timeout",
        type=float,
        default=GRIPPER_DEFAULTS.left_timeout,
    )
    parser.add_argument(
        "--left-gripper-grip-done-wait",
        type=float,
        default=GRIPPER_DEFAULTS.left_grip_done_wait,
    )
    parser.add_argument(
        "--left-gripper-release-target",
        type=int,
        default=GRIPPER_DEFAULTS.left_release_target,
    )
    parser.add_argument(
        "--left-gripper-release-speed",
        type=int,
        default=GRIPPER_DEFAULTS.left_release_speed,
    )
    parser.add_argument(
        "--left-gripper-release-torque",
        type=int,
        default=GRIPPER_DEFAULTS.left_release_torque,
    )
    parser.add_argument(
        "--left-gripper-release-wait",
        type=float,
        default=GRIPPER_DEFAULTS.left_release_wait,
    )
    parser.add_argument(
        "--right-gripper-server",
        default=GRIPPER_DEFAULTS.right_server,
        help="Remote CAN gRPC server for the right gripper.",
    )
    parser.add_argument(
        "--gripper-clamp-pos",
        "--right-gripper-clamp-pos",
        type=int,
        default=GRIPPER_DEFAULTS.clamp_pos,
    )
    parser.add_argument(
        "--gripper-open-pos",
        "--right-gripper-open-pos",
        type=int,
        default=GRIPPER_DEFAULTS.open_pos,
    )
    parser.add_argument(
        "--gripper-max-itinerary",
        "--right-gripper-max-itinerary",
        type=int,
        default=GRIPPER_DEFAULTS.max_itinerary,
    )
    parser.add_argument(
        "--gripper-speed-coe",
        "--right-gripper-speed-coe",
        type=int,
        default=GRIPPER_DEFAULTS.speed_coe,
    )
    parser.add_argument(
        "--gripper-calibration-tolerance",
        type=int,
        default=GRIPPER_DEFAULTS.calibration_tolerance,
    )
    parser.add_argument(
        "--gripper-connect-attempts",
        type=int,
        default=GRIPPER_DEFAULTS.connect_attempts,
        help="gRPC connection attempts used by the gripper SDK receiver.",
    )
    parser.add_argument(
        "--gripper-connect-timeout-sec",
        type=float,
        default=GRIPPER_DEFAULTS.connect_timeout_sec,
        help="Seconds to wait for each gripper SDK gRPC connection attempt.",
    )
    parser.add_argument(
        "--gripper-connect-retry-delay-sec",
        type=float,
        default=GRIPPER_DEFAULTS.connect_retry_delay_sec,
        help="Delay between gripper SDK gRPC connection retries.",
    )
    parser.add_argument(
        "--gripper-allow-homing-fallback",
        action=argparse.BooleanOptionalAction,
        default=GRIPPER_DEFAULTS.allow_homing_fallback,
        help=(
            "Allow grip_signal_receiver.py to run SDK grip_init() if known "
            "calibration init fails. This performs a homing motion."
        ),
    )
    parser.add_argument(
        "--gripper-min-pos",
        "--right-gripper-min-pos",
        type=int,
        default=GRIPPER_DEFAULTS.min_pos,
    )
    parser.add_argument(
        "--gripper-max-pos",
        "--right-gripper-max-pos",
        type=int,
        default=GRIPPER_DEFAULTS.max_pos,
    )
    parser.add_argument(
        "--gripper-grip-speed",
        "--right-gripper-grip-speed",
        type=int,
        default=GRIPPER_DEFAULTS.grip_speed,
    )
    parser.add_argument(
        "--gripper-grip-torque",
        "--right-gripper-grip-torque",
        type=int,
        default=GRIPPER_DEFAULTS.grip_torque,
    )
    parser.add_argument(
        "--gripper-hold-torque",
        "--right-gripper-hold-torque",
        type=int,
        default=GRIPPER_DEFAULTS.hold_torque,
    )
    parser.add_argument(
        "--gripper-current-threshold",
        "--right-gripper-current-threshold",
        type=int,
        default=GRIPPER_DEFAULTS.current_threshold,
    )
    parser.add_argument(
        "--gripper-poll-interval",
        "--right-gripper-poll-interval",
        type=float,
        default=GRIPPER_DEFAULTS.poll_interval,
    )
    parser.add_argument(
        "--gripper-contact-grace",
        "--right-gripper-contact-grace",
        type=float,
        default=GRIPPER_DEFAULTS.contact_grace,
    )
    parser.add_argument(
        "--gripper-progress-epsilon",
        "--right-gripper-progress-epsilon",
        type=int,
        default=GRIPPER_DEFAULTS.progress_epsilon,
    )
    parser.add_argument(
        "--gripper-stall-samples",
        "--right-gripper-stall-samples",
        type=int,
        default=GRIPPER_DEFAULTS.stall_samples,
    )
    parser.add_argument(
        "--gripper-empty-grip-margin",
        "--right-gripper-empty-grip-margin",
        type=int,
        default=GRIPPER_DEFAULTS.empty_grip_margin,
        help=(
            "If grip stalls at or below gripper-min-pos plus this margin, treat it "
            "as an empty/min-limit grasp failure instead of confirmed contact."
        ),
    )
    parser.add_argument(
        "--gripper-target-pos-tolerance",
        "--right-gripper-target-pos-tolerance",
        type=int,
        default=GRIPPER_DEFAULTS.target_pos_tolerance,
        help=(
            "Warn when the measured gripper SDK position is farther than this "
            "from the commanded target; useful for detecting bad known calibration."
        ),
    )
    parser.add_argument(
        "--gripper-timeout",
        "--right-gripper-timeout",
        type=float,
        default=GRIPPER_DEFAULTS.timeout,
    )
    parser.add_argument(
        "--gripper-grip-done-wait",
        "--right-gripper-grip-done-wait",
        type=float,
        default=GRIPPER_DEFAULTS.grip_done_wait,
        help="Seconds the gripper receiver waits after hold torque before reporting grip done.",
    )
    parser.add_argument(
        "--gripper-release-target",
        "--right-gripper-release-target",
        type=int,
        default=GRIPPER_DEFAULTS.release_target,
    )
    parser.add_argument(
        "--gripper-release-speed",
        "--right-gripper-release-speed",
        type=int,
        default=GRIPPER_DEFAULTS.release_speed,
    )
    parser.add_argument(
        "--gripper-release-torque",
        "--right-gripper-release-torque",
        type=int,
        default=GRIPPER_DEFAULTS.release_torque,
    )
    parser.add_argument(
        "--gripper-release-wait",
        "--right-gripper-release-wait",
        type=float,
        default=GRIPPER_DEFAULTS.release_wait,
    )
    parser.add_argument(
        "--grip-settle-sec",
        type=float,
        default=GRIP_SIGNAL_DEFAULTS.settle_sec,
        help="Minimum seconds to hold the lowest pick waypoint before sending grip.",
    )
    parser.add_argument(
        "--grip-post-confirm-hold-sec",
        type=float,
        default=GRIP_SIGNAL_DEFAULTS.post_confirm_hold_sec,
        help=(
            "Seconds to hold the grasp pose after the gripper confirms success. "
            "Keep this low to start the put motion immediately after contact."
        ),
    )
    parser.add_argument(
        "--grip-lift-hold-sec",
        type=float,
        default=GRIP_SIGNAL_DEFAULTS.lift_hold_sec,
        help=(
            "Seconds to hold after the post-grip lift waypoint before auto put. "
            "This only affects the intermediate lift after a confirmed grasp."
        ),
    )
    parser.add_argument(
        "--grip-retry-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry capture/SAM3/FlowPose/grasp when grip reaches min limit.",
    )
    parser.add_argument(
        "--grip-retry-max-attempts",
        type=int,
        default=GRIP_SIGNAL_DEFAULTS.retry_max_attempts,
        help="Maximum automatic retries after min-limit grip failure; 0 means unlimited.",
    )
    parser.add_argument(
        "--enable-put-after-grasp",
        type=parse_bool,
        default=True,
        metavar="TRUE/FALSE",
        help=(
            "TRUE automatically runs the fixed put action after a "
            "gripper-confirmed successful grasp; FALSE disables auto put."
        ),
    )
    parser.add_argument(
        "--continuous-grasp-after-put",
        type=parse_bool,
        default=True,
        metavar="TRUE/FALSE",
        help=(
            "TRUE starts the same capture -> SAM3 -> FlowPose -> grasp workflow "
            "as the A key after every successful put (default: TRUE); "
            "FALSE stops after put."
        ),
    )
    parser.add_argument(
        "--put-target-hold-sec",
        type=float,
        default=PUT_DEFAULTS.target_hold_sec,
        help="Seconds to hold the put target before opening the gripper.",
    )
    parser.add_argument(
        "--put-home-hold-sec",
        type=float,
        default=PUT_DEFAULTS.home_hold_sec,
        help="Seconds to hold the home target after put release.",
    )
    parser.add_argument(
        "--put-keep-pose",
        type=parse_bool,
        default=PUT_DEFAULTS.keep_put_pose,
        metavar="TRUE/FALSE",
        help=(
            "TRUE keeps the arm at the released put pose and starts the next "
            "A-key grasp from there; FALSE returns the arm home after release."
        ),
    )
    return apply_grasp_config_defaults(parser.parse_args())
