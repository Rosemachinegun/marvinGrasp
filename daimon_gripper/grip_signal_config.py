from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .grip_signal_utils import clamp
except ImportError:
    from grip_signal_utils import clamp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = "192.168.14.10:55551"
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
CALIBRATION_CACHE_PATH = PROJECT_ROOT / "daimon_gripper" / ".gripper_calibration_cache.json"
ACTION_COMMANDS = {"grip", "release", "check", "feedback"}
COMMAND_ERROR = "ERR command must be 'grip', 'release', 'check', 'feedback', or 'status'"
CALIBRATION_FIELDS = ("clamp_pos", "open_pos", "max_itinerary", "speed_coe")

GRIPPER_ARG_SPECS = (
    ("--server", {"default": DEFAULT_SERVER}),
    ("--clamp-pos", {"type": int, "default": DEFAULT_CLAMP_POS}),
    ("--open-pos", {"type": int, "default": DEFAULT_OPEN_POS}),
    ("--max-itinerary", {"type": int, "default": DEFAULT_MAX_ITINERARY}),
    ("--speed-coe", {"type": int, "default": DEFAULT_SPEED_COE}),
    ("--calibration-tolerance", {"type": int, "default": DEFAULT_CALIBRATION_TOLERANCE}),
    ("--connect-attempts", {"type": int, "default": DEFAULT_CONNECT_ATTEMPTS}),
    ("--connect-timeout-sec", {"type": float, "default": DEFAULT_CONNECT_TIMEOUT_SEC}),
    (
        "--connect-retry-delay-sec",
        {"type": float, "default": DEFAULT_CONNECT_RETRY_DELAY_SEC},
    ),
    ("--calibration-speed", {"type": int, "default": 100}),
    ("--calibration-torque", {"type": int, "default": 50}),
    ("--min-pos", {"type": int, "default": 10}),
    ("--max-pos", {"type": int, "default": 1000}),
    ("--grip-speed", {"type": int, "default": 80}),
    ("--grip-torque", {"type": int, "default": 30}),
    ("--hold-torque", {"type": int, "default": 10}),
    ("--current-threshold", {"type": int, "default": 20}),
    ("--contact-confirm-samples", {"type": int, "default": 3}),
    ("--poll-interval", {"type": float, "default": 0.01}),
    ("--contact-grace", {"type": float, "default": 0.2}),
    ("--progress-epsilon", {"type": int, "default": 2}),
    ("--stall-samples", {"type": int, "default": 5}),
    ("--empty-limit-confirm-samples", {"type": int, "default": 5}),
    ("--release-target", {"type": int, "default": 1000}),
    ("--release-speed", {"type": int, "default": 40}),
    ("--release-torque", {"type": int, "default": 20}),
    ("--release-wait", {"type": float, "default": 0.5}),
)


def add_gripper_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hand",
        choices=("right", "left"),
        default="right",
        help="Select independent right/left gripper grasp parameters.",
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
    parser.add_argument(
        "--calibrate-each-command",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run SDK grip_init() before each direct grip/release/check/feedback command.",
    )
    parser.add_argument(
        "--stdin-control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable typing grip/release/check/feedback/status into stdin while serving.",
    )
    for name, kwargs in GRIPPER_ARG_SPECS:
        parser.add_argument(name, **kwargs)
    parser.add_argument(
        "--empty-grip-margin",
        type=int,
        default=10,
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


def normalize_args(args: argparse.Namespace) -> None:
    for name, low, high in (
        ("min_pos", 0, 1000),
        ("max_pos", 0, 1000),
        ("release_target", 0, 1000),
        ("grip_speed", 10, 100),
        ("release_speed", 10, 100),
        ("calibration_speed", 10, 100),
        ("grip_torque", 10, 100),
        ("release_torque", 10, 100),
        ("hold_torque", 10, 100),
        ("calibration_torque", 10, 100),
    ):
        setattr(args, name, int(clamp(getattr(args, name), low, high)))
    for name, minimum in (
        ("progress_epsilon", 0),
        ("stall_samples", 1),
        ("contact_confirm_samples", 1),
        ("empty_limit_confirm_samples", 1),
        ("empty_grip_margin", 0),
        ("target_pos_tolerance", 0),
        ("calibration_tolerance", 0),
        ("connect_attempts", 1),
    ):
        setattr(args, name, max(int(getattr(args, name)), minimum))
    for name, minimum in (
        ("poll_interval", 0.01),
        ("contact_grace", 0.0),
        ("timeout", 0.1),
        ("grip_done_wait", 0.0),
        ("release_wait", 0.0),
        ("connect_timeout_sec", 0.1),
        ("connect_retry_delay_sec", 0.0),
    ):
        setattr(args, name, max(float(getattr(args, name)), minimum))
    if args.min_pos > args.max_pos:
        raise SystemExit("--min-pos must be less than or equal to --max-pos")


def gripper_config_summary(args: argparse.Namespace) -> str:
    homing_fallback_effective = (
        bool(args.allow_homing_fallback)
        and getattr(args, CALIBRATION_SOURCE_ATTR, "") != "cache"
    )
    values = {
        "hand": args.hand,
        "server": args.server,
        "clamp_pos": args.clamp_pos,
        "open_pos": args.open_pos,
        "max_itinerary": args.max_itinerary,
        "speed_coe": args.speed_coe,
        "min_pos": args.min_pos,
        "max_pos": args.max_pos,
        "empty_grip_margin": args.empty_grip_margin,
        "empty_limit": args.min_pos + args.empty_grip_margin,
        "target_pos_tolerance": args.target_pos_tolerance,
        "grip_done_wait": f"{args.grip_done_wait:.2f}",
        "release_target": args.release_target,
        "connect_attempts": args.connect_attempts,
        "calibration_speed": args.calibration_speed,
        "calibration_torque": args.calibration_torque,
        "connect_timeout_sec": f"{args.connect_timeout_sec:.2f}",
        "connect_retry_delay_sec": f"{args.connect_retry_delay_sec:.2f}",
        "allow_homing_fallback": bool(args.allow_homing_fallback),
        "calibrate_each_command": bool(args.calibrate_each_command),
        "stdin_control": bool(args.stdin_control),
        "homing_fallback_effective": homing_fallback_effective,
        "calibration_source": getattr(args, CALIBRATION_SOURCE_ATTR, "default"),
    }
    return " ".join(f"{name}={value}" for name, value in values.items())
