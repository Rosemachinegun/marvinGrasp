#!/usr/bin/env python3
"""配置层：集中管理运行参数和 grasp_core/resources/tool.yaml 默认配置。"""

from __future__ import annotations

from pathlib import Path

import argparse
from dataclasses import dataclass

import numpy as np
from grasp_core.config.parsing import (
    parse_bool,
)
from grasp_core.config.resource_paths import (
    DEFAULT_ROBOT_XACRO_PATH,
    DEFAULT_SAM3_ROI_XYXY,
    DEFAULT_TOOL_TEMPLATE_PATH,
    PROJECT_ROOT,
)
from grasp_core.execution.config import (
    DEFAULT_JOINT_TRAJECTORY_CSV_DIR,
    DEFAULT_TARGET_PUBLISH_RATE_HZ,
    DEFAULT_TARGET_TRAJECTORY_ANGULAR_SPEED_DPS,
    DEFAULT_TARGET_TRAJECTORY_MIN_STEPS,
    DEFAULT_TARGET_TRAJECTORY_SPEED_MPS,
    DEFAULT_TARGET_TRAJECTORY_STEP_DEG,
    DEFAULT_TARGET_TRAJECTORY_STEP_M,
    MOTION_CONFIG,
)
from grasp_core.config.gripper_config import (
    GripSignalDefaults,
    GripperDefaults,
    GRIP_SIGNAL_DEFAULTS,
    GRIPPER_DEFAULTS,
)

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


DEFAULT_GRIP_SETTLE_SEC = GRIP_SIGNAL_DEFAULTS.settle_sec
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
    args.left_gripper_poll_interval = max(float(args.left_gripper_poll_interval), 0.01)
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
    args.gripper_poll_interval = max(float(args.gripper_poll_interval), 0.01)
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
    args.gripper_calibration_tolerance = max(
        int(args.gripper_calibration_tolerance),
        0,
    )
    args.gripper_allow_homing_fallback = bool(args.gripper_allow_homing_fallback)
    args.dual_gripper = bool(args.dual_gripper)
    args.left_gripper_server = str(args.left_gripper_server)
    args.right_gripper_server = str(args.right_gripper_server)
    args.grip_signal_port = int(args.grip_signal_port)
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
    parser.add_argument("--prompts", default="toy, yellow_screwdriver_handle,pen,rectangular object,ribbon")
    parser.add_argument(
        "--sam3-checkpoint-path",
        default=str(PROJECT_ROOT / "perception" / "models" / "sam3.pt"),
    )
    parser.add_argument("--sam3-root", default=None)
    parser.add_argument(
        "--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD
    )
    parser.add_argument(
        "--target-order-cluster-eps-m",
        type=float,
        default=0.12,
        help="DBSCAN neighborhood radius for target centers in meters.",
    )
    parser.add_argument(
        "--target-order-cluster-use-z",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include base_link Z when running target DBSCAN.",
    )
    parser.add_argument(
        "--target-order-singleton-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prioritize DBSCAN clusters containing exactly one target.",
    )
    parser.add_argument(
        "--target-order-distance-axis",
        choices=("xy", "xyz"),
        default="xy",
        help="Axes used for nearest/outermost target distance calculations.",
    )
    parser.add_argument(
        "--target-order-max-volume-m3",
        type=float,
        default=0.0005,
        help="Discard targets whose estimated volume exceeds this value.",
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
            "Defaults to the configured camera workspace ROI."
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
        "--target-publish-rate-hz",
        type=float,
        default=DEFAULT_TARGET_PUBLISH_RATE_HZ,
        help="Rate used for request_ik_tester target publishing and trajectory steps.",
    )
    parser.add_argument(
        "--target-publish-sec",
        type=float,
        default=MOTION_CONFIG.publish_sec,
        help="Seconds to keep publishing the final target after a trajectory finishes.",
    )
    parser.add_argument(
        "--target-smooth-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Deprecated compatibility option; Cartesian robot motion always uses "
            "the unified Bezier trajectory planner."
        ),
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
        default=DEFAULT_TARGET_TRAJECTORY_MIN_STEPS,
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
        default=str(MOTION_CONFIG.plot_dir),
        help="Directory where grasp path visualization PNG files are saved.",
    )
    parser.add_argument(
        "--joint-trajectory-csv-dir",
        default=str(DEFAULT_JOINT_TRAJECTORY_CSV_DIR),
        help="Directory where timestamped joint trajectory CSV files are saved.",
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
        help="Use grasp_core/resources/tool.yaml pick waypoints for visual grasp targets when available.",
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
        help="Start daimon_gripper/grip_signal_receiver.py together with this demo.",
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
        help="Seconds between gripper position samples during transport to place.",
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
            "Keep this low to start the place motion immediately after contact."
        ),
    )
    parser.add_argument(
        "--grip-lift-hold-sec",
        type=float,
        default=GRIP_SIGNAL_DEFAULTS.lift_hold_sec,
        help=(
            "Seconds to hold after the post-grip lift waypoint before auto place. "
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
        help=(
            "Deprecated compatibility option; failed-grasp retries are now "
            "unlimited. Use --no-grip-retry-loop to stop automatic retries."
        ),
    )
    parser.add_argument(
        "--enable-place-after-grasp",
        type=parse_bool,
        default=True,
        metavar="TRUE/FALSE",
        help=(
            "TRUE automatically runs the fixed place action after a "
            "gripper-confirmed successful grasp; FALSE disables auto place."
        ),
    )
    parser.add_argument(
        "--continuous-grasp-after-place",
        type=parse_bool,
        default=True,
        metavar="TRUE/FALSE",
        help=(
            "TRUE starts the same capture -> SAM3 -> FlowPose -> grasp workflow "
            "as the A key after every successful place and after completed grip-"
            "failure recovery (default: TRUE); FALSE stops automatic continuation."
        ),
    )
    return normalize_gripper_args(parser.parse_args())
