#!/usr/bin/env python3
"""核心几何工具：位姿、四元数、坐标方向和抓取姿态基础变换函数。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np

from grasp_core.core.robot_target_pose import TargetObjectPose, matrix_to_quaternion

PoseWaypoint = tuple[np.ndarray, tuple[float, float, float, float]]
PickTemplateWaypoint = tuple[np.ndarray, tuple[float, float, float, float], float]

def ik_orientation_quat(args: argparse.Namespace) -> tuple[float, float, float, float]:
    return normalize_quaternion(
        tuple(float(value) for value in args.ik_orientation_quat)
    )


def ik_orientation_rotation(args: argparse.Namespace) -> np.ndarray:
    return quaternion_to_rotation_matrix(ik_orientation_quat(args))


def ik_wrist_orientation_quat(
    args: argparse.Namespace,
    hand: str | None = None,
) -> tuple[float, float, float, float]:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = ik_orientation_rotation(args)
    pose = apply_downward_end_effector_tilt(pose, args, hand=hand)
    return normalize_quaternion(matrix_to_quaternion(pose))


def ik_home_wrist_orientation_quat(
    args: argparse.Namespace,
    hand: str | None = None,
) -> tuple[float, float, float, float]:
    """Build an independent Home wrist pose; Z/Y default to the neutral 0°."""
    hand_name = "left" if str(hand).strip().lower() == "left" else "right"
    home_z_deg = float(getattr(args, f"home_tilt_z_{hand_name}_deg", 0.0))
    home_y_deg = float(getattr(args, f"home_tilt_y_{hand_name}_deg", 0.0))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = ik_orientation_rotation(args)
    home_rotation = rotation_matrix_from_zyx_euler_deg(
        pitch_deg=home_y_deg,
        yaw_deg=home_z_deg,
    )
    if getattr(args, "ik_downward_tilt_frame", "local") == "base":
        pose[:3, :3] = home_rotation @ pose[:3, :3]
    else:
        pose[:3, :3] = pose[:3, :3] @ home_rotation
    return normalize_quaternion(matrix_to_quaternion(pose))


def home_position_for_hand(hand: str, args: argparse.Namespace) -> np.ndarray:
    values = args.left_home_xyz if hand == "left" else args.right_home_xyz
    return checked_position(np.asarray(values, dtype=np.float64))


def build_grasp_pose(T_base_object: np.ndarray, object_type: str) -> np.ndarray:
    T_object_gripper = get_relative_grasp_template(object_type)
    if T_object_gripper is None:
        raise ValueError(
            f"no grasp template configured for object_type={object_type!r}"
        )
    return np.asarray(T_base_object, dtype=np.float64) @ T_object_gripper


def get_relative_grasp_template(object_type: str) -> np.ndarray | None:
    name = normalize_object_type(object_type)
    if name in {
        "blue_block",
        "blue_cube",
        "block",
        "cube",
        "box",
        "cuboid",
    }:
        return make_relative_grasp_template()
    if name in {
        "pen",
        "pencil",
        "screwdriver",
        "knife",
        "rod",
        "stick",
        "bar",
        "long_object",
    }:
        # FlowPose long-object templates currently assume x_obj is the long axis.
        # The gripper pose follows the object frame; tune this matrix if your
        # gripper's closing/approach axes use a different convention.
        return make_relative_grasp_template()
    return None


def make_relative_grasp_template(
    xyz_obj: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_obj_gripper: np.ndarray | None = None,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(xyz_obj, dtype=np.float64)
    if rotation_obj_gripper is not None:
        transform[:3, :3] = np.asarray(rotation_obj_gripper, dtype=np.float64)
    return transform


def make_fixed_front_gripper_target_pose(
    target: TargetObjectPose,
    args: argparse.Namespace,
) -> np.ndarray:
    object_pose = np.eye(4, dtype=np.float64)
    object_pose[:3, 3] = target.base_xyz
    object_pose[:3, :3] = ik_orientation_rotation(args)
    grasp_pose = object_pose.copy()
    grasp_pose[:3, 3] = target.base_xyz + np.asarray(
        args.ik_grasp_tcp_offset_m,
        dtype=np.float64,
    )
    if args.ik_target_stage == "grasp":
        return grasp_pose

    pregrasp_pose = grasp_pose.copy()
    pregrasp_pose[:3, 3] = (
        grasp_pose[:3, 3]
        - approach_direction(object_pose, args.approach_axis, args.approach_sign)
        * max(float(args.pregrasp_distance_m), 0.0)
        + np.asarray(args.ik_pregrasp_extra_offset_m, dtype=np.float64)
    )
    return pregrasp_pose


def apply_pregrasp_offset(
    grasp_pose: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    pregrasp_pose = np.asarray(grasp_pose, dtype=np.float64).copy()
    pregrasp_pose[:3, 3] = (
        pregrasp_pose[:3, 3]
        - approach_direction(grasp_pose, args.approach_axis, args.approach_sign)
        * max(float(args.pregrasp_distance_m), 0.0)
        + np.asarray(args.ik_pregrasp_extra_offset_m, dtype=np.float64)
    )
    return pregrasp_pose


def apply_grasp_tcp_offset(
    pose: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    offset_pose = np.asarray(pose, dtype=np.float64).copy()
    offset_pose[:3, 3] += np.asarray(args.ik_grasp_tcp_offset_m, dtype=np.float64)
    return offset_pose


def apply_grasp_rotation_mode(
    pose: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    rotation_pose = np.asarray(pose, dtype=np.float64).copy()
    if bool(args.use_flowpose_grasp_rotation):
        return rotation_pose
    rotation_pose[:3, :3] = ik_orientation_rotation(args)
    return rotation_pose


def apply_downward_end_effector_tilt(
    pose: np.ndarray,
    args: argparse.Namespace,
    hand: str | None = None,
) -> np.ndarray:
    """Apply a tunable downward tilt to the end-effector orientation only."""
    tilt_deg = ik_downward_tilt_deg_for_hand(args, hand)
    tilt_y_deg = ik_downward_tilt_y_deg_for_hand(args, hand)
    if abs(tilt_deg) < 1e-8 and abs(tilt_y_deg) < 1e-8:
        return np.asarray(pose, dtype=np.float64).copy()

    tilted_pose = np.asarray(pose, dtype=np.float64).copy()
    tilt_rotation = make_downward_tilt_rotation(
        tilt_deg,
        axis=str(getattr(args, "ik_downward_tilt_axis", "y")),
        y_deg=tilt_y_deg,
    )
    if getattr(args, "ik_downward_tilt_frame", "local") == "base":
        tilted_pose[:3, :3] = tilt_rotation @ tilted_pose[:3, :3]
    else:
        tilted_pose[:3, :3] = tilted_pose[:3, :3] @ tilt_rotation
    return tilted_pose


def ik_downward_tilt_deg_for_hand(
    args: argparse.Namespace,
    hand: str | None,
) -> float:
    hand_name = str(hand or "").strip().lower()
    if hand_name in {"left", "right"}:
        hand_value = getattr(args, f"ik_downward_tilt_{hand_name}_deg", None)
        if hand_value is not None:
            return float(hand_value)
    return float(getattr(args, "ik_downward_tilt_deg", 0.0) or 0.0)


def ik_downward_tilt_y_deg_for_hand(
    args: argparse.Namespace,
    hand: str | None,
) -> float:
    hand_name = str(hand or "").strip().lower()
    if hand_name in {"left", "right"}:
        hand_value = getattr(args, f"ik_downward_tilt_y_{hand_name}_deg", None)
        if hand_value is not None:
            return float(hand_value)
    return float(getattr(args, "ik_downward_tilt_y_deg", 0.0) or 0.0)


def make_downward_tilt_rotation(
    tilt_deg: float,
    *,
    axis: str = "y",
    y_deg: float = 0.0,
) -> np.ndarray:
    """Return a ZYX Euler rotation matrix for the adjustable down-tilt."""
    roll_deg = pitch_deg = yaw_deg = 0.0
    axis_name = axis.lower()
    if axis_name == "x":
        roll_deg = float(tilt_deg)
    elif axis_name == "z":
        yaw_deg = float(tilt_deg)
    else:
        pitch_deg = float(tilt_deg)
    pitch_deg += float(y_deg)
    return rotation_matrix_from_zyx_euler_deg(
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
    )


def rotation_matrix_from_zyx_euler_deg(
    *,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll), using degrees."""
    roll = np.deg2rad(float(roll_deg))
    pitch = np.deg2rad(float(pitch_deg))
    yaw = np.deg2rad(float(yaw_deg))

    cr, sr = float(np.cos(roll)), float(np.sin(roll))
    cp, sp = float(np.cos(pitch)), float(np.sin(pitch))
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))

    rotation_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )
    rotation_y = np.asarray(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )
    rotation_z = np.asarray(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


def validate_pose_matrix(pose: np.ndarray) -> str | None:
    matrix = np.asarray(pose, dtype=np.float64)
    if matrix.shape != (4, 4):
        return f"pose shape is {matrix.shape}, expected (4, 4)"
    if not np.all(np.isfinite(matrix)):
        return "pose contains non-finite values"
    if not np.allclose(matrix[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
        return f"pose bottom row is invalid: {matrix[3, :].tolist()}"

    rotation = matrix[:3, :3]
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 5e-2:
        return f"rotation is not orthonormal, error={orthogonality_error:.4f}"
    if abs(determinant - 1.0) > 5e-2:
        return f"rotation determinant={determinant:.4f}, expected 1.0"
    return None


def normalize_object_type(object_type: str) -> str:
    text = str(object_type).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.rstrip("_0123456789") or "object"


def quaternion_to_rotation_matrix(
    quat_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quat_xyzw)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_from_position_quaternion(
    position_xyz: np.ndarray,
    orientation_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = checked_position(position_xyz)
    pose[:3, :3] = quaternion_to_rotation_matrix(orientation_xyzw)
    return pose


def log_grasp_pose_plan(
    target: TargetObjectPose,
    gripper_pose: np.ndarray,
    template: np.ndarray | None,
    fallback_reason: str | None,
) -> None:
    object_quat = safe_matrix_to_quaternion(target.base_pose)
    gripper_quat = safe_matrix_to_quaternion(gripper_pose)
    print(
        "[grasp] T_base_object "
        f"object_type={target.label!r} frame={target.frame_id} "
        f"xyz={format_xyz(target.base_xyz)} "
        f"quat_xyzw={format_quat(object_quat)}",
        flush=True,
    )
    if template is not None:
        print(
            "[grasp] T_object_gripper template "
            f"xyz={format_xyz(template[:3, 3])} "
            f"quat_xyzw={format_quat(safe_matrix_to_quaternion(template))} "
            f"matrix={np.array2string(template, precision=4, suppress_small=True)}",
            flush=True,
        )
    print(
        "[grasp] T_base_gripper "
        f"xyz={format_xyz(gripper_pose[:3, 3])} "
        f"quat_xyzw={format_quat(gripper_quat)} "
        f"fallback={bool(fallback_reason)}"
        f"{f' reason={fallback_reason}' if fallback_reason else ''}",
        flush=True,
    )


def format_xyz(values: np.ndarray) -> str:
    return "(" + ", ".join(f"{float(value):.4f}" for value in values[:3]) + ")"


def safe_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    try:
        quat = matrix_to_quaternion(matrix)
    except Exception:
        return 0.0, 0.0, 0.0, 1.0
    if not np.all(np.isfinite(np.asarray(quat, dtype=np.float64))):
        return 0.0, 0.0, 0.0, 1.0
    return quat


def format_quat(values: tuple[float, float, float, float]) -> str:
    return "(" + ", ".join(f"{float(value):.5f}" for value in values) + ")"


def approach_direction(
    pose_base: np.ndarray,
    approach_axis: str,
    approach_sign: float,
) -> np.ndarray:
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    index = axis_to_index.get(approach_axis.lower(), 2)
    direction = pose_base[:3, index] * float(approach_sign)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return direction / norm


def select_ik_hand(position_xyz: np.ndarray, hand_mode: str) -> str:
    if hand_mode in {"left", "right"}:
        return hand_mode
    return "left" if float(position_xyz[1]) > 0.0 else "right"


def normalize_quaternion(
    quat_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    quat /= norm
    return tuple(float(value) for value in quat)


def checked_position(position_xyz: np.ndarray) -> np.ndarray:
    position = np.asarray(position_xyz, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"target position must have 3 values, got {position.shape}")
    if not np.all(np.isfinite(position)):
        raise ValueError(f"target position contains non-finite values: {position}")
    return position


def smoothstep(alpha: float) -> float:
    t = float(np.clip(alpha, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def quaternion_angle_rad(
    start_xyzw: tuple[float, float, float, float],
    end_xyzw: tuple[float, float, float, float],
) -> float:
    q0 = np.asarray(normalize_quaternion(start_xyzw), dtype=np.float64)
    q1 = np.asarray(normalize_quaternion(end_xyzw), dtype=np.float64)
    dot = abs(float(np.dot(q0, q1)))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def path_segment_steps(
    start_position: np.ndarray,
    start_orientation: tuple[float, float, float, float],
    end_position: np.ndarray,
    end_orientation: tuple[float, float, float, float],
    *,
    max_step_m: float,
    max_step_deg: float,
    min_steps: int = 1,
) -> int:
    distance_m = float(
        np.linalg.norm(
            checked_position(end_position) - checked_position(start_position)
        )
    )
    angle_rad = quaternion_angle_rad(start_orientation, end_orientation)
    if distance_m < 1e-6 and angle_rad < np.deg2rad(0.01):
        return 1
    linear_steps = int(np.ceil(distance_m / max(float(max_step_m), 1e-4)))
    angular_steps = int(np.ceil(angle_rad / np.deg2rad(max(float(max_step_deg), 0.1))))
    return max(linear_steps, angular_steps, int(min_steps), 1)


def slerp_quaternion(
    start_xyzw: tuple[float, float, float, float],
    end_xyzw: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    q0 = np.asarray(normalize_quaternion(start_xyzw), dtype=np.float64)
    q1 = np.asarray(normalize_quaternion(end_xyzw), dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    t = float(np.clip(alpha, 0.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion(tuple((q0 + t * (q1 - q0)).tolist()))

    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * t
    sin_theta = float(np.sin(theta))
    scale_0 = float(np.cos(theta) - dot * sin_theta / sin_theta_0)
    scale_1 = sin_theta / sin_theta_0
    return normalize_quaternion(tuple((scale_0 * q0 + scale_1 * q1).tolist()))
