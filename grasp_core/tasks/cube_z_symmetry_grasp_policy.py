#!/usr/bin/env python3
"""Cube local-Z symmetry policy used between FlowPose and IK."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grasp_core.core.pose_math import PickTemplateWaypoint
from grasp_core.core.robot_target_pose import TargetObjectPose
from grasp_core.tasks.screwdriver_handle_grasp_policy import (
    is_screwdriver_handle_object,
)

LEFT_GRASP_SECTOR_DEG = (60.0, 150.0)
RIGHT_GRASP_SECTOR_DEG = (210.0, 300.0)
LEFT_FALLBACK_ANGLE_DEG = 105.0
RIGHT_FALLBACK_ANGLE_DEG = 255.0
HAND_SIDE_EPS = 1e-8
WORLD_Z_AXIS = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class CubeZSymmetryCandidate:
    """One FlowPose-equivalent cube pose rotated about the object's local Z axis."""

    name: str
    pose: np.ndarray
    angle_deg: float


@dataclass(frozen=True)
class CubeZSymmetrySelection:
    """Selected cube pose and metadata for logging/debugging."""

    target: TargetObjectPose
    candidate: CubeZSymmetryCandidate
    candidates: tuple[CubeZSymmetryCandidate, ...]
    raw_candidate: CubeZSymmetryCandidate
    desired_y_sign: float


def apply_cube_z_symmetry_grasp_policy(
    target: TargetObjectPose,
    *,
    hand: str,
    args,
    relative_pick_waypoints: list[PickTemplateWaypoint] | None = None,
) -> CubeZSymmetrySelection | None:
    """Select a FlowPose-equivalent pose for the active grasp policy.

    The pose is made Z-up first.  Generic objects use the local -X gripper
    facing rule. Long objects bypass this selection: their perception frame
    is normalized separately and the dedicated policy aligns the closing axis.
    """

    del relative_pick_waypoints
    if not bool(getattr(args, "use_cube_z_symmetry_grasp_policy", False)):
        return None
    if is_screwdriver_handle_object(target.label):
        return None
    object_pose = np.asarray(target.base_pose, dtype=np.float64)
    if object_pose.shape != (4, 4) or not np.all(np.isfinite(object_pose)):
        return None
    object_pose = pose_with_world_z_axis(object_pose)

    candidates = tuple(make_candidates(object_pose))
    raw = candidates[0]
    interval_hand = hand_interval_for_y(float(object_pose[1, 3])) or hand
    desired_y_sign = desired_side_sign(interval_hand)
    if desired_y_sign == 0.0:
        return None

    sector = grasp_sector_for_hand(interval_hand)
    fallback_angle_deg = fallback_angle_for_hand(interval_hand)
    if sector is None or fallback_angle_deg is None:
        return None
    best = select_best_candidate(
        candidates,
        sector,
        fallback_angle_deg,
        desired_y_sign,
    )
    adjusted_target = TargetObjectPose(
        label=target.label,
        frame_id=target.frame_id,
        camera_pose=target.camera_pose,
        base_pose=best.pose,
        size=target.size.copy() if isinstance(target.size, np.ndarray) else target.size,
        score=target.score,
    )
    return CubeZSymmetrySelection(
        target=adjusted_target,
        candidate=best,
        candidates=candidates,
        raw_candidate=raw,
        desired_y_sign=desired_y_sign,
    )


def pose_with_world_z_axis(object_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(object_pose, dtype=np.float64).copy()
    x_axis = horizontal_unit_vector(pose[:3, 0])
    if x_axis is None:
        y_axis = horizontal_unit_vector(pose[:3, 1])
        if y_axis is None:
            x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            y_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            x_axis = normalized(np.cross(y_axis, WORLD_Z_AXIS))
    else:
        y_axis = normalized(np.cross(WORLD_Z_AXIS, x_axis))
    x_axis = normalized(np.cross(y_axis, WORLD_Z_AXIS))
    pose[:3, :3] = np.column_stack((x_axis, y_axis, WORLD_Z_AXIS))
    return pose


def horizontal_unit_vector(vector: np.ndarray) -> np.ndarray | None:
    projected = np.asarray(vector, dtype=np.float64).copy()
    projected[2] = 0.0
    norm = float(np.linalg.norm(projected))
    if norm < 1e-8:
        return None
    return projected / norm


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("cannot normalize near-zero vector")
    return np.asarray(vector, dtype=np.float64) / norm


def make_candidates(object_pose: np.ndarray) -> list[CubeZSymmetryCandidate]:
    turns = (
        ("raw", 0.0),
        ("rz+90", 90.0),
        ("rz180", 180.0),
        ("rz-90", -90.0),
    )
    candidates: list[CubeZSymmetryCandidate] = []
    for name, angle_deg in turns:
        candidate_pose = object_pose.copy()
        candidate_pose[:3, :3] = (
            object_pose[:3, :3] @ local_z_rotation_deg(angle_deg)
        )
        candidates.append(
            CubeZSymmetryCandidate(
                name=name,
                pose=candidate_pose,
                angle_deg=angle_deg,
            )
        )
    return candidates


def select_best_candidate(
    candidates: tuple[CubeZSymmetryCandidate, ...],
    sector: tuple[float, float],
    fallback_angle_deg: float,
    desired_y_sign: float,
) -> CubeZSymmetryCandidate:
    same_side_candidates = tuple(
        candidate
        for candidate in candidates
        if is_in_hand_y_interval(candidate.pose, desired_y_sign)
    )
    candidates_to_check = same_side_candidates or candidates
    for candidate in candidates_to_check:
        if local_minus_x_faces_gripper(candidate.pose, sector):
            return candidate
    return max(
        candidates_to_check,
        key=lambda candidate: alignment_score(candidate.pose, fallback_angle_deg),
    )


def local_minus_x_faces_gripper(
    object_pose: np.ndarray,
    sector: tuple[float, float],
) -> bool:
    angle_deg = local_minus_x_xy_angle_deg(object_pose)
    if angle_deg is None:
        return False
    start_deg, end_deg = sector
    return start_deg <= angle_deg <= end_deg


def alignment_score(object_pose: np.ndarray, fallback_angle_deg: float) -> float:
    local_minus_x = local_minus_x_base(object_pose)[:2]
    norm = float(np.linalg.norm(local_minus_x))
    if norm < 1e-8:
        return -1.0
    direction = local_minus_x / norm
    angle_rad = np.deg2rad(float(fallback_angle_deg))
    desired = np.asarray(
        [float(np.cos(angle_rad)), float(np.sin(angle_rad))],
        dtype=np.float64,
    )
    return float(direction @ desired)


def local_minus_x_base(object_pose: np.ndarray) -> np.ndarray:
    return -np.asarray(object_pose, dtype=np.float64)[:3, 0]


def hand_interval_for_y(y: float) -> str | None:
    if y > HAND_SIDE_EPS:
        return "left"
    if y < -HAND_SIDE_EPS:
        return "right"
    return None


def is_in_hand_y_interval(object_pose: np.ndarray, desired_y_sign: float) -> bool:
    y = float(local_minus_x_base(object_pose)[1])
    if desired_y_sign > 0.0:
        return y > HAND_SIDE_EPS
    if desired_y_sign < 0.0:
        return y < -HAND_SIDE_EPS
    return False


def local_minus_x_xy_angle_deg(object_pose: np.ndarray) -> float | None:
    local_minus_x = local_minus_x_base(object_pose)[:2]
    norm = float(np.linalg.norm(local_minus_x))
    if norm < 1e-8:
        return None
    angle_deg = float(np.rad2deg(np.arctan2(local_minus_x[1], local_minus_x[0])))
    return angle_deg % 360.0


def desired_side_sign(hand: str) -> float:
    if hand == "right":
        return -1.0
    if hand == "left":
        return 1.0
    return 0.0


def grasp_sector_for_hand(hand: str) -> tuple[float, float] | None:
    if hand == "left":
        return LEFT_GRASP_SECTOR_DEG
    if hand == "right":
        return RIGHT_GRASP_SECTOR_DEG
    return None


def fallback_angle_for_hand(hand: str) -> float | None:
    if hand == "left":
        return LEFT_FALLBACK_ANGLE_DEG
    if hand == "right":
        return RIGHT_FALLBACK_ANGLE_DEG
    return None


def local_z_rotation_deg(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_deg))
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
