#!/usr/bin/env python3
"""Long-object grasp policy for screwdriver handles and similar objects.

Coordinate conventions
----------------------

Published perception frame:
    local X = horizontal physical long axis
    local Y = horizontal short axis
    local Z = world +Z

Grasp-policy frame (same as the published perception frame):
    local X = physical long axis
    local Y = physical short axis
    local Z = world +Z

The sign of policy +X is object-centric and follows FlowPose's raw long-axis
sign.  The active arm is used only to select an equivalent wrist yaw / IK
orientation; it must not change the position-offset direction.

The resulting policy frame is always right-handed:
    X x Y = Z

Gripper TCP convention:
    local Y = gripper closing axis
    local Z = gripper approach axis
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from grasp_core.core.math.pose import (
    PickTemplateWaypoint,
    PoseWaypoint,
    apply_pregrasp_offset,
    checked_position,
    ik_downward_tilt_deg_for_hand,
    ik_downward_tilt_y_deg_for_hand,
    ik_orientation_rotation,
    make_downward_tilt_rotation,
    normalize_object_type,
    validate_pose_matrix,
)
from grasp_core.core.types.robot_target_pose import (
    TargetObjectPose,
    matrix_to_quaternion,
)
from grasp_core.core.math.vector import horizontal_unit_vector, normalized


@dataclass(frozen=True)
class LongObjectGraspPolicy:
    """Per-object grasp policy knobs for long, thin objects.

    closing_axis is in the reconstructed policy frame and must be "x" or "y":
        x -> close along the object long axis
        y -> close across the object, along the short axis
    """

    keyword: str
    closing_axis: str


LONG_OBJECT_POLICIES = (
    LongObjectGraspPolicy(
        keyword="screwdriver_handle",
        closing_axis="y",
    ),
    LongObjectGraspPolicy(
        keyword="pen",
        closing_axis="y",
    ),
)
POLICY_AXIS_INDEX_BY_NAME = {
    "x": 0,
    "y": 1,
}

WORLD_X_AXIS = np.asarray(
    [1.0, 0.0, 0.0],
    dtype=np.float64,
)
WORLD_Z_AXIS = np.asarray(
    [0.0, 0.0, 1.0],
    dtype=np.float64,
)

# request_ik TCP convention:
#
# local Y = closing axis
# local Z = approach axis
GRIPPER_CLOSING_AXIS_INDEX = 1
GRIPPER_APPROACH_AXIS_INDEX = 2

# FlowPose screwdriver convention:
#
# Published local X is the long axis and local Y is the short axis.
FLOWPOSE_LONG_AXIS_INDEX = 0
POLICY_LONG_AXIS_INDEX = 0
FINAL_TOOL_YAW_MIN_DEG = -15.0
FINAL_TOOL_YAW_MAX_DEG = 45.0


@dataclass(frozen=True)
class CuboidGraspPolicyResult:
    """Adjusted screwdriver-handle grasp plan."""

    pose: np.ndarray

    # Reconstructed Z-up policy frame:
    #
    # X = long axis
    # Y = lateral/short direction
    # Z = world up
    object_pose: np.ndarray

    # +1 left, -1 right
    side_sign: float

    # Policy Y axis, used as desired physical gripper closing direction.
    side_axis: np.ndarray

    # Policy X axis, physical screwdriver long axis.
    long_axis: np.ndarray

    long_axis_index: int
    closing_axis_name: str
    closing_axis_index: int
    closing_axis: np.ndarray

    yaw_deg: float
    tilt_y_deg: float


def long_object_grasp_policy_for_object(
    object_type: str | None = None,
    *,
    force: bool = False,
) -> LongObjectGraspPolicy | None:
    """Return the long-object policy, optionally without label dispatch."""

    if force:
        return LongObjectGraspPolicy(keyword="generic_long_object", closing_axis="y")

    normalized_type = normalize_object_type(object_type or "")

    for policy in LONG_OBJECT_POLICIES:
        if policy.keyword in normalized_type:
            validate_policy_axis_name(policy.closing_axis)
            return policy

    return None


def validate_policy_axis_name(axis_name: str) -> None:
    """Raise if an object policy uses an unsupported closing axis."""

    if axis_name not in POLICY_AXIS_INDEX_BY_NAME:
        raise ValueError(
            f"unsupported long-object closing axis {axis_name!r}; "
            "expected 'x' or 'y'"
        )


def policy_axis_index(axis_name: str) -> int:
    """Return reconstructed policy-frame axis index for "x"/"y"."""

    validate_policy_axis_name(axis_name)
    return POLICY_AXIS_INDEX_BY_NAME[axis_name]


def is_cuboid_object(object_type: str) -> bool:
    """Return True for supported screwdriver/pen style long objects."""

    return long_object_grasp_policy_for_object(object_type) is not None


def make_cuboid_gripper_pose(
    target: TargetObjectPose,
    args: argparse.Namespace,
    *,
    hand: str,
    force_long_object: bool = False,
) -> tuple[
    np.ndarray,
    CuboidGraspPolicyResult,
] | None:
    """Construct the gripper target pose for a screwdriver handle."""

    policy = long_object_grasp_policy_for_object(target.label, force=force_long_object)
    if policy is None:
        return None

    side_sign = side_sign_for_hand(hand)
    if side_sign == 0.0:
        return None

    object_pose = cuboid_z_up_object_pose(
        target.base_pose,
    )

    fallback_reason = validate_pose_matrix(object_pose)
    if fallback_reason is not None:
        return None

    # After cuboid_z_up_object_pose():
    #
    # object_pose[:, 0] = physical long axis
    # object_pose[:, 1] = lateral/short axis
    # object_pose[:, 2] = world +Z
    long_axis = object_pose[:3, 0].copy()
    side_axis = object_pose[:3, 1].copy()
    closing_axis_index = policy_axis_index(policy.closing_axis)
    closing_axis = object_pose[:3, closing_axis_index].copy()

    gripper_pose = np.eye(4, dtype=np.float64)
    gripper_pose[:3, 3] = object_pose[:3, 3]

    orientation, yaw_deg, tilt_y_deg = (
        cuboid_wrist_orientation(
            args,
            hand=hand,
            closing_axis=closing_axis,
        )
    )

    gripper_pose[:3, :3] = orientation

    gripper_pose[:3, 3] += np.asarray(
        args.ik_grasp_tcp_offset_m,
        dtype=np.float64,
    )

    if args.ik_target_stage == "pregrasp":
        gripper_pose = apply_pregrasp_offset(
            gripper_pose,
            args,
        )

    return (
        gripper_pose,
        CuboidGraspPolicyResult(
            pose=gripper_pose,
            object_pose=object_pose,
            side_sign=side_sign,
            side_axis=side_axis,
            long_axis=long_axis,
            long_axis_index=POLICY_LONG_AXIS_INDEX,
            closing_axis_name=policy.closing_axis,
            closing_axis_index=closing_axis_index,
            closing_axis=closing_axis,
            yaw_deg=yaw_deg,
            tilt_y_deg=tilt_y_deg,
        ),
    )


def build_cuboid_pick_waypoints(
    target: TargetObjectPose,
    relative_waypoints: list[PickTemplateWaypoint],
    args: argparse.Namespace,
    *,
    hand: str,
    force_long_object: bool = False,
) -> list[PickTemplateWaypoint] | None:
    """Build screwdriver-handle pick waypoints.

    Relative XYZ positions are interpreted in the reconstructed policy frame:

        X = physical long axis
        Y = lateral/short direction
        Z = world up

    Therefore:
        relative X -> offset along the screwdriver long axis
        relative Y -> lateral offset across the screwdriver
        relative Z -> vertical offset
    """

    policy = long_object_grasp_policy_for_object(target.label, force=force_long_object)
    if policy is None:
        return None

    if not relative_waypoints:
        return []

    side_sign = side_sign_for_hand(hand)
    if side_sign == 0.0:
        return None

    object_pose = cuboid_z_up_object_pose(
        target.base_pose,
    )

    fallback_reason = validate_pose_matrix(object_pose)
    if fallback_reason is not None:
        raise ValueError(
            "invalid screwdriver_handle pose: "
            f"{fallback_reason}"
        )

    closing_axis_index = policy_axis_index(policy.closing_axis)
    closing_axis = object_pose[:3, closing_axis_index].copy()

    gripper_pose = np.eye(4, dtype=np.float64)

    gripper_pose[:3, :3] = (
        cuboid_wrist_orientation(
            args,
            hand=hand,
            closing_axis=closing_axis,
        )[0]
    )

    orientation = matrix_to_quaternion(
        gripper_pose,
    )

    waypoints: list[PickTemplateWaypoint] = []

    for (
        relative_xyz,
        _relative_quat,
        gripper_value,
    ) in relative_waypoints:

        relative_point = np.ones(
            4,
            dtype=np.float64,
        )

        relative_point[:3] = checked_position(
            relative_xyz,
        )

        position = (
            object_pose @ relative_point
        )[:3]

        waypoints.append(
            (
                position.copy(),
                orientation,
                float(gripper_value),
            )
        )

    return waypoints


def cuboid_pose_waypoints(
    target: TargetObjectPose,
    args: argparse.Namespace,
    *,
    hand: str,
) -> tuple[
    list[PoseWaypoint],
    CuboidGraspPolicyResult,
] | None:
    """Return the single IK target waypoint for the screwdriver policy."""

    result = make_cuboid_gripper_pose(
        target,
        args,
        hand=hand,
    )

    if result is None:
        return None

    pose, metadata = result

    return (
        [
            (
                pose[:3, 3].copy(),
                matrix_to_quaternion(pose),
            )
        ],
        metadata,
    )


def cuboid_z_up_object_pose(
    object_pose: np.ndarray,
    size: np.ndarray | None = None,
    *,
    hand: str | None = None,
) -> np.ndarray:
    """Validate and return the already-canonical long-object frame.

    Convention:
        X = physical long axis
        Y = physical short axis
        Z = world +Z

    The long-axis sign is normalized for grasping: the XY yaw of +X is always
    within [-90, 90] degrees in the robot base frame.  The short axis is
    flipped together with X, so the frame stays right-handed.  This removes
    the raw perception sign ambiguity without depending on the target's
    distance or direction from the robot base.

    The frame always remains right-handed.
    """

    pose = np.asarray(object_pose, dtype=np.float64).copy()

    if hand is not None and side_sign_for_hand(hand) == 0.0:
        raise ValueError(f"unsupported hand: {hand!r}")

    del size
    reason = validate_pose_matrix(pose)
    if reason is not None:
        raise ValueError(f"invalid long-object pose: {reason}")
    if not np.allclose(pose[:3, 2], WORLD_Z_AXIS, atol=1e-5):
        raise ValueError("long-object pose must already be Z-up")
    if abs(float(pose[2, FLOWPOSE_LONG_AXIS_INDEX])) > 1e-5:
        raise ValueError("long-object X long axis must already be horizontal")

    pose = orient_long_object_axis_to_forward_half_plane(pose)
    return pose


def orient_long_object_axis_to_forward_half_plane(object_pose: np.ndarray) -> np.ndarray:
    """Keep the long-object X yaw within [-90, 90] degrees.

    The reference is only the robot base X axis; target position is not used.
    A long-object axis is an unoriented line, so reversing X is physically
    equivalent. Reverse Y at the same time to preserve ``X x Y = Z``.
    """

    pose = np.asarray(object_pose, dtype=np.float64).copy()
    long_axis = pose[:3, 0]
    yaw_rad = float(np.arctan2(long_axis[1], long_axis[0]))
    if abs(yaw_rad) > (0.5 * np.pi):
        pose[:3, 0] *= -1.0
        pose[:3, 1] *= -1.0
    return pose


def cuboid_long_axis_index(
    size: np.ndarray | None,
) -> int:
    """Return the long-axis index in the unified Z-up object frame."""

    del size
    return POLICY_LONG_AXIS_INDEX


def cuboid_long_axis(
    object_pose: np.ndarray,
    size: np.ndarray | None = None,
    *,
    hand: str | None = None,
) -> np.ndarray:
    """Return signed horizontal physical long axis.

    The returned axis is local X in the canonical long-object frame.

    The returned direction is object-centric and does not depend on the active
    hand.  ``hand`` is accepted for backward-compatible validation only.
    """

    pose = np.asarray(
        object_pose,
        dtype=np.float64,
    )

    if pose.shape != (4, 4):
        raise ValueError(
            f"object_pose must be 4x4, got {pose.shape}"
        )

    side_sign = side_sign_for_hand(hand)

    if hand is not None and side_sign == 0.0:
        raise ValueError(
            f"unsupported hand: {hand!r}"
        )

    z_up_pose = cuboid_z_up_object_pose(
        pose,
        size,
    )

    long_axis = horizontal_unit_vector(
        z_up_pose[:3, POLICY_LONG_AXIS_INDEX],
    )

    if long_axis is None:
        raise ValueError(
            "screwdriver long axis has no valid "
            "horizontal component"
        )

    return long_axis.copy()


def cuboid_lateral_axis(
    object_pose: np.ndarray,
) -> np.ndarray:
    """Return policy local Y, the lateral/closing direction.

    cuboid_z_up_object_pose() already defines:

        Y = Z x X

    so X must not be flipped independently, otherwise the frame would
    become left-handed.
    """

    pose = np.asarray(
        object_pose,
        dtype=np.float64,
    )

    return normalized(
        pose[:3, 1]
    )


def cuboid_wrist_orientation(
    args: argparse.Namespace,
    *,
    hand: str,
    closing_axis: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Construct screwdriver wrist orientation.

    The gripper closing axis is aligned with the configured policy X/Y axis,
    while the shared Y-axis downward tilt is preserved.
    """

    base_rotation = ik_orientation_rotation(
        args,
    )

    yaw_deg = cuboid_yaw_deg(
        base_rotation,
        closing_axis,
        reference_yaw_deg=(
            ik_downward_tilt_deg_for_hand(
                args,
                hand,
            )
        ),
    )

    tilt_y_deg = (
        ik_downward_tilt_y_deg_for_hand(
            args,
            hand,
        )
    )

    tilt_rotation = (
        make_downward_tilt_rotation(
            yaw_deg,
            axis="z",
            y_deg=tilt_y_deg,
        )
    )

    if (
        getattr(
            args,
            "ik_downward_tilt_frame",
            "local",
        )
        == "base"
    ):
        orientation = (
            tilt_rotation @ base_rotation
        )
    else:
        orientation = (
            base_rotation @ tilt_rotation
        )

    orientation = normalize_final_z_yaw(orientation)

    return (
        orientation,
        yaw_deg,
        tilt_y_deg,
    )


def normalize_final_z_yaw(rotation: np.ndarray) -> np.ndarray:
    """Force final tool-Z yaw into [-15, 45] degrees.

    The correction is applied around the base-frame Z axis.  This changes the
    actual orientation and therefore the quaternion sent to the robot.
    """

    result = np.asarray(rotation, dtype=np.float64).copy()
    z_axis = result[:3, 2]
    yaw_deg = float(np.rad2deg(np.arctan2(z_axis[1], z_axis[0])))

    # First select the equivalent robot-facing representation.  Directly
    # clipping 111 degrees to +55 would keep the unsafe sign; 111 - 180 is
    # -69 degrees and must then be clipped on the negative side.
    normalized_yaw_deg = yaw_deg
    if normalized_yaw_deg > 90.0:
        normalized_yaw_deg -= 180.0
    elif normalized_yaw_deg < -90.0:
        normalized_yaw_deg += 180.0

    clipped_yaw_deg = float(
        np.clip(
            normalized_yaw_deg,
            FINAL_TOOL_YAW_MIN_DEG,
            FINAL_TOOL_YAW_MAX_DEG,
        )
    )
    correction_deg = clipped_yaw_deg - yaw_deg

    if correction_deg != 0.0:
        result = (
            make_downward_tilt_rotation(
                correction_deg,
                axis="z",
            )
            @ result
        )

    final_z_axis = result[:3, 2]
    final_yaw_deg = float(
        np.rad2deg(
            np.arctan2(final_z_axis[1], final_z_axis[0])
        )
    )
    print(
        "\033[93m"
        f"[long_object] z_yaw raw={yaw_deg:.2f}deg "
        f"normalized={normalized_yaw_deg:.2f}deg "
        f"final={final_yaw_deg:.2f}deg"
        "\033[0m",
        flush=True,
    )
    return result


def cuboid_yaw_deg(
    base_rotation: np.ndarray,
    closing_axis: np.ndarray,
    *,
    reference_yaw_deg: float = 0.0,
) -> float:
    """Compute yaw that aligns the gripper closing axis with a policy axis.

    Two orientations separated by 180 degrees describe the same parallel
    closing-axis line.  The orientation closest to the hand-specific
    reference yaw is selected.
    """

    desired_closing = normalized(
        closing_axis,
    )

    desired_in_base_wrist = (
        np.asarray(
            base_rotation,
            dtype=np.float64,
        ).T
        @ desired_closing
    )

    desired_horizontal = (
        horizontal_unit_vector(
            desired_in_base_wrist,
        )
    )

    if desired_horizontal is None:
        desired_horizontal = (
            WORLD_X_AXIS.copy()
        )

    yaw_rad = yaw_rad_for_horizontal_axis_alignment(
        desired_horizontal,
        GRIPPER_CLOSING_AXIS_INDEX,
    )

    yaw_deg = float(
        np.rad2deg(
            yaw_rad,
        )
    )

    return closest_parallel_axis_yaw_deg(
        yaw_deg,
        reference_yaw_deg,
    )


def yaw_rad_for_horizontal_axis_alignment(
    desired_horizontal: np.ndarray,
    local_axis_index: int,
) -> float:
    """Return yaw that points the selected local XY axis at desired_horizontal."""

    desired = normalized(
        np.asarray(
            desired_horizontal,
            dtype=np.float64,
        )
    )

    if local_axis_index == 1:
        # Rz(yaw) local Y = [-sin(yaw), cos(yaw), 0].
        return float(np.arctan2(-desired[0], desired[1]))

    if local_axis_index == 0:
        # Rz(yaw) local X = [cos(yaw), sin(yaw), 0].
        return float(np.arctan2(desired[1], desired[0]))

    raise ValueError(
        f"unsupported horizontal local axis index: {local_axis_index}"
    )


def closest_parallel_axis_yaw_deg(
    yaw_deg: float,
    reference_yaw_deg: float,
) -> float:
    """Choose yaw or yaw+180 closest to the preferred wrist orientation."""

    candidates = (
        normalize_angle_deg(
            yaw_deg,
        ),
        normalize_angle_deg(
            yaw_deg + 180.0,
        ),
    )

    return min(
        candidates,
        key=lambda candidate: (
            angular_distance_deg(
                candidate,
                reference_yaw_deg,
            ),
            abs(candidate),
        ),
    )


def normalize_angle_deg(
    angle_deg: float,
) -> float:
    """Normalize angle to [-180, 180)."""

    return (
        float(angle_deg) + 180.0
    ) % 360.0 - 180.0


def angular_distance_deg(
    a_deg: float,
    b_deg: float,
) -> float:
    """Shortest absolute angular distance in degrees."""

    return abs(
        normalize_angle_deg(
            float(a_deg)
            - float(b_deg)
        )
    )


def closing_axis_from_orientation(
    rotation: np.ndarray,
) -> np.ndarray:
    """Return gripper closing axis expressed in base frame."""

    return np.asarray(
        rotation,
        dtype=np.float64,
    )[:, GRIPPER_CLOSING_AXIS_INDEX]


def approach_axis_from_orientation(
    rotation: np.ndarray,
) -> np.ndarray:
    """Return gripper local Z expressed in base frame."""

    return np.asarray(
        rotation,
        dtype=np.float64,
    )[:, GRIPPER_APPROACH_AXIS_INDEX]


def side_sign_for_hand(
    hand: str,
) -> float:
    """Return body's Y-side sign for a hand.

    left  -> +1
    right -> -1
    """

    hand_name = (
        str(hand)
        .strip()
        .lower()
    )

    if hand_name == "left":
        return 1.0

    if hand_name == "right":
        return -1.0

    return 0.0
