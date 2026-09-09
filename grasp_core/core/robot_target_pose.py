"""核心数据模型：定义相机外参和机器人坐标系下的目标物体位姿。"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from grasp_core.core.long_object_axes import canonical_long_object_pose, is_long_object


@dataclass(frozen=True)
class CameraExtrinsic:
    parent_frame_id: str
    child_frame_id: str
    xyz: np.ndarray
    rpy: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        transform = rpy_to_matrix(self.rpy)
        transform[:3, 3] = self.xyz
        return transform


@dataclass(frozen=True)
class TargetObjectPose:
    label: str
    frame_id: str
    camera_pose: np.ndarray
    base_pose: np.ndarray
    size: np.ndarray | None = None
    score: float | None = None

    @property
    def base_xyz(self) -> np.ndarray:
        return self.base_pose[:3, 3]


def load_camera_extrinsic_from_xacro(
    xacro_path: Path,
    joint_name: str = "camera_joint",
) -> CameraExtrinsic:
    root = ET.parse(xacro_path).getroot()
    for joint in root.iter("joint"):
        if joint.attrib.get("name") != joint_name:
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        if parent is None or child is None or origin is None:
            raise ValueError(f"Joint {joint_name} is missing parent/child/origin.")
        return CameraExtrinsic(
            parent_frame_id=parent.attrib["link"],
            child_frame_id=child.attrib["link"],
            xyz=parse_vector(origin.attrib.get("xyz", "0 0 0")),
            rpy=parse_vector(origin.attrib.get("rpy", "0 0 0")),
        )
    raise ValueError(f"Joint {joint_name} was not found in {xacro_path}.")


def load_target_objects_from_flowpose_json(
    flowpose_json: str | Path,
    base_to_camera: np.ndarray,
) -> list[TargetObjectPose]:
    payload = json.loads(Path(flowpose_json).read_text(encoding="utf-8"))
    objects = payload.get("objects") or []
    if objects:
        labels = [str(obj.get("name") or f"object_{index + 1}") for index, obj in enumerate(objects)]
        frame_ids = make_child_frame_ids(labels)
        result: list[TargetObjectPose] = []
        for index, (obj, frame_id) in enumerate(zip(objects, frame_ids, strict=False)):
            camera_pose = np.asarray(obj["pose"], dtype=np.float64)
            result.append(
                make_target_object_pose(
                    label=labels[index],
                    frame_id=frame_id,
                    camera_pose=camera_pose,
                    base_to_camera=base_to_camera,
                    size=np.asarray(obj.get("size"), dtype=np.float64)
                    if obj.get("size") is not None
                    else None,
                    score=obj.get("score"),
                )
            )
        return result

    labels = payload.get("labels") or []
    pose_all = payload.get("pose_all") or []
    length_all = payload.get("length_all") or []
    if not labels:
        labels = [f"object_{index + 1}" for index in range(len(pose_all))]
    frame_ids = make_child_frame_ids(labels)
    result = []
    for index, (label, pose) in enumerate(zip(labels, pose_all, strict=False)):
        size = (
            np.asarray(length_all[index], dtype=np.float64)
            if index < len(length_all)
            else None
        )
        result.append(
            make_target_object_pose(
                label=str(label),
                frame_id=frame_ids[index],
                camera_pose=np.asarray(pose, dtype=np.float64),
                base_to_camera=base_to_camera,
                size=size,
                score=None,
            )
        )
    return result


def make_target_object_pose(
    *,
    label: str,
    frame_id: str,
    camera_pose: np.ndarray,
    base_to_camera: np.ndarray,
    size: np.ndarray | None = None,
    score: float | None = None,
) -> TargetObjectPose:
    if camera_pose.shape != (4, 4):
        raise ValueError(f"{label} pose must be 4x4, got {camera_pose.shape}.")
    base_pose = base_to_camera @ camera_pose
    # Normalize historical captures as well as live targets. Published long
    # objects use X for length throughout perception, templates and grasping.
    if is_long_object(label):
        base_pose, _ = canonical_long_object_pose(base_pose, source_long_axis=0)
        camera_pose = np.linalg.inv(base_to_camera) @ base_pose
    return TargetObjectPose(
        label=label,
        frame_id=frame_id,
        camera_pose=camera_pose,
        base_pose=base_pose,
        size=size,
        score=float(score) if score is not None else None,
    )


def parse_vector(raw: str) -> np.ndarray:
    values = [float(part) for part in raw.split()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got: {raw}")
    return np.asarray(values, dtype=np.float64)


def rpy_to_matrix(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float64,
    )
    ry = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float64,
    )
    rz = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rz @ ry @ rx
    return matrix


def matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return tuple(float(value) for value in quat)


def make_child_frame_ids(labels: Iterable[str]) -> list[str]:
    counts: dict[str, int] = {}
    frame_ids: list[str] = []
    for label in labels:
        base = normalize_label(label)
        counts[base] = counts.get(base, 0) + 1
        frame_ids.append(f"{base}_{counts[base]}")
    return frame_ids


def normalize_label(label: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(label).strip())
    text = re.sub(r"_+", "_", text).strip("_").lower()
    text = re.sub(r"_\d+$", "", text)
    return text or "object"
