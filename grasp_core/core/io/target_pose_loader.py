"""Load target and camera data without exposing file I/O from callers."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from grasp_core.core.types.camera import CameraExtrinsic
from grasp_core.core.types.target_pose import (
    TargetObjectPose,
    make_child_frame_ids,
    make_target_object_pose,
    parse_vector,
)


def load_camera_extrinsic_from_xacro(
    xacro_path: Path,
    joint_name: str = "camera_joint",
) -> CameraExtrinsic:
    root = ET.parse(xacro_path).getroot()
    for joint in root.iter("joint"):
        if joint.attrib.get("name") != joint_name:
            continue
        parent, child, origin = (joint.find(name) for name in ("parent", "child", "origin"))
        if parent is None or child is None or origin is None:
            raise ValueError(f"Joint {joint_name} is missing parent/child/origin.")
        return CameraExtrinsic(
            parent.attrib["link"], child.attrib["link"],
            parse_vector(origin.attrib.get("xyz", "0 0 0")),
            parse_vector(origin.attrib.get("rpy", "0 0 0")),
        )
    raise ValueError(f"Joint {joint_name} was not found in {xacro_path}.")


def load_target_objects_from_flowpose_json(
    flowpose_json: str | Path,
    base_to_camera: np.ndarray,
) -> list[TargetObjectPose]:
    payload = json.loads(Path(flowpose_json).read_text(encoding="utf-8"))
    objects = payload.get("objects") or []
    if objects:
        labels = [str(obj.get("name") or f"object_{i + 1}") for i, obj in enumerate(objects)]
        return [make_target_object_pose(
            label=labels[i], frame_id=frame_id,
            camera_pose=np.asarray(obj["pose"], dtype=np.float64),
            base_to_camera=base_to_camera,
            size=np.asarray(obj["size"], dtype=np.float64) if obj.get("size") is not None else None,
            score=obj.get("score"),
        ) for i, (obj, frame_id) in enumerate(zip(objects, make_child_frame_ids(labels), strict=False))]

    labels = payload.get("labels") or []
    poses = payload.get("pose_all") or []
    lengths = payload.get("length_all") or []
    labels = labels or [f"object_{i + 1}" for i in range(len(poses))]
    frame_ids = make_child_frame_ids(labels)
    return [make_target_object_pose(
        label=str(label), frame_id=frame_ids[i],
        camera_pose=np.asarray(pose, dtype=np.float64),
        base_to_camera=base_to_camera,
        size=np.asarray(lengths[i], dtype=np.float64) if i < len(lengths) else None,
    ) for i, (label, pose) in enumerate(zip(labels, poses, strict=False))]
