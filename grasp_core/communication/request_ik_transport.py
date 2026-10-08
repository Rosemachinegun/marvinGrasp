#!/usr/bin/env python3
"""Low-level ROS2 PoseStamped target transport for request IK."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from grasp_core.core.math.pose import checked_position, normalize_quaternion
from grasp_core.core.types.target_pose import matrix_to_quaternion

if TYPE_CHECKING:
    from grasp_core.core.types.target_pose import TargetObjectPose


GRASP_TARGET_MARKER_TOPIC = "/flowpose/grasp_targets"


class Ros2PoseTargetPublisher:
    """Low-level ROS2 publisher for left/right IK target topics."""

    def __init__(
        self,
        *,
        left_topic: str,
        right_topic: str,
        frame_id: str,
        node_name: str = "flowpose_request_ik_tester",
    ) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from rclpy.node import Node
            from visualization_msgs.msg import Marker, MarkerArray
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 Python packages are not importable. Source your ROS2 workspace first."
            ) from exc

        self.rclpy = rclpy
        self.PoseStamped = PoseStamped
        self.Marker = Marker
        self.MarkerArray = MarkerArray
        self.frame_id = frame_id
        self.left_topic = left_topic
        self.right_topic = right_topic

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node(node_name)
        self.left_pub = self.node.create_publisher(PoseStamped, left_topic, 10)
        self.right_pub = self.node.create_publisher(PoseStamped, right_topic, 10)
        marker_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.grasp_target_marker_pub = self.node.create_publisher(
            MarkerArray,
            GRASP_TARGET_MARKER_TOPIC,
            marker_qos,
        )

    def wait_for_settled_tool_pose(
        self, hand: str, *, timeout_sec: float = 5.0, cancelled=lambda: False,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        from grasp_core.communication.request_ik_feedback import wait_for_measured_home_pose

        return wait_for_measured_home_pose(
            self, hand, timeout_sec=timeout_sec, cancelled=cancelled,
        )

    def wait_for_current_tool_pose(
        self, hand: str, *, timeout_sec: float = 5.0, cancelled=lambda: False,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        from grasp_core.communication.request_ik_feedback import wait_for_measured_home_pose

        return wait_for_measured_home_pose(
            self,
            hand,
            timeout_sec=timeout_sec,
            cancelled=cancelled,
            require_settled=False,
        )

    def close(self) -> None:
        measured_reader = getattr(self, "_measured_pose_reader", None)
        if measured_reader is not None:
            measured_reader.close()
            self._measured_pose_reader = None
        self.node.destroy_node()

    def ok(self) -> bool:
        return bool(self.rclpy.ok())

    def topic_for_hand(self, hand: str) -> str:
        return self.left_topic if hand == "left" else self.right_topic

    def publish_pose(
        self,
        hand: str,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
    ) -> None:
        publisher = self.left_pub if hand == "left" else self.right_pub
        publisher.publish(self.make_pose_stamped(position_xyz, orientation_xyzw))

    def publish_grasp_target_markers(
        self,
        targets: list["TargetObjectPose"],
        *,
        selected_index: int = 0,
    ) -> None:
        """Show the latest FlowPose object poses in the request-IK RViz window."""
        marker_array = self.MarkerArray()
        clear = self.Marker()
        clear.action = self.Marker.DELETEALL
        marker_array.markers.append(clear)

        if targets:
            selected_index = min(max(int(selected_index), 0), len(targets) - 1)
        stamp = self.node.get_clock().now().to_msg()
        axis_colors = (
            (1.0, 0.12, 0.12),
            (0.12, 1.0, 0.12),
            (0.12, 0.35, 1.0),
        )

        for target_index, target in enumerate(targets):
            pose = np.asarray(target.base_pose, dtype=np.float64)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                continue
            xyz = pose[:3, 3]
            quat = matrix_to_quaternion(pose)
            size = (
                np.asarray(target.size, dtype=np.float64).reshape(-1)
                if target.size is not None
                else np.asarray([], dtype=np.float64)
            )
            if size.size < 3 or not np.all(np.isfinite(size[:3])):
                size = np.asarray([0.08, 0.08, 0.08], dtype=np.float64)
            size = np.maximum(np.abs(size[:3]), 0.015)
            selected = target_index == selected_index
            base_id = target_index * 10

            box = self.Marker()
            box.header.frame_id = self.frame_id
            box.header.stamp = stamp
            box.ns = "flowpose_grasp_objects"
            box.id = base_id
            box.type = self.Marker.CUBE
            box.action = self.Marker.ADD
            self._set_marker_pose(box, xyz, quat)
            box.scale.x, box.scale.y, box.scale.z = map(float, size)
            if selected:
                box.color.r, box.color.g, box.color.b = 1.0, 0.55, 0.05
                box.color.a = 0.72
            else:
                box.color.r, box.color.g, box.color.b = 0.1, 0.75, 1.0
                box.color.a = 0.38
            marker_array.markers.append(box)

            axis_length = max(float(np.max(size)) * 0.8, 0.10)
            axis_radius = max(axis_length * 0.055, 0.006)
            for axis_index, color in enumerate(axis_colors):
                axis = self.Marker()
                axis.header.frame_id = self.frame_id
                axis.header.stamp = stamp
                axis.ns = "flowpose_grasp_axes"
                axis.id = base_id + axis_index + 1
                axis.type = self.Marker.ARROW
                axis.action = self.Marker.ADD
                axis.pose.orientation.w = 1.0
                start = type(axis.pose.position)()
                start.x, start.y, start.z = map(float, xyz)
                endpoint = xyz + pose[:3, axis_index] * axis_length
                end = type(axis.pose.position)()
                end.x, end.y, end.z = map(float, endpoint)
                axis.points.extend((start, end))
                axis.scale.x = axis_radius
                axis.scale.y = axis_radius * 1.8
                axis.scale.z = axis_radius * 2.4
                axis.color.r, axis.color.g, axis.color.b = color
                axis.color.a = 1.0
                marker_array.markers.append(axis)

            text = self.Marker()
            text.header.frame_id = self.frame_id
            text.header.stamp = stamp
            text.ns = "flowpose_grasp_labels"
            text.id = base_id + 4
            text.type = self.Marker.TEXT_VIEW_FACING
            text.action = self.Marker.ADD
            label_height = float(size[2]) * 0.5 + 0.06
            self._set_marker_pose(
                text,
                xyz + np.asarray([0.0, 0.0, label_height]),
                (0.0, 0.0, 0.0, 1.0),
            )
            prefix = "TARGET " if selected else ""
            text.text = (
                f"{prefix}[{target_index}] {target.label}  "
                f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m"
            )
            text.scale.z = 0.045
            text.color.r = 1.0 if selected else 0.8
            text.color.g = 0.85 if selected else 0.9
            text.color.b = 0.15 if selected else 1.0
            text.color.a = 1.0
            marker_array.markers.append(text)

        self.grasp_target_marker_pub.publish(marker_array)

    @staticmethod
    def _set_marker_pose(marker, xyz, quaternion_xyzw) -> None:
        marker.pose.position.x = float(xyz[0])
        marker.pose.position.y = float(xyz[1])
        marker.pose.position.z = float(xyz[2])
        qx, qy, qz, qw = quaternion_xyzw
        marker.pose.orientation.x = float(qx)
        marker.pose.orientation.y = float(qy)
        marker.pose.orientation.z = float(qz)
        marker.pose.orientation.w = float(qw)

    def hold_pose(
        self,
        hand: str,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
        *,
        duration_sec: float,
        publish_rate_hz: float,
    ) -> int:
        duration_sec = max(float(duration_sec), 0.0)
        if duration_sec <= 0.0:
            return 0

        period_sec = 1.0 / max(float(publish_rate_hz), 0.1)
        deadline = time.monotonic() + duration_sec
        count = 0
        while self.ok() and time.monotonic() < deadline:
            self.publish_pose(hand, position_xyz, orientation_xyzw)
            count += 1
            time.sleep(period_sec)
        return count

    def make_pose_stamped(
        self,
        position_xyz: np.ndarray,
        orientation_xyzw: tuple[float, float, float, float],
    ):
        position = checked_position(position_xyz)
        qx, qy, qz, qw = normalize_quaternion(orientation_xyzw)
        msg = self.PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg
