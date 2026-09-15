"""Read settled hardware feedback for post-interruption HOME motion.

URDF TF tool offsets need not match the controller's MJCF tool offsets. Never
use TF, joint_states_cmd, or a commanded endpoint as a substitute for feedback.
"""
from __future__ import annotations

import time
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from grasp_core.core.math.pose import normalize_quaternion, quaternion_angle_rad


class MeasuredToolFK:
    def __init__(self, scene: Path, frame: str, hand: str) -> None:
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromMJCF(str(scene))
        self.data = self.model.createData()
        for name in (frame, f"{hand}_tool"):
            if not self.model.existFrame(name):
                raise RuntimeError(f"IK model has no frame {name!r}")
        self.root = self.model.getFrameId(frame)
        self.tool = self.model.getFrameId(f"{hand}_tool")
        self.joints = []
        for index in range(1, self.model.njoints):
            joint = self.model.joints[index]
            if joint.nq != 1 or joint.nv != 1:
                raise RuntimeError("HOME feedback FK requires scalar robot joints")
            self.joints.append((self.model.names[index], joint.idx_q))

    def evaluate(self, msg):
        if len(msg.name) != len(msg.position) or len(msg.name) != len(set(msg.name)):
            raise RuntimeError("invalid measured JointState names/positions")
        if len(msg.velocity) != len(msg.name):
            raise RuntimeError("measured JointState velocities unavailable")
        indices = {name: i for i, name in enumerate(msg.name)}
        q = self.pin.neutral(self.model)
        velocities = []
        for name, qi in self.joints:
            if name not in indices:
                raise RuntimeError(f"measured JointState missing IK joint {name}")
            i = indices[name]
            q[qi] = msg.position[i]
            velocities.append(msg.velocity[i])
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(velocities)):
            raise RuntimeError("non-finite measured joint state")
        self.pin.framesForwardKinematics(self.model, self.data, q)
        pose = self.data.oMf[self.root].inverse() * self.data.oMf[self.tool]
        xyzquat = self.pin.SE3ToXYZQUAT(pose)
        return q, np.asarray(velocities), (xyzquat[:3].copy(), normalize_quaternion(xyzquat[3:]))


def _running_colcon_install(package: str) -> Path | None:
    """Find the install root used by the currently running package process."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    marker = ("install", package, "lib", package)
    for cmdline in proc.glob("[0-9]*/cmdline"):
        try:
            arguments = cmdline.read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        for raw_argument in arguments:
            try:
                parts = Path(os.fsdecode(raw_argument)).parts
            except (TypeError, ValueError):
                continue
            for index in range(len(parts) - len(marker) + 1):
                if tuple(parts[index:index + len(marker)]) == marker:
                    return Path(*parts[:index + 1])
    return None


def _package_share(package: str, *, install_root: Path | None = None) -> Path:
    """Resolve a share directory even if the caller did not source the overlay."""
    if install_root is not None:
        candidate = install_root / package / "share" / package
        if candidate.is_dir():
            return candidate

    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory(package))
    except Exception:  # The running overlay may be absent from this process.
        pass

    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        candidate = Path(prefix) / "share" / package
        if candidate.is_dir():
            return candidate

    # Last resort for an unsourced local colcon workspace. Keep the search
    # shallow and refuse ambiguity instead of silently choosing a stale model.
    candidates = set()
    home = Path.home()
    for pattern in (
        f"*/install/{package}/share/{package}",
        f"*/*/install/{package}/share/{package}",
    ):
        candidates.update(path.resolve() for path in home.glob(pattern) if path.is_dir())
    if len(candidates) == 1:
        return candidates.pop()
    detail = ", ".join(str(path) for path in sorted(candidates)) or "none"
    raise RuntimeError(f"cannot resolve unique package {package!r}; candidates: {detail}")


def resolve_ik_scene(params: dict) -> Path:
    """Resolve the same scene/config selection as request_ik_tester."""
    import yaml

    install_root = _running_colcon_install("marvin_qp_controller")
    controller_share = _package_share(
        "marvin_qp_controller", install_root=install_root,
    )
    config_root = controller_share / "config"
    config = Path(params["robot_param_file"] or f"robot_param_{params['robot_model']}.yaml")
    if not config.is_absolute():
        config = config_root / config
    with config.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)["/**"]["ros__parameters"]
    scene = Path(params["scene_file"] or settings.get("scene_file")
                 or settings.get("mink_scene_file") or "teleop_scene_pro.xml")
    if not scene.is_absolute():
        description_share = _package_share(
            "marvin_description", install_root=install_root,
        )
        scene = description_share / "mjcf" / scene
    # request_ik_tester unwraps the first include of its scene wrapper.
    include = ET.parse(scene).getroot().find(".//include")
    if include is not None:
        included = Path(include.attrib["file"])
        scene = included if included.is_absolute() else scene.parent / included
    return scene.resolve(strict=True)


class SettledFeedback:
    """Only advancing, recent samples can establish a stable interval."""
    def __init__(self, requested_ns: int) -> None:
        self.last_stamp = requested_ns
        self.anchor = None
        self.stable_since = None

    def reset(self) -> None:
        self.anchor = None
        self.stable_since = None

    def update(self, stamp_ns, now_ns, q, velocities, pose) -> bool:
        if not 0 <= now_ns - stamp_ns <= 250_000_000:
            self.reset()
            return False
        if stamp_ns <= self.last_stamp:
            return False
        if stamp_ns - self.last_stamp > 250_000_000:
            self.reset()
        self.last_stamp = stamp_ns
        if np.max(np.abs(velocities)) > 0.01:
            self.reset()
            return False
        if self.anchor is None or (
            np.max(np.abs(q - self.anchor[0])) > 0.002
            or np.linalg.norm(pose[0] - self.anchor[1][0]) > 0.001
            or quaternion_angle_rad(pose[1], self.anchor[1][1]) > np.deg2rad(0.5)
        ):
            self.anchor = (q.copy(), pose)
            self.stable_since = stamp_ns
            return False
        return stamp_ns - self.stable_since >= 500_000_000


class MeasuredToolPoseReader:
    """Reuse ROS feedback resources and FK models across pose reads."""

    def __init__(self, client) -> None:
        from rcl_interfaces.srv import GetParameters
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState

        self.client = client
        self.GetParameters = GetParameters
        self.node = client.rclpy.create_node(
            f"measured_pose_sync_{os.getpid()}_{time.monotonic_ns()}"
        )
        self.executor = SingleThreadedExecutor(context=self.node.context)
        self.executor.add_node(self.node)
        self.service = self.node.create_client(
            GetParameters, "/request_ik_tester/get_parameters"
        )
        self.latest = []
        self.subscription = self.node.create_subscription(
            JointState,
            "/joint_states",
            lambda msg: self.latest.__setitem__(slice(None), [msg]),
            qos_profile_sensor_data,
        )
        self.params = None
        self.scene = None
        self.fk_by_hand = {}

    def close(self) -> None:
        self.node.destroy_subscription(self.subscription)
        self.executor.remove_node(self.node)
        self.executor.shutdown(timeout_sec=0.0)
        self.node.destroy_node()

    def _spin(self, deadline: float, cancelled) -> None:
        if cancelled():
            raise RuntimeError("HOME interrupted by S")
        if not self.client.ok() or time.monotonic() >= deadline:
            raise RuntimeError(
                "HOME measured joint feedback/model unavailable or arm not settled"
            )
        self.executor.spin_once(timeout_sec=0.05)

    def _load_model(self, hand: str, deadline: float, cancelled) -> MeasuredToolFK:
        if hand in self.fk_by_hand:
            return self.fk_by_hand[hand]

        if self.params is None:
            while not self.service.service_is_ready():
                self._spin(deadline, cancelled)
            request = self.GetParameters.Request()
            request.names = [
                "scene_file",
                "robot_model",
                "robot_param_file",
                "marker_frame_id",
            ]
            future = self.service.call_async(request)
            while not future.done():
                self._spin(deadline, cancelled)
            values = future.result().values
            if len(values) != len(request.names) or any(v.type != 4 for v in values):
                raise RuntimeError("request_ik_tester model parameters unavailable")
            self.params = dict(zip(request.names, (v.string_value for v in values)))
            if self.params["marker_frame_id"] != self.client.frame_id:
                raise RuntimeError(
                    "HOME frame differs from request_ik_tester marker_frame_id"
                )
            self.scene = resolve_ik_scene(self.params)

        self.fk_by_hand[hand] = MeasuredToolFK(
            self.scene, self.client.frame_id, hand
        )
        return self.fk_by_hand[hand]

    def read(
        self,
        hand: str,
        *,
        timeout_sec: float,
        cancelled,
        require_settled: bool,
    ):
        deadline = time.monotonic() + max(float(timeout_sec), 0.1)
        fk = self._load_model(hand, deadline, cancelled)
        requested_ns = self.node.get_clock().now().nanoseconds
        settled = SettledFeedback(requested_ns)
        while True:
            self._spin(deadline, cancelled)
            if not self.latest:
                continue
            msg = self.latest.pop()
            stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            q, velocities, pose = fk.evaluate(msg)
            now_ns = self.node.get_clock().now().nanoseconds
            is_fresh = (
                stamp_ns > requested_ns
                and 0 <= now_ns - stamp_ns <= 250_000_000
            )
            if (not require_settled and is_fresh) or (
                require_settled and settled.update(stamp_ns, now_ns, q, velocities, pose)
            ):
                if cancelled():
                    raise RuntimeError("HOME interrupted by S")
                self.node.get_logger().info(
                    f"measured start hand={hand} source=/joint_states "
                    f"settled={require_settled} model={self.scene} "
                    f"stamp_ns={stamp_ns} xyz={pose[0].tolist()} "
                    f"quat={pose[1]} "
                    f"max_joint_speed={np.max(np.abs(velocities)):.6f}rad/s"
                )
                return pose


def wait_for_measured_home_pose(
    client,
    hand,
    *,
    timeout_sec=5.0,
    cancelled=lambda: False,
    require_settled=True,
):
    reader = getattr(client, "_measured_pose_reader", None)
    if reader is None:
        reader = MeasuredToolPoseReader(client)
        client._measured_pose_reader = reader
    return reader.read(
        hand,
        timeout_sec=timeout_sec,
        cancelled=cancelled,
        require_settled=require_settled,
    )
