#!/usr/bin/env python3
"""应用编排层：负责相机主循环、按键事件、异步任务状态和资源生命周期。

这里不直接实现感知算法、抓取规划、轨迹插值或底层通信，只调用各功能模块。
"""

from __future__ import annotations

import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_core.perception.flowpose_pipeline import (  # noqa: E402
    FlowPoseRunner,
    Sam3FrameResult,
    run_flowpose_job,
    run_sam3_job,
)
from grasp_core.communication.gripper_signal import (  # noqa: E402
    send_gripper_signal,
    start_gripper_signal_receiver,
    stop_gripper_signal_receiver,
)
from grasp_core.perception.perception_runtime import (  # noqa: E402
    build_ros_bridge,
    build_runner_kwargs,
    collect_flowpose_results,
    collect_sam3_results,
    freeze_bundle,
)
from grasp_core.perception.realsense_sam3 import (  # noqa: E402
    RealSenseD435,
    Sam3Runner,
    draw_live_hud,
    save_capture,
)
from grasp_core.config.request_ik_config import parse_args  # noqa: E402
from grasp_core.communication.request_ik_publisher import (  # noqa: E402
    build_ik_target_publisher,
)
from grasp_core.ui.request_ik_ui import make_dashboard  # noqa: E402
from grasp_core.core.robot_target_pose import (  # noqa: E402
    TargetObjectPose,
    load_camera_extrinsic_from_xacro,
)
from grasp_core.core.pose_math import select_ik_hand  # noqa: E402
from grasp_core.tasks.robot_actions import (  # noqa: E402
    RobotActionService,
    grip_confirmed,
    grip_failed_min_limit,
    grip_success_hand,
)
from grasp_core.planning.tool_pick_templates import load_tool_pick_templates  # noqa: E402


# OpenCV returns a single byte for keyboard input, so normalize to lowercase once.
KEY_QUIT = {ord("q"), 27}
KEY_CAPTURE = ord("a")
KEY_FLOWPOSE = ord("b")
KEY_TARGET = ord("c")
KEY_PERCEPTION_PIPELINE = ord("z")
KEY_RIGHT_HOME = ord("h")
KEY_LEFT_HOME = ord("j")
KEY_GRIP = ord("l")
KEY_RELEASE = ord("p")
KEY_PAUSE = ord("s")

DASHBOARD_WINDOW = "RealSense + SAM3 + FlowPose"


class RetryStage(str, Enum):
    """Stages of the automatic re-grasp state machine."""

    IDLE = "idle"
    RECOVERY = "recovery"
    RECAPTURE = "recapture"
    SAM3 = "sam3"
    FLOWPOSE = "flowpose"


class PipelineStage(str, Enum):
    """Stages for the one-key capture -> FlowPose -> grasp workflow."""

    IDLE = "idle"
    SAM3 = "sam3"
    FLOWPOSE = "flowpose"


@dataclass(slots=True)
class RuntimeState:
    """Mutable UI/inference state for the live loop."""

    sam_result: Sam3FrameResult | None = None
    sam_overlay: np.ndarray | None = None
    flowpose_overlay: np.ndarray | None = None
    base_targets: list[TargetObjectPose] = field(default_factory=list)
    status: str = "Ready"

    sam_future: Future | None = None
    flowpose_future: Future | None = None
    recovery_futures: list[Future] = field(default_factory=list)

    retry_stage: RetryStage = RetryStage.IDLE
    retry_attempts: int = 0
    retry_will_regrasp: bool = False
    replan_pending: bool = False
    pipeline_stage: PipelineStage = PipelineStage.IDLE
    pipeline_grasp_on_done: bool = True
    grasp_confirmed: bool = False
    grasp_confirmed_hand: str | None = None
    grasp_confirmed_label: str | None = None
    last_gripper_hand: str | None = None
    paused: bool = False
    home_needs_sync: set[str] = field(default_factory=set)
    drop_recovery_futures: list[Future] = field(default_factory=list)
    drop_regrasp_pending: bool = False


def is_pending(future: Future | None) -> bool:
    """Check whether a Future exists and has not finished yet."""
    return future is not None and not future.done()


def normalize_key(key: int) -> int:
    """Map ASCII uppercase keys to lowercase while preserving ESC."""
    key &= 0xFF
    if ord("A") <= key <= ord("Z"):
        return key + 32
    return key


class GraspDemoApp:
    """Owns resources and coordinates camera, perception and robot actions."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self.state = RuntimeState()
        self.gripper_future: Future | None = None
        self.gripper_future_command: str | None = None
        self.gripper_future_hand: str | None = None
        self.grasp_future: Future | None = None
        self.manual_home_future: Future | None = None
        self.manual_home_hand: str | None = None
        self.gripper_interrupted = False

        sam_kwargs, flowpose_kwargs, capture_dir = build_runner_kwargs(args)
        self.sam_kwargs = sam_kwargs
        self.flowpose_kwargs = flowpose_kwargs
        self.capture_dir = capture_dir

        self.camera_extrinsic = load_camera_extrinsic_from_xacro(
            Path(args.robot_xacro_path).expanduser(),
            args.camera_joint,
        )
        self._print_camera_extrinsic()

        self.camera = RealSenseD435(args.serial, args.width, args.height, args.fps)
        self.ros_bridge = None
        self.ik_publisher = None
        self.gripper_receiver = None
        self.robot_actions: RobotActionService | None = None

        # SAM3 and FlowPose are sequential in this workflow, so one inference worker
        # is enough and avoids running two GPU-heavy models at the same time.
        self.inference_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="perception",
        )

        # Recovery sends robot-home and gripper-release concurrently.
        self.action_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="robot-action",
        )
        self.gripper_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gripper-command",
        )

        self.sam_cache: dict[str, Sam3Runner | None] = {"runner": None}
        self.flowpose_cache: dict[str, FlowPoseRunner | None] = {"runner": None}
        self.pick_templates = None

    def _print_camera_extrinsic(self) -> None:
        ext = self.camera_extrinsic
        print(
            "[robot] loaded camera extrinsic "
            f"{ext.parent_frame_id} -> {ext.child_frame_id} "
            f"xyz={ext.xyz.tolist()} rpy={ext.rpy.tolist()}",
            flush=True,
        )

    def open(self) -> None:
        """Open hardware/services and preload heavy perception models."""
        self.camera.open()

        if self.args.ros2_publish:
            self.ros_bridge = build_ros_bridge(
                self.args,
                self.camera_extrinsic.matrix,
            )

        self.ik_publisher = build_ik_target_publisher(self.args)
        self.pick_templates = load_tool_pick_templates(self.args)
        self.robot_actions = RobotActionService(
            args=self.args,
            ik_publisher=self.ik_publisher,
            pick_templates=self.pick_templates,
        )
        self.gripper_receiver = start_gripper_signal_receiver(self.args)

        print("[startup] preloading SAM3 and FlowPose runners...", flush=True)
        self.sam_cache["runner"] = Sam3Runner(**self.sam_kwargs)
        self.flowpose_cache["runner"] = FlowPoseRunner(**self.flowpose_kwargs)
        print("[startup] SAM3 and FlowPose runners ready", flush=True)

    def close(self) -> None:
        """Release all resources. Safe to call after partial startup."""
        self.camera.close()

        if self.ik_publisher is not None:
            self.ik_publisher.request_stop()
        self.action_executor.shutdown(wait=True, cancel_futures=True)
        if self.ik_publisher is not None:
            self.ik_publisher.close()
        if self.ros_bridge is not None:
            self.ros_bridge.close()
        if self.gripper_receiver is not None:
            stop_gripper_signal_receiver(self.gripper_receiver)

        if is_pending(self.state.sam_future) or is_pending(self.state.flowpose_future):
            print("[exit] waiting for queued inference job(s)...", flush=True)

        self.inference_executor.shutdown(wait=True, cancel_futures=True)
        self.gripper_executor.shutdown(wait=True, cancel_futures=True)
        cv2.destroyAllWindows()

    def _collect_gripper_result(self) -> None:
        if self.gripper_future is None or not self.gripper_future.done():
            return
        if getattr(self, "gripper_interrupted", False):
            # A grip completed during/after S must not trigger put or recovery.
            self.gripper_future = None
            self.gripper_future_command = None
            self.gripper_future_hand = None
            self.gripper_interrupted = False
            return
        try:
            status = self.gripper_future.result()
        except Exception as exc:  # noqa: BLE001
            status = f"Gripper command failed: {exc}"
        self.state.status = status
        command = self.gripper_future_command
        hand = self.gripper_future_hand
        if command == "grip":
            if grip_failed_min_limit(status):
                failed_hand = (
                    hand
                    if hand in {"left", "right"}
                    else grip_success_hand(status) or self.state.last_gripper_hand
                )
                self.state.grasp_confirmed = False
                self.state.grasp_confirmed_hand = None
                self.state.grasp_confirmed_label = None
                if failed_hand in {"left", "right"}:
                    self.state.last_gripper_hand = failed_hand
                self.gripper_future = None
                self.gripper_future_command = None
                self.gripper_future_hand = None
                self.start_grip_failure_recovery(
                    status,
                    failed_hand=failed_hand if failed_hand in {"left", "right"} else None,
                )
                return

            if grip_confirmed(status) and hand in {"left", "right"}:
                self.state.grasp_confirmed = True
                self.state.grasp_confirmed_hand = (
                    grip_success_hand(status) or self.state.last_gripper_hand
                )
                self.state.grasp_confirmed_label = self.current_target_label()
                self.state.last_gripper_hand = self.state.grasp_confirmed_hand
                self.auto_put_after_confirmed_grasp()
            else:
                self.state.grasp_confirmed = False
                self.state.grasp_confirmed_hand = None
                self.state.grasp_confirmed_label = None
        elif command == "release":
            self.state.grasp_confirmed = False
            self.state.grasp_confirmed_hand = None
            self.state.grasp_confirmed_label = None
        self.gripper_future = None
        self.gripper_future_command = None
        self.gripper_future_hand = None

    def gripper_hand_for_manual_command(self) -> str | None:
        if self.state.grasp_confirmed_hand in {"left", "right"}:
            return self.state.grasp_confirmed_hand
        if self.state.last_gripper_hand in {"left", "right"}:
            return self.state.last_gripper_hand
        hand_mode = str(getattr(self.args, "ik_hand", "auto"))
        if hand_mode in {"left", "right"}:
            return hand_mode
        index = min(
            max(int(getattr(self.args, "ik_target_index", 0)), 0),
            max(len(self.state.base_targets) - 1, 0),
        )
        if self.state.base_targets:
            return select_ik_hand(self.state.base_targets[index].base_xyz, hand_mode)
        if bool(getattr(self.args, "dual_gripper", False)):
            return None
        return "right"

    def current_target_label(self) -> str | None:
        if not self.state.base_targets:
            return None
        index = min(
            max(int(getattr(self.args, "ik_target_index", 0)), 0),
            len(self.state.base_targets) - 1,
        )
        return self.state.base_targets[index].label

    def auto_pipeline_on_a(self) -> bool:
        return bool(getattr(self.args, "auto_pipeline_on_a", True))

    def send_gripper(self, command: str, hand: str | None = None) -> None:
        if self.gripper_future is not None and not self.gripper_future.done():
            self.state.status = "Gripper command already running"
            return
        hand = hand or self.gripper_hand_for_manual_command()
        if hand in {"left", "right"}:
            self.state.last_gripper_hand = hand
        target_label = hand if hand in {"left", "right"} else "both"
        self.state.status = f"Gripper {command} running hand={target_label}"
        self.gripper_interrupted = False
        self.gripper_future_command = command
        self.gripper_future_hand = hand
        self.gripper_future = self.gripper_executor.submit(
            send_gripper_signal,
            command,
            self.args,
            hand,
        )

    def reset_retry(self, *, reset_attempts: bool = False) -> None:
        """Stop automatic re-planning without touching running inference."""
        self.state.retry_stage = RetryStage.IDLE
        self.state.retry_will_regrasp = False
        self.state.replan_pending = False
        if reset_attempts:
            self.state.retry_attempts = 0

    def submit_sam3(self, bundle, *, retry: bool = False) -> None:
        """Capture one frame and submit exactly one SAM3 inference job."""
        if is_pending(self.state.sam_future):
            self.state.status = "SAM3 already running"
            return

        meta_path, metadata = save_capture(bundle, self.capture_dir)
        frozen_bundle = freeze_bundle(bundle)

        # A new capture invalidates every downstream visualization/result.
        self.state.sam_result = None
        self.state.sam_overlay = None
        self.state.flowpose_overlay = None
        self.state.base_targets.clear()

        # print(self.state)
        self.state.sam_future = self.inference_executor.submit(
            run_sam3_job,
            self.sam_cache,
            self.sam_kwargs,
            frozen_bundle,
            self.args.prompts,
            meta_path,
            metadata,
            self.args,
        )
        # print(self.state)

        prefix = f"Retry {self.state.retry_attempts}: " if retry else ""
        self.state.status = f"{prefix}captured frame {bundle.frame_id}; SAM3 running"
        print(f"[capture] saved metadata: {meta_path}", flush=True)
        if retry:
            print(f"[grip_retry] {self.state.status}", flush=True)

    def submit_flowpose(self, *, retry: bool = False) -> None:
        """Submit FlowPose for the latest completed SAM3 result."""
        if self.state.sam_result is None:
            self.state.status = "Run SAM3 first and wait for result"
            print(f"[FlowPose] {self.state.status}", flush=True)
            return

        if is_pending(self.state.flowpose_future):
            self.state.status = "FlowPose already running"
            return

        # print(self.state)
        self.state.flowpose_future = self.inference_executor.submit(
            run_flowpose_job,
            self.flowpose_cache,
            self.flowpose_kwargs,
            self.state.sam_result,
            self.args,
            self.camera_extrinsic.matrix,
        )
        # print(self.state)

        self.state.status = (
            f"Retry {self.state.retry_attempts}: FlowPose running"
            if retry
            else "FlowPose running"
        )

    def collect_inference_results(self) -> None:
        """Consume finished SAM3/FlowPose jobs without blocking the camera loop."""
        s = self.state

        if s.sam_future is not None:
            sam_futures = [s.sam_future]
            s.sam_result, s.sam_overlay, s.flowpose_overlay, s.status = (
                collect_sam3_results(
                    sam_futures,
                    s.sam_result,
                    s.sam_overlay,
                    s.flowpose_overlay,
                    s.status,
                )
            )
            if s.sam_future.done():
                s.sam_future = None

        if s.flowpose_future is not None:
            flowpose_futures = [s.flowpose_future]
            s.flowpose_overlay, s.base_targets, s.status = collect_flowpose_results(
                flowpose_futures,
                s.flowpose_overlay,
                s.base_targets,
                s.status,
                self.ros_bridge,
                self.camera_extrinsic.matrix,
                self.args.show_base_targets,
                self.args,
            )
            if s.flowpose_future.done():
                s.flowpose_future = None

    def publish_grasp(self) -> None:
        """Send the latest target to IK and start recovery on grip failure."""
        if self.robot_actions is None:
            self.state.status = "Robot action service unavailable"
            return
        if self.grasp_future is not None and not self.grasp_future.done():
            self.state.status = "Grasp action already running"
            return
        if self.ik_publisher is not None:
            self.ik_publisher.clear_stop()

        targets = list(self.state.base_targets)
        self.grasp_future = self.action_executor.submit(
            self.robot_actions.publish_grasp,
            targets,
        )
        self.state.status = "Grasp action running"

    def collect_grasp_result(self) -> None:
        if self.grasp_future is None or not self.grasp_future.done():
            return
        try:
            result = self.grasp_future.result()
        except Exception as exc:  # noqa: BLE001
            self.state.status = f"Grasp action failed: {exc}"
            print(f"[request_ik_tester] {self.state.status}", flush=True)
            self.grasp_future = None
            return
        self.grasp_future = None
        self.finish_grasp_result(result)

    def finish_grasp_result(self, result) -> None:
        if self.state.paused or (
            self.ik_publisher is not None and self.ik_publisher.stop_requested()
        ):
            self.state.grasp_confirmed = False
            self.state.grasp_confirmed_hand = None
            self.state.grasp_confirmed_label = None
            self.state.status = f"Paused by S; grasp result held: {result.status}"
            return

        self.state.status = result.status

        if result.failed_min_limit or not result.ok:
            self.state.grasp_confirmed = False
            self.state.grasp_confirmed_hand = None
            self.state.grasp_confirmed_label = None
            self.state.last_gripper_hand = result.failed_hand
            self.start_grip_failure_recovery(
                result.status,
                failed_hand=result.failed_hand,
            )
        else:
            self.state.retry_attempts = 0
            if result.grasp_confirmed:
                self.state.grasp_confirmed = True
                self.state.grasp_confirmed_hand = result.grasp_hand
                self.state.grasp_confirmed_label = result.object_label
                self.state.last_gripper_hand = result.grasp_hand
                self.auto_put_after_confirmed_grasp()

    def auto_put_after_confirmed_grasp(self) -> None:
        """Run fixed put immediately when the gripper confirms a successful grasp."""
        if not bool(getattr(self.args, "enable_put_after_grasp", True)):
            return
        self.publish_put()

    def publish_put(self) -> None:
        """Place only after a successful grip result confirmed object contact."""
        if not bool(getattr(self.args, "enable_put_after_grasp", True)):
            self.state.status = (
                "Put-after-grasp disabled by --enable-put-after-grasp false"
            )
            return

        if self.robot_actions is None:
            self.state.status = "Robot action service unavailable"
            return

        try:
            result = self.robot_actions.publish_put(
                grasp_confirmed=self.state.grasp_confirmed,
                hand=self.state.grasp_confirmed_hand,
                object_label=self.state.grasp_confirmed_label,
            )
        except Exception as exc:  # noqa: BLE001
            self.state.status = f"Auto put failed: {exc}"
            print(f"[put] {self.state.status}", flush=True)
            return

        self.state.status = result.status
        if "GRASP_DROPPED" in result.status:
            self.start_drop_recovery(result.status, result.grasp_hand)
        elif result.ok:
            self.state.grasp_confirmed = False
            self.state.grasp_confirmed_hand = None
            self.state.grasp_confirmed_label = None

    def start_drop_recovery(self, status: str, hand: str | None) -> None:
        """Apply S-style stop, return home smoothly, then schedule the A workflow."""
        s = self.state
        drop_hand = hand or s.grasp_confirmed_hand or (
            "left" if self.args.ik_hand == "left" else "right"
        )
        s.paused = True
        s.pipeline_stage = PipelineStage.IDLE
        self.reset_retry()
        s.grasp_confirmed = False
        s.grasp_confirmed_hand = None
        s.grasp_confirmed_label = None
        s.home_needs_sync.add(drop_hand)
        if self.ik_publisher is not None:
            self.ik_publisher.request_stop()
            # The put worker has observed the stop before returning.  Clear only
            # for the measured-start HOME trajectory while the UI stays paused.
            self.ik_publisher.clear_stop()
        s.drop_recovery_futures = [
            self.action_executor.submit(
                self.robot_actions.publish_home,
                drop_hand,
                fresh_measured_start=True,
            ),
            self.action_executor.submit(
                self.robot_actions.send_gripper,
                "release",
                drop_hand,
            ),
        ]
        s.status = f"Paused after drop; smoothly returning {drop_hand} home: {status}"
        print(f"[grip_drop] {s.status}", flush=True)

    def update_drop_recovery(self) -> None:
        """Resume after drop recovery and arm the same workflow as pressing A."""
        s = self.state
        if not s.drop_recovery_futures:
            return
        if any(not future.done() for future in s.drop_recovery_futures):
            return
        failed = []
        for future in s.drop_recovery_futures:
            try:
                print(f"[grip_drop] recovery result: {future.result()}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed.append(str(exc))
        s.drop_recovery_futures = []
        if failed:
            s.status = "Drop recovery failed; remaining paused: " + "; ".join(failed)
            return
        s.paused = False
        s.home_needs_sync.clear()
        s.drop_regrasp_pending = True
        s.status = "Drop recovery complete; restarting A grasp pipeline"

    def advance_drop_regrasp(self) -> None:
        if not self.state.drop_regrasp_pending or self.state.paused:
            return
        self.state.drop_regrasp_pending = False
        # This is the same capture -> SAM3 -> FlowPose -> grasp chain as KEY_CAPTURE.
        self.start_pipeline(grasp_on_done=True)

    def start_grip_failure_recovery(
        self,
        failed_status: str,
        *,
        failed_hand: str | None = None,
    ) -> None:
        """Return the failed hand home and release the gripper in parallel."""
        s = self.state
        s.pipeline_stage = PipelineStage.IDLE

        if s.recovery_futures:
            s.status = "Grip failure recovery already running"
            return

        s.retry_attempts += 1
        max_attempts = max(int(getattr(self.args, "grip_retry_max_attempts", 3)), 0)
        retry_enabled = bool(getattr(self.args, "grip_retry_loop", True))
        s.retry_will_regrasp = retry_enabled and (
            max_attempts == 0 or s.retry_attempts <= max_attempts
        )

        hand = failed_hand or ("left" if self.args.ik_hand == "left" else "right")
        s.last_gripper_hand = hand
        if self.robot_actions is None:
            s.status = "Robot action service unavailable"
            return

        s.recovery_futures = [
            self.action_executor.submit(
                self.robot_actions.publish_home,
                hand,
            ),
            self.action_executor.submit(
                self.robot_actions.send_gripper,
                "release",
                hand,
            ),
        ]
        s.retry_stage = RetryStage.RECOVERY

        retry_text = (
            f"retry {s.retry_attempts}"
            if s.retry_will_regrasp
            else (
                "retry loop disabled"
                if not retry_enabled
                else f"retry limit reached ({max_attempts})"
            )
        )
        s.status = f"Grip failed; returning {hand} home and releasing in parallel; {retry_text}"
        print(f"[grip_retry] {s.status}", flush=True)

    def _collect_recovery_results(self) -> bool:
        """Consume finished recovery actions; return True when all are done."""
        s = self.state
        if not s.recovery_futures:
            return True

        pending: list[Future] = []
        for future in s.recovery_futures:
            if not future.done():
                pending.append(future)
                continue
            try:
                print(f"[grip_retry] recovery result: {future.result()}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[grip_retry] recovery failed: {exc}", flush=True)

        s.recovery_futures = pending
        return not pending

    def update_recovery(self) -> None:
        """Finish Home/Release recovery and clean up its Futures.

        This method never starts SAM3, FlowPose, or another grasp. It only
        manages the safety recovery that follows a failed grasp.
        """
        s = self.state

        if s.retry_stage is not RetryStage.RECOVERY:
            return

        if not self._collect_recovery_results():
            return

        # Recovery is complete. Automatic re-planning is handled separately.
        s.retry_stage = RetryStage.IDLE
        s.replan_pending = s.retry_will_regrasp

        if s.replan_pending:
            s.status = (
                f"Recovery done; retry {s.retry_attempts} ready to replan"
            )
        else:
            s.status = "Recovery done; grip retry limit reached"

        print(f"[grip_retry] {s.status}", flush=True)

    def advance_replan_state(self, bundle) -> None:
        """Advance automatic re-planning after recovery.

        Commenting out this method call disables automatic recapture, SAM3,
        FlowPose and re-grasp, while recovery itself continues to work.
        """
        s = self.state

        if s.replan_pending:
            s.replan_pending = False
            s.retry_stage = RetryStage.RECAPTURE

        if s.retry_stage is RetryStage.RECAPTURE:
            # Do not overlap a fresh retry with an older inference job.
            if is_pending(s.sam_future) or is_pending(s.flowpose_future):
                return
            s.retry_stage = RetryStage.SAM3
            self.submit_sam3(bundle, retry=True)
            return

        if s.retry_stage is RetryStage.SAM3:
            if is_pending(s.sam_future):
                return
            if s.sam_result is None:
                s.retry_stage = RetryStage.IDLE
                s.status = "Retry SAM3 failed; stopping automatic replan"
                return

            s.retry_stage = RetryStage.FLOWPOSE
            self.submit_flowpose(retry=True)
            print(f"[grip_retry] {s.status}", flush=True)
            return

        if s.retry_stage is RetryStage.FLOWPOSE:
            if is_pending(s.flowpose_future):
                return
            if not s.base_targets:
                s.retry_stage = RetryStage.IDLE
                s.status = "Retry FlowPose produced no target; stopping automatic replan"
                return

            s.retry_stage = RetryStage.IDLE
            print(
                f"[grip_retry] Retry {s.retry_attempts}: fresh FlowPose target ready; grasping",
                flush=True,
            )
            self.publish_grasp()

    def start_pipeline(self, *, grasp_on_done: bool = True) -> None:
        """Start one-key capture -> SAM3 -> FlowPose, optionally followed by grasp."""
        s = self.state
        if (
            s.pipeline_stage is not PipelineStage.IDLE
            or is_pending(s.sam_future)
            or is_pending(s.flowpose_future)
            or s.retry_stage is not RetryStage.IDLE
            or bool(s.recovery_futures)
        ):
            s.status = "Pipeline already running"
            return

        bundle = self.camera.read_latest()
        if bundle is None:
            s.status = "Pipeline stopped: failed to capture fresh frame"
            return

        self.reset_retry(reset_attempts=True)
        s.pipeline_grasp_on_done = grasp_on_done
        s.pipeline_stage = PipelineStage.SAM3
        self.submit_sam3(bundle)
        if s.sam_future is None:
            s.pipeline_stage = PipelineStage.IDLE
            s.pipeline_grasp_on_done = True

    def advance_pipeline(self) -> None:
        """Advance the one-key workflow as soon as each async result is ready."""
        s = self.state

        if s.pipeline_stage is PipelineStage.SAM3:
            if is_pending(s.sam_future):
                return
            if s.sam_result is None:
                s.pipeline_stage = PipelineStage.IDLE
                s.status = "Pipeline stopped: SAM3 produced no result"
                s.pipeline_grasp_on_done = True
                return

            s.pipeline_stage = PipelineStage.FLOWPOSE
            self.submit_flowpose()
            if s.flowpose_future is None:
                s.pipeline_stage = PipelineStage.IDLE
            return

        if s.pipeline_stage is PipelineStage.FLOWPOSE:
            if is_pending(s.flowpose_future):
                return
            s.pipeline_stage = PipelineStage.IDLE
            if not s.base_targets:
                s.status = "Pipeline stopped: FlowPose produced no target"
                s.pipeline_grasp_on_done = True
                return

            if s.pipeline_grasp_on_done:
                s.pipeline_grasp_on_done = True
                self.publish_grasp()
            else:
                s.pipeline_grasp_on_done = True
                s.status = f"Pipeline done: FlowPose produced {len(s.base_targets)} target(s)"

    def render(self, bundle) -> None:
        """Render the live camera, SAM3 overlay and FlowPose overlay."""
        live_image = bundle.color_image
        if bool(getattr(self.args, "sam3_roi_filter", True)):
            roi_xyxy = getattr(self.args, "sam3_roi_xyxy", None)
            if roi_xyxy is not None:
                live_image = bundle.color_image.copy()
                x_min, y_min, x_max, y_max = roi_xyxy
                cv2.rectangle(
                    live_image,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        live_panel = draw_live_hud(
            live_image,
            f"frame={bundle.frame_id} depth_scale={bundle.depth_scale:.6f}",
        )
        dashboard = make_dashboard(
            live_panel,
            self.state.sam_overlay,
            self.state.flowpose_overlay,
            status=self.state.status,
            prompt=self.args.prompts,
            sam_pending=int(is_pending(self.state.sam_future)),
            flowpose_pending=int(is_pending(self.state.flowpose_future)),
            put_enabled=bool(getattr(self.args, "enable_put_after_grasp", True)),
        )
        cv2.imshow(DASHBOARD_WINDOW, dashboard)

    def publish_manual_home(self, hand: str) -> None:
        if (self.grasp_future is not None or self.gripper_future is not None
                or self.manual_home_future is not None or self.state.recovery_futures
                or self.state.drop_recovery_futures
                or self.state.pipeline_stage is not PipelineStage.IDLE):
            self.state.status = "HOME deferred: another robot action is still running"
            return
        self.manual_home_hand = hand
        self.manual_home_future = self.action_executor.submit(
            self.robot_actions.publish_home, hand,
            fresh_measured_start=hand in self.state.home_needs_sync,
        )
        self.state.status = f"{hand} HOME: synchronizing measured start and returning"

    def collect_manual_home_result(self) -> None:
        if self.manual_home_future is None or not self.manual_home_future.done():
            return
        try:
            status = self.manual_home_future.result()
        except Exception as exc:  # noqa: BLE001
            self.state.status = f"HOME deferred: {exc}"
            print(f"[home] {self.state.status}", flush=True)
        else:
            if (not self.state.paused and self.ik_publisher is not None
                    and not self.ik_publisher.stop_requested()):
                self.state.status = status
                self.state.home_needs_sync.discard(self.manual_home_hand)
        self.manual_home_future = None
        self.manual_home_hand = None

    def handle_key(self, key: int, bundle) -> bool:
        """Handle one keyboard command. Return False to exit the app."""
        key = normalize_key(key)

        if key in KEY_QUIT:
            return False

        if key == KEY_PAUSE:
            if self.state.paused and (
                is_pending(self.grasp_future)
                or is_pending(self.manual_home_future)
                or any(is_pending(f) for f in self.state.recovery_futures)
                or any(is_pending(f) for f in self.state.drop_recovery_futures)
            ):
                self.state.status = "Paused by S: waiting for interrupted actions to exit"
                return True
            self.state.paused = not self.state.paused
            if self.state.paused:
                self.state.home_needs_sync.update({"left", "right"})
                # Resume must never restart the pre-pause automatic pipeline.
                self.state.pipeline_stage = PipelineStage.IDLE
                self.reset_retry()
                self.gripper_interrupted = self.gripper_future is not None
                self.state.grasp_confirmed = False
                self.state.grasp_confirmed_hand = None
                self.state.grasp_confirmed_label = None
                if self.ik_publisher is not None:
                    self.ik_publisher.request_stop()
                self.state.status = "Paused by S: target publishing stop requested"
            else:
                # Consume an interrupted result while stop is still latched;
                # it must not be interpreted as a fresh successful grasp.
                self.collect_grasp_result()
                self.collect_manual_home_result()
                if self.ik_publisher is not None:
                    self.ik_publisher.clear_stop()
                self.state.status = "Resumed by S"
            return True

        if self.state.paused or is_pending(self.manual_home_future):
            return True

        if key == KEY_CAPTURE:
            if self.auto_pipeline_on_a():
                self.start_pipeline(grasp_on_done=True)
            else:
                fresh_bundle = self.camera.read_latest()
                if fresh_bundle is None:
                    self.state.status = "Capture failed: no fresh camera frame"
                else:
                    self.reset_retry(reset_attempts=True)
                    self.submit_sam3(fresh_bundle)
        elif key == KEY_PERCEPTION_PIPELINE:
            self.start_pipeline(grasp_on_done=False)
        elif key == KEY_FLOWPOSE and not self.auto_pipeline_on_a():
            self.reset_retry()
            self.submit_flowpose()
        elif key == KEY_TARGET and not self.auto_pipeline_on_a():
            self.reset_retry(reset_attempts=True)
            self.publish_grasp()
        elif key == KEY_RIGHT_HOME:
            self.state.last_gripper_hand = "right"
            if self.robot_actions is not None:
                self.publish_manual_home("right")
        elif key == KEY_LEFT_HOME:
            self.state.last_gripper_hand = "left"
            if self.robot_actions is not None:
                self.publish_manual_home("left")
        elif key == KEY_GRIP:
            self.send_gripper("grip")
        elif key == KEY_RELEASE:
            self.state.grasp_confirmed = False
            self.state.grasp_confirmed_hand = None
            self.send_gripper("release")

        return True

    def run(self) -> int:
        """Run the real-time event loop until Q or ESC is pressed."""
        cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)

        while True:
            bundle = self.camera.read_latest()
            if bundle is None:
                print("[camera] failed to read frame; retrying...", flush=True)
                continue

            if self.state.paused:
                if not self.state.status.startswith("Paused by S"):
                    self.state.status = "Paused by S"
            else:
                self.collect_inference_results()
                self.advance_pipeline()
            self.collect_grasp_result()
            self.collect_manual_home_result()
            self._collect_gripper_result()
            self.update_drop_recovery()
            if not self.state.paused:
                self.update_recovery()
                self.advance_drop_regrasp()
           # self.advance_replan_state(bundle)
            self.render(bundle)

            if not self.handle_key(cv2.waitKey(1), bundle):
                return 0


def main() -> int:
    """Application entry point."""
    app = GraspDemoApp(parse_args())
    try:
        app.open()
        return app.run()
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
