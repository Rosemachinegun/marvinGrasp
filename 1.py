#!/usr/bin/env python3
"""Tablet web UI and its thread-safe bridge to the application task loop.

This module deliberately knows nothing about SAM3, FlowPose, ROS2, IK, or robot
actions. Browser callbacks only enqueue commands; the existing application
loop remains the sole owner of perception and robot state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from queue import Empty, SimpleQueue
from threading import Lock

import cv2
import numpy as np


# =============================================================================
# Commands
# =============================================================================


class TabletCommand(str, Enum):
    """Commands understood by the existing application loop."""

    PERCEIVE = "perceive"
    GRASP = "grasp"
    VOICE = "voice"
    HOME_LEFT = "home_left"
    HOME_RIGHT = "home_right"
    STOP = "stop"


# =============================================================================
# Snapshot
# =============================================================================


@dataclass(frozen=True, slots=True)
class TabletSnapshot:
    image_rgb: np.ndarray | None
    sam_rgb: np.ndarray | None
    flowpose_rgb: np.ndarray | None
    base_target_text: str
    status: str
    activity: str


# =============================================================================
# Thread-safe bridge
# =============================================================================


class TabletTaskLoopBridge:
    """Small shared-state boundary between Gradio and ``GraspDemoApp.run``."""

    def __init__(self) -> None:
        self._commands: SimpleQueue[TabletCommand] = SimpleQueue()
        self._lock = Lock()

        # Raw RGB is still stored for compatibility with the existing backend,
        # but the compact tablet UI does not display it.
        self._image_rgb: np.ndarray | None = None
        self._sam_rgb: np.ndarray | None = None
        self._flowpose_rgb: np.ndarray | None = None
        self._base_target_text = ""

        self._status = "Connecting to robot vision..."
        self._activity = "Tablet control ready"

    def request(
        self,
        command: TabletCommand,
    ) -> tuple[str, str]:
        """Queue a tablet command without directly touching robot state."""

        messages = {
            TabletCommand.PERCEIVE:
                "Requested: SAM3 + FlowPose",

            TabletCommand.GRASP:
                "Requested: Autonomous grasp",

            TabletCommand.VOICE:
                "Requested: 4-second voice command",

            TabletCommand.HOME_LEFT:
                "Requested: Left arm home",

            TabletCommand.HOME_RIGHT:
                "Requested: Right arm home",

            TabletCommand.STOP:
                "Requested: Stop current task",
        }

        activity = messages[command]

        self._commands.put(command)

        with self._lock:
            self._activity = activity
            return self._status, self._activity

    def drain_commands(self) -> list[TabletCommand]:
        """Drain all pending browser commands for the main application loop."""

        commands: list[TabletCommand] = []

        while True:
            try:
                commands.append(
                    self._commands.get_nowait()
                )
            except Empty:
                return commands

    @staticmethod
    def _to_browser_rgb(
        image_bgr: np.ndarray | None,
    ) -> np.ndarray | None:
        """Convert OpenCV BGR frame to RGB for browser display."""

        if image_bgr is None:
            return None

        return cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        ).copy()

    def publish(
        self,
        image_bgr: np.ndarray,
        status: str,
        sam_bgr: np.ndarray | None = None,
        flowpose_bgr: np.ndarray | None = None,
        base_target_text: str = "",
    ) -> None:
        """Publish perception frames for the next browser polling cycle."""

        image_rgb = self._to_browser_rgb(image_bgr)
        sam_rgb = self._to_browser_rgb(sam_bgr)
        flowpose_rgb = self._to_browser_rgb(flowpose_bgr)

        with self._lock:
            self._image_rgb = image_rgb
            self._sam_rgb = sam_rgb
            self._flowpose_rgb = flowpose_rgb
            self._base_target_text = str(base_target_text)
            self._status = str(status)

    def set_activity(
        self,
        activity: str,
    ) -> None:
        """Update the current activity text displayed on the tablet."""

        with self._lock:
            self._activity = str(activity)

    def snapshot(self) -> TabletSnapshot:
        """Return a thread-safe copy of the latest UI state."""

        with self._lock:
            image = (
                None
                if self._image_rgb is None
                else self._image_rgb.copy()
            )

            sam = (
                None
                if self._sam_rgb is None
                else self._sam_rgb.copy()
            )

            flowpose = (
                None
                if self._flowpose_rgb is None
                else self._flowpose_rgb.copy()
            )

            return TabletSnapshot(
                image_rgb=image,
                sam_rgb=sam,
                flowpose_rgb=flowpose,
                base_target_text=self._base_target_text,
                status=self._status,
                activity=self._activity,
            )


# =============================================================================
# Compact single-screen visual style
# =============================================================================


TABLET_CSS = r"""
:root {
    --page: #f1f2ef;
    --surface: #fbfbf8;
    --surface-soft: #f5f6f2;
    --surface-deep: #eceee9;

    --ink: #111411;
    --ink-soft: #3f4540;
    --muted: #7a817b;
    --muted-2: #a2a7a2;

    --mint: #10c89c;
    --mint-soft: #dff8ef;
    --mint-line: rgba(16, 200, 156, .30);

    --line: rgba(17, 20, 17, .10);
    --line-strong: rgba(17, 20, 17, .16);

    --danger: #b94f43;
    --danger-soft: #fff0ee;

    --sans:
        Inter,
        "Helvetica Neue",
        "Noto Sans SC",
        "PingFang SC",
        Arial,
        system-ui,
        sans-serif;

    --mono:
        "SFMono-Regular",
        "JetBrains Mono",
        Consolas,
        monospace;
}

/* =============================================================================
   Gento-inspired page shell
   ============================================================================= */

html,
body {
    margin: 0;
    width: 100%;
    height: 100%;
    background: var(--page);
}

body {
    overflow: hidden;
}

.gradio-container {
    position: relative;
    min-height: 100vh;
    color: var(--ink) !important;
    font-family: var(--sans) !important;
    background:
        radial-gradient(circle at 82% -12%, rgba(16, 200, 156, .08), transparent 30rem),
        linear-gradient(180deg, #f7f8f5 0%, var(--page) 100%) !important;
}

/* Remove the old engineering-grid / noise treatment. */
.gradio-container::before,
.gradio-container::after {
    display: none !important;
}

.gradio-container > .main,
.gradio-container .main {
    position: relative;
    z-index: 1;
    width: min(1400px, calc(100vw - 28px)) !important;
    max-width: 1400px !important;
    height: 100vh;
    margin: 0 auto !important;
    padding: 12px 0 10px !important;
    box-sizing: border-box;
}

footer {
    display: none !important;
}

#tablet-shell {
    height: 100%;
    gap: 10px !important;
}

/* =============================================================================
   Header — clean product-page treatment
   ============================================================================= */

#compact-header {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 68px;
    padding: 11px 18px 11px 20px;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 22px;
    background: rgba(251, 251, 248, .94);
    box-shadow: 0 10px 30px rgba(20, 25, 20, .055);
    backdrop-filter: blur(14px);
}

/* A restrained Gento-like brand accent instead of decorative rings. */
#compact-header::after {
    content: "";
    position: absolute;
    right: 0;
    top: 0;
    width: 190px;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--mint));
    pointer-events: none;
}

.brand-block {
    min-width: 0;
}

.brand-top {
    display: flex;
    align-items: center;
    gap: 9px;
    color: var(--muted);
    font-family: var(--sans);
    font-size: 9px;
    font-weight: 600;
    letter-spacing: .14em;
    white-space: nowrap;
}

.live-dot {
    width: 10px;
    height: 10px;
    flex: 0 0 auto;
    border-radius: 3px 7px 3px 7px;
    background: var(--mint);
    box-shadow: 0 0 0 4px rgba(16, 200, 156, .08);
    transform: rotate(8deg);
}

.brand-title {
    margin: 4px 0 0;
    color: var(--ink);
    font-family: var(--sans);
    font-size: clamp(24px, 2.5vw, 34px);
    font-weight: 420;
    line-height: .98;
    letter-spacing: -.045em;
}

.pipeline {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 6px;
}

.pipeline span {
    padding: 6px 9px;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--ink-soft);
    background: rgba(255, 255, 255, .72);
    font-family: var(--sans);
    font-size: 8px;
    font-weight: 550;
    letter-spacing: .055em;
    white-space: nowrap;
}

.pipeline span:last-child {
    border-color: var(--mint-line);
    color: #087c62;
    background: var(--mint-soft);
}

.pipeline b {
    color: var(--muted-2);
    font-size: 10px;
    font-weight: 400;
}

/* =============================================================================
   Workspace
   ============================================================================= */

#workspace {
    flex: 1 1 auto;
    min-height: 0;
    gap: 10px !important;
    align-items: stretch !important;
}

/* =============================================================================
   Shared panels
   ============================================================================= */

.art-panel {
    position: relative;
    overflow: hidden;
    min-width: 0 !important;
    border: 1px solid var(--line) !important;
    border-radius: 22px !important;
    background: rgba(251, 251, 248, .96) !important;
    box-shadow: 0 12px 34px rgba(20, 25, 20, .05) !important;
}

.art-panel::before {
    content: none !important;
}

/* =============================================================================
   Perception panel
   ============================================================================= */

#perception-panel {
    padding: 14px !important;
}

.panel-heading {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 15px;
    margin-bottom: 9px;
}

.panel-kicker {
    color: #088a6c;
    font-family: var(--sans);
    font-size: 8px;
    font-weight: 650;
    letter-spacing: .14em;
    text-transform: uppercase;
}

.panel-title {
    margin: 2px 0 0;
    color: var(--ink);
    font-family: var(--sans);
    font-size: 24px;
    font-weight: 430;
    line-height: 1.02;
    letter-spacing: -.035em;
}

.panel-meta {
    color: var(--muted);
    font-family: var(--sans);
    font-size: 8px;
    font-weight: 550;
    letter-spacing: .08em;
    text-align: right;
}

#perception-results {
    gap: 9px !important;
}

.result-card {
    min-width: 0 !important;
    padding: 10px !important;
    border: 1px solid var(--line) !important;
    border-radius: 18px !important;
    background: var(--surface-soft) !important;
    box-shadow: none !important;
}

.result-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 7px;
}

.result-name {
    color: var(--ink);
    font-family: var(--sans);
    font-size: 17px;
    font-weight: 520;
    letter-spacing: -.02em;
}

.result-type {
    color: #0b9a78;
    font-family: var(--sans);
    font-size: 7px;
    font-weight: 650;
    letter-spacing: .10em;
}

.result-view {
    overflow: hidden;
    border: 1px solid rgba(17, 20, 17, .07) !important;
    border-radius: 14px !important;
    background:
        radial-gradient(circle at 50% 38%, #ffffff 0%, #f5f6f2 68%, #eceee9 100%) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, .9) !important;
}

.result-view img {
    width: 100% !important;
    height: 100% !important;
    max-height: 300px !important;
    border-radius: 12px !important;
    object-fit: contain !important;
    filter: saturate(.92) contrast(1.015);
}

.result-view .image-container,
.result-view > div {
    min-height: 275px !important;
    height: 275px !important;
    max-height: 275px !important;
}

#base-targets {
    margin-top: 5px !important;
}

#base-targets textarea {
    min-height: 67px !important;
    color: var(--ink-soft) !important;
    background: #ffffff !important;
    border-color: var(--line) !important;
    font-family: var(--mono) !important;
    font-size: 10px !important;
    line-height: 1.45 !important;
}

/* =============================================================================
   Control panel
   ============================================================================= */

#control-panel {
    padding: 14px !important;
    gap: 8px !important;
}

.control-heading {
    margin-bottom: 3px;
}

.control-heading .panel-title {
    font-size: 22px;
}

.flow-strip {
    display: grid;
    grid-template-columns: 1fr auto 1fr auto 1fr;
    align-items: center;
    gap: 5px;
    margin: 4px 0 6px;
    padding: 8px 9px;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--surface-soft);
    color: var(--muted);
    font-family: var(--sans);
    font-size: 7px;
    font-weight: 600;
    letter-spacing: .055em;
    text-align: center;
}

.flow-strip b {
    color: var(--mint);
    font-size: 10px;
    font-weight: 500;
}

/* =============================================================================
   Buttons — same controls, Gento-style visual system
   ============================================================================= */

#grasp-btn {
    position: relative;
    min-height: 78px !important;
    border: 1px solid #101310 !important;
    border-radius: 17px !important;
    color: #ffffff !important;
    background: #111411 !important;
    box-shadow: 0 12px 24px rgba(17, 20, 17, .14) !important;
    font-size: 18px !important;
    font-weight: 650 !important;
    letter-spacing: .055em !important;
    text-shadow: none !important;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease !important;
}

#grasp-btn::before {
    content: "";
    position: absolute;
    left: 14px;
    top: 50%;
    width: 8px;
    height: 8px;
    margin-top: -4px;
    border-radius: 50%;
    background: var(--mint);
    box-shadow: 0 0 0 4px rgba(16, 200, 156, .12);
}

#grasp-btn:hover {
    transform: translateY(-2px);
    border-color: #1a1e1a !important;
    background: #1a1e1a !important;
    box-shadow: 0 16px 30px rgba(17, 20, 17, .18) !important;
}

#grasp-btn:active {
    transform: translateY(0) scale(.994);
}

#grasp-btn::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: 0;
    width: 38%;
    height: 2px;
    background: linear-gradient(90deg, var(--mint), transparent);
    transition: width .30s ease;
    pointer-events: none;
}

#grasp-btn:hover::after {
    width: 72%;
}

#voice-btn,
#perceive-btn,
#home-left-btn,
#home-right-btn {
    border: 1px solid var(--line-strong) !important;
    color: var(--ink) !important;
    background: #ffffff !important;
    box-shadow: none !important;
}

#voice-btn,
#perceive-btn {
    min-height: 49px !important;
    border-radius: 13px !important;
    font-size: 10px !important;
    font-weight: 650 !important;
    letter-spacing: .04em !important;
}

#perceive-btn {
    border-color: var(--mint-line) !important;
    background: var(--mint-soft) !important;
    color: #087c62 !important;
}

#voice-btn:hover,
#home-left-btn:hover,
#home-right-btn:hover {
    transform: translateY(-1px);
    border-color: rgba(17, 20, 17, .24) !important;
    background: #f8f9f6 !important;
}

#perceive-btn:hover {
    transform: translateY(-1px);
    border-color: rgba(16, 200, 156, .48) !important;
    background: #d3f5e9 !important;
}

#home-left-btn,
#home-right-btn {
    min-height: 43px !important;
    border-radius: 12px !important;
    font-size: 9px !important;
    font-weight: 650 !important;
    letter-spacing: .035em !important;
}

#stop-btn {
    min-height: 43px !important;
    border: 1px solid rgba(185, 79, 67, .26) !important;
    border-radius: 12px !important;
    color: var(--danger) !important;
    background: var(--danger-soft) !important;
    box-shadow: none !important;
    font-size: 9px !important;
    font-weight: 700 !important;
    letter-spacing: .055em !important;
}

#stop-btn:hover {
    transform: translateY(-1px);
    border-color: rgba(185, 79, 67, .45) !important;
    background: #ffe7e3 !important;
}

/* =============================================================================
   Status
   ============================================================================= */

#status-area {
    gap: 7px !important;
}

#robot-status,
#activity-status {
    min-width: 0 !important;
}

#robot-status textarea,
#activity-status textarea {
    min-height: 53px !important;
    height: 53px !important;
    padding: 8px 10px !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    color: var(--ink-soft) !important;
    background: var(--surface-soft) !important;
    font-family: var(--sans) !important;
    font-size: 9px !important;
    line-height: 1.42 !important;
    resize: none !important;
    box-shadow: none !important;
}

#mode-actions,
#home-actions {
    gap: 7px !important;
}

/* =============================================================================
   Gradio native tweaks
   ============================================================================= */

.gradio-container label,
.gradio-container .label-wrap {
    color: var(--muted) !important;
    font-family: var(--sans) !important;
    font-size: 8px !important;
    font-weight: 600 !important;
    letter-spacing: .035em;
}

.gradio-container button {
    overflow: hidden;
    font-family: var(--sans) !important;
    transition: transform .18s ease, border-color .18s ease, background .18s ease, box-shadow .18s ease !important;
}

.gradio-container textarea {
    scrollbar-width: thin;
    scrollbar-color: #c7cbc6 transparent;
}

/* =============================================================================
   Footer
   ============================================================================= */

.compact-footer {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin: 0 5px;
    color: var(--muted-2);
    font-family: var(--sans);
    font-size: 7px;
    font-weight: 600;
    letter-spacing: .095em;
    text-transform: uppercase;
}

/* =============================================================================
   Responsive — retain the single-workspace concept
   ============================================================================= */

@media (max-width: 1100px) {
    .gradio-container > .main,
    .gradio-container .main {
        width: calc(100vw - 18px) !important;
        padding-top: 9px !important;
    }

    #compact-header {
        min-height: 62px;
        border-radius: 18px;
    }

    .pipeline span:nth-of-type(2),
    .pipeline span:nth-of-type(4) {
        display: none;
    }

    .art-panel {
        border-radius: 18px !important;
    }

    .result-view .image-container,
    .result-view > div {
        min-height: 230px !important;
        height: 230px !important;
        max-height: 230px !important;
    }

    .result-view img {
        max-height: 230px !important;
    }
}

@media (max-width: 820px) {
    body {
        overflow-y: auto;
    }

    .gradio-container > .main,
    .gradio-container .main {
        height: auto;
        min-height: 100vh;
    }

    .pipeline {
        display: none;
    }

    #compact-header {
        min-height: 54px;
        padding: 9px 13px;
    }

    .brand-title {
        font-size: 24px;
    }

    #workspace {
        flex-direction: column !important;
    }

    .result-view .image-container,
    .result-view > div {
        min-height: 205px !important;
        height: 205px !important;
        max-height: 205px !important;
    }

    .result-view img {
        max-height: 205px !important;
    }

    #grasp-btn {
        min-height: 65px !important;
    }
}
"""


# =============================================================================
# HTML fragments
# =============================================================================


COMPACT_HEADER_HTML = r"""
<section id="compact-header">

    <div class="brand-block">

        <div class="brand-top">
            <span class="live-dot"></span>
            MARVINGRASP · EMBODIED AI CONTROL
        </div>

        <h1 class="brand-title">
            Perception into action.
        </h1>

    </div>

    <div class="pipeline">

        <span>RGB-D</span>
        <b>→</b>

        <span>SAM3</span>
        <b>→</b>

        <span>FLOWPOSE</span>
        <b>→</b>

        <span>MOTION</span>
        <b>→</b>

        <span>GRASP</span>

    </div>

</section>
"""


PERCEPTION_HEAD_HTML = r"""
<div class="panel-heading">

    <div>
        <div class="panel-kicker">
            LIVE PERCEPTION
        </div>

        <h2 class="panel-title">
            See. Understand.
        </h2>
    </div>

    <div class="panel-meta">
        SAM3 + FLOWPOSE
    </div>

</div>
"""


CONTROL_HEAD_HTML = r"""
<div class="control-heading">

    <div class="panel-kicker">
        ROBOT CONTROL
    </div>

    <h2 class="panel-title">
        Turn perception into action.
    </h2>

</div>


<div class="flow-strip">

    <span>
        01 · SEE
    </span>

    <b>→</b>

    <span>
        02 · UNDERSTAND
    </span>

    <b>→</b>

    <span>
        03 · ACT
    </span>

</div>
"""


COMPACT_FOOTER_HTML = r"""
<div class="compact-footer">

    <span>
        MARVINGRASP · TABLET CONTROL
    </span>

    <span>
        SAM3 → FLOWPOSE → MOTION
    </span>

</div>
"""


# =============================================================================
# Gradio tablet UI
# =============================================================================


def create_tablet_interface(
    bridge: TabletTaskLoopBridge,
    *,
    voice_enabled: bool = False,
):
    """Build the compact single-screen tablet browser interface.

    The raw RealSense frame is intentionally not displayed.
    Only SAM3 and FlowPose visualization results are shown.

    This function contains only UI and browser callback bindings.
    Perception and robot execution remain inside the existing application loop.
    """

    try:
        import gradio as gr

    except ImportError as exc:
        raise RuntimeError(
            "Tablet UI is enabled but Gradio is missing. "
            "Please install requirements.txt."
        ) from exc

    refresh_sec = float(
        os.getenv(
            "TASK_LOOP_UI_REFRESH_SEC",
            "0.25",
        )
    )

    def refresh():
        """Poll latest perception and state.

        Raw ``image_rgb`` is intentionally omitted from the browser output.
        """

        snap = bridge.snapshot()

        return (
            snap.sam_rgb,
            snap.flowpose_rgb,
            snap.base_target_text,
            snap.status,
            snap.activity,
        )

    with gr.Blocks(
        title="MarvinGrasp · Tablet Control",
        css=TABLET_CSS,
    ) as demo:

        with gr.Column(
            elem_id="tablet-shell",
        ):

            # =================================================================
            # Compact Header
            # =================================================================

            gr.HTML(
                COMPACT_HEADER_HTML
            )

            # =================================================================
            # Main Workspace
            #
            # Left  = SAM3 + FlowPose
            # Right = robot controls
            # =================================================================

            with gr.Row(
                equal_height=True,
                elem_id="workspace",
            ):

                # =============================================================
                # LEFT: Perception
                # =============================================================

                with gr.Column(
                    scale=7,
                    min_width=520,
                    elem_id="perception-panel",
                    elem_classes=["art-panel"],
                ):

                    gr.HTML(
                        PERCEPTION_HEAD_HTML
                    )

                    with gr.Row(
                        equal_height=True,
                        elem_id="perception-results",
                    ):

                        # -----------------------------------------------------
                        # SAM3
                        # -----------------------------------------------------

                        with gr.Column(
                            scale=1,
                            elem_classes=["result-card"],
                        ):

                            gr.HTML(
                                """
                                <div class="result-head">

                                    <span class="result-name">
                                        SAM3
                                    </span>

                                    <span class="result-type">
                                        OBJECT BOUNDARY
                                    </span>

                                </div>
                                """
                            )

                            sam_result = gr.Image(
                                label="Segmentation",
                                type="numpy",
                                interactive=False,
                                show_label=False,
                                elem_classes=["result-view"],
                            )

                        # -----------------------------------------------------
                        # FlowPose
                        # -----------------------------------------------------

                        with gr.Column(
                            scale=1,
                            elem_classes=["result-card"],
                        ):

                            gr.HTML(
                                """
                                <div class="result-head">

                                    <span class="result-name">
                                        FlowPose
                                    </span>

                                    <span class="result-type">
                                        6D ORIENTATION
                                    </span>

                                </div>
                                """
                            )

                            flowpose_result = gr.Image(
                                label="6D Pose",
                                type="numpy",
                                interactive=False,
                                show_label=False,
                                elem_classes=["result-view"],
                            )

                            base_targets = gr.Textbox(
                                label="base_link targets",
                                interactive=False,
                                lines=3,
                                max_lines=4,
                                elem_id="base-targets",
                            )

                # =============================================================
                # RIGHT: Control
                # =============================================================

                with gr.Column(
                    scale=4,
                    min_width=350,
                    elem_id="control-panel",
                    elem_classes=["art-panel"],
                ):

                    gr.HTML(
                        CONTROL_HEAD_HTML
                    )

                    # ---------------------------------------------------------
                    # Primary operation
                    # ---------------------------------------------------------

                    grasp = gr.Button(
                        "START GRASP",
                        variant="primary",
                        elem_id="grasp-btn",
                    )

                    # ---------------------------------------------------------
                    # Mode controls
                    # ---------------------------------------------------------

                    with gr.Row(
                        equal_height=True,
                        elem_id="mode-actions",
                    ):

                        perceive = gr.Button(
                            "PERCEPTION ONLY",
                            elem_id="perceive-btn",
                        )

                        if voice_enabled:
                            voice = gr.Button(
                                "VOICE GRASP · 4 S",
                                elem_id="voice-btn",
                            )

                    # ---------------------------------------------------------
                    # Status
                    # ---------------------------------------------------------

                    with gr.Row(
                        equal_height=True,
                        elem_id="status-area",
                    ):

                        status = gr.Textbox(
                            label="ROBOT STATUS",
                            value="Connecting to robot vision...",
                            interactive=False,
                            lines=2,
                            elem_id="robot-status",
                        )

                        activity = gr.Textbox(
                            label="CURRENT ACTIVITY",
                            value="Tablet control ready",
                            interactive=False,
                            lines=2,
                            elem_id="activity-status",
                        )

                    # ---------------------------------------------------------
                    # Safety / home
                    # ---------------------------------------------------------

                    with gr.Row(
                        equal_height=True,
                        elem_id="home-actions",
                    ):

                        home_left = gr.Button(
                            "LEFT · HOME",
                            elem_id="home-left-btn",
                        )

                        home_right = gr.Button(
                            "RIGHT · HOME",
                            elem_id="home-right-btn",
                        )

                        stop = gr.Button(
                            "STOP",
                            variant="stop",
                            elem_id="stop-btn",
                        )

            # =================================================================
            # Tiny footer
            # =================================================================

            gr.HTML(
                COMPACT_FOOTER_HTML
            )

        # =====================================================================
        # Browser polling
        # =====================================================================

        timer = gr.Timer(
            value=refresh_sec,
            active=True,
        )

        timer.tick(
            refresh,
            outputs=[
                sam_result,
                flowpose_result,
                base_targets,
                status,
                activity,
            ],
        )

        # =====================================================================
        # Commands
        # =====================================================================

        perceive.click(
            lambda:
                bridge.request(
                    TabletCommand.PERCEIVE
                ),
            outputs=[
                status,
                activity,
            ],
        )

        grasp.click(
            lambda:
                bridge.request(
                    TabletCommand.GRASP
                ),
            outputs=[
                status,
                activity,
            ],
        )

        if voice_enabled:
            voice.click(
                lambda:
                    bridge.request(
                        TabletCommand.VOICE
                    ),
                outputs=[
                    status,
                    activity,
                ],
            )

        home_left.click(
            lambda:
                bridge.request(
                    TabletCommand.HOME_LEFT
                ),
            outputs=[
                status,
                activity,
            ],
        )

        home_right.click(
            lambda:
                bridge.request(
                    TabletCommand.HOME_RIGHT
                ),
            outputs=[
                status,
                activity,
            ],
        )

        stop.click(
            lambda:
                bridge.request(
                    TabletCommand.STOP
                ),
            outputs=[
                status,
                activity,
            ],
        )

    return demo


# =============================================================================
# Web Service
# =============================================================================


class TabletWebService:
    """Lifecycle wrapper kept separate from perception and robot services."""

    def __init__(
        self,
        bridge: TabletTaskLoopBridge,
        host: str,
        port: int,
        voice_enabled: bool = False,
    ) -> None:
        self.bridge = bridge
        self.host = host
        self.port = int(port)
        self.voice_enabled = bool(voice_enabled)

        self.demo = None

    def start(self) -> None:
        """Start the tablet web server."""

        self.demo = create_tablet_interface(
            self.bridge,
            voice_enabled=self.voice_enabled,
        )

        self.demo.queue(
            default_concurrency_limit=1
        ).launch(
            server_name=self.host,
            server_port=self.port,
            share=False,
            prevent_thread_lock=True,
            show_error=True,
        )

        print(
            f"[tablet_ui] listening on "
            f"http://{self.host}:{self.port}",
            flush=True,
        )

        print(
            "[tablet_ui] open "
            f"http://<robot-ip>:{self.port} "
            "from the tablet browser",
            flush=True,
        )

    def close(self) -> None:
        """Stop the tablet web service."""

        if self.demo is not None:
            self.demo.close()
            self.demo = None
