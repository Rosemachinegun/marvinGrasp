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
from typing import Any, Iterable

import cv2
import numpy as np

from add.settings import VOICE_ENABLED, WEB_ENABLED


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


def format_base_link_targets(targets: Iterable[Any]) -> str:
    """Format tablet target coordinates with a compact fixed precision."""
    lines = []
    for index, target in enumerate(targets):
        xyz = ", ".join(f"{float(value):.3f}" for value in target.base_xyz)
        lines.append(f"{index}: {target.label} xyz=[{xyz}]")
    return "\n".join(lines)


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
                "Requested: 2-second voice command",

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
    --ui-scale: 1.5;

    --page: #f1f2ef;
    --surface: #fbfbf8;
    --surface-soft: #f5f6f2;
    --surface-deep: #eceee9;

    --ink: #111411;
    --ink-soft: #3f4540;
    --muted: #7a817b;
    --muted-2: #a2a7a2;

    --brand-blue: #0d93d8;
    --brand-blue-soft: #e7f4fc;
    --brand-blue-line: rgba(13, 147, 216, .30);

    --line: rgba(17, 20, 17, .10);
    --line-strong: rgba(17, 20, 17, .16);

    --danger: #b94f43;
    --danger-soft: #fff0ee;

    /* White + cobalt gallery palette (non-button UI only). */
    --art-page: #ffffff;
    --art-surface: #ffffff;
    --art-surface-soft: #f5f8fd;
    --art-surface-deep: #eaf0f8;

    --art-ink: #0b1f3a;
    --art-ink-soft: #314966;
    --art-muted: #687d96;
    --art-muted-2: #9aabba;

    --art-blue: #002fa7;
    --art-blue-soft: #edf3ff;
    --art-blue-line: rgba(0, 47, 167, .24);

    --art-line: rgba(24, 58, 104, .10);
    --art-line-strong: rgba(24, 58, 104, .16);

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
    background: var(--art-page);
}

body {
    overflow: hidden;
}

.gradio-container {
    position: relative;
    min-height: 100vh;
    color: var(--art-ink) !important;
    font-family: var(--sans) !important;
    background:
        radial-gradient(circle at 82% -12%, rgba(0, 47, 167, .055), transparent 30rem),
        linear-gradient(180deg, #ffffff 0%, #ffffff 100%) !important;
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
    width: min(1400px, calc((100vw - 28px) / var(--ui-scale))) !important;
    max-width: 1400px !important;
    height: calc(100vh / var(--ui-scale));
    zoom: var(--ui-scale);
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
    border: 1px solid var(--art-line);
    border-radius: 22px;
    background: rgba(255, 255, 255, .97);
    box-shadow: 0 10px 30px rgba(18, 48, 92, .055);
    backdrop-filter: blur(14px);
}

/* A restrained KernelMind blue brand accent instead of decorative rings. */
#compact-header::after {
    content: "";
    position: absolute;
    right: 0;
    top: 0;
    width: 190px;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--art-blue));
    pointer-events: none;
}

.brand-lockup {
    display: flex;
    align-items: center;
    min-width: 0;
    gap: 14px;
}

.brand-logo-wrap {
    display: flex;
    align-items: center;
    flex: 0 0 auto;
}

.brand-logo {
    display: block;
    width: 169.43px;
    height: auto;
    max-height: 45.79px;
    object-fit: contain;
}

.brand-divider {
    width: 1px;
    height: 36px;
    flex: 0 0 auto;
    background: var(--art-line-strong);
}

.brand-block {
    min-width: 0;
}

.brand-top {
    display: flex;
    align-items: center;
    gap: 9px;
    color: var(--art-muted);
    font-family: var(--sans);
    font-size: 10.3px;
    font-weight: 600;
    letter-spacing: .14em;
    white-space: nowrap;
}

.live-dot {
    width: 10px;
    height: 10px;
    flex: 0 0 auto;
    border-radius: 3px 7px 3px 7px;
    background: var(--art-blue);
    box-shadow: 0 0 0 4px rgba(0, 47, 167, .07);
    transform: rotate(8deg);
}

.brand-title {
    margin: 4px 0 0;
    color: var(--art-ink);
    font-family: var(--sans);
    font-size: clamp(25.92px, 2.65vw, 36.72px);
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
    border: 1px solid var(--art-line);
    border-radius: 999px;
    color: var(--art-ink-soft);
    background: rgba(255, 255, 255, .72);
    font-family: var(--sans);
    font-size: 9.16px;
    font-weight: 550;
    letter-spacing: .055em;
    white-space: nowrap;
}

.pipeline span:last-child {
    border-color: var(--art-blue-line);
    color: #002fa7;
    background: var(--art-blue-soft);
}

.pipeline b {
    color: var(--art-muted-2);
    font-size: 11.45px;
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
    border: 1px solid var(--art-line) !important;
    border-radius: 22px !important;
    background: rgba(255, 255, 255, .985) !important;
    box-shadow: 0 12px 34px rgba(18, 48, 92, .05) !important;
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
    color: #174fc9;
    font-family: var(--sans);
    font-size: 9.16px;
    font-weight: 650;
    letter-spacing: .14em;
    text-transform: uppercase;
}

.panel-title {
    margin: 2px 0 0;
    color: var(--art-ink);
    font-family: var(--sans);
    font-size: 27.48px;
    font-weight: 430;
    line-height: 1.02;
    letter-spacing: -.035em;
}

.panel-meta {
    color: var(--art-muted);
    font-family: var(--sans);
    font-size: 9.16px;
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
    border: 1px solid var(--art-line) !important;
    border-radius: 18px !important;
    background: var(--art-surface-soft) !important;
    box-shadow: none !important;
}

.result-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 7px;
}

.result-name {
    color: var(--art-ink);
    font-family: var(--sans);
    font-size: 19.46px;
    font-weight: 520;
    letter-spacing: -.02em;
}

.result-type {
    color: var(--art-blue);
    font-family: var(--sans);
    font-size: 8.01px;
    font-weight: 650;
    letter-spacing: .10em;
}

.result-view {
    overflow: hidden;
    border: 1px solid rgba(24, 58, 104, .075) !important;
    border-radius: 14px !important;
    background:
        radial-gradient(circle at 50% 38%, #ffffff 0%, #f7faff 68%, #edf2fa 100%) !important;
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
    color: var(--art-ink-soft) !important;
    background: #ffffff !important;
    border-color: var(--art-line) !important;
    font-family: var(--mono) !important;
    font-size: 11.45px !important;
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
    font-size: 25.19px;
}

.flow-strip {
    display: grid;
    grid-template-columns: 1fr auto 1fr auto 1fr;
    align-items: center;
    gap: 5px;
    margin: 4px 0 6px;
    padding: 8px 9px;
    border: 1px solid var(--art-line);
    border-radius: 12px;
    background: var(--art-surface-soft);
    color: var(--art-muted);
    font-family: var(--sans);
    font-size: 8.01px;
    font-weight: 600;
    letter-spacing: .055em;
    text-align: center;
}

.flow-strip b {
    color: var(--art-blue);
    font-size: 11.45px;
    font-weight: 500;
}

/* =============================================================================
   Buttons — same controls, KernelMind blue visual system
   ============================================================================= */

#grasp-btn {
    position: relative;
    min-height: 78px !important;
    border: 1px solid #286b53 !important;
    border-radius: 17px !important;
    color: #ffffff !important;
    background: linear-gradient(
        135deg,
        #245f4a 0%,
        #2f7d62 52%,
        #45a17f 100%
    ) !important;
    box-shadow:
        0 12px 26px rgba(35, 113, 82, .20),
        inset 0 1px 0 rgba(255, 255, 255, .18) !important;
    font-size: 20.61px !important;
    font-weight: 650 !important;
    letter-spacing: .055em !important;
    text-shadow: 0 1px 1px rgba(19, 68, 49, .18) !important;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, filter .18s ease !important;
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
    background: #b8efd2;
    box-shadow: 0 0 0 4px rgba(184, 239, 210, .16);
}

#grasp-btn:hover {
    transform: translateY(-2px);
    border-color: #1f6048 !important;
    background: linear-gradient(
        135deg,
        #1f5b45 0%,
        #2a755a 48%,
        #4baa86 100%
    ) !important;
    box-shadow:
        0 16px 32px rgba(35, 113, 82, .26),
        inset 0 1px 0 rgba(255, 255, 255, .20) !important;
    filter: saturate(1.04);
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
    background: linear-gradient(90deg, #a5e8c5, rgba(165, 232, 197, 0));
    transition: width .30s ease;
    pointer-events: none;
}

#grasp-btn:hover::after {
    width: 72%;
}

/* Secondary controls: pale blue gradient surfaces + cobalt-to-sky gradient type. */
#voice-btn,
#perceive-btn,
#home-left-btn,
#home-right-btn {
    border: 1px solid rgba(46, 118, 214, .28) !important;
    color: transparent !important;
    -webkit-text-fill-color: transparent !important;
    background-image:
        linear-gradient(90deg, #002fa7 0%, #0d73d8 52%, #17a8e8 100%),
        linear-gradient(135deg, #f8fbff 0%, #eaf4ff 48%, #d9ecff 100%) !important;
    background-origin: border-box, padding-box !important;
    background-clip: text, padding-box !important;
    -webkit-background-clip: text, padding-box !important;
    box-shadow:
        0 5px 14px rgba(20, 89, 176, .07),
        inset 0 1px 0 rgba(255, 255, 255, .92) !important;
}

#voice-btn,
#perceive-btn {
    min-height: 49px !important;
    border-radius: 13px !important;
    font-size: 11.45px !important;
    font-weight: 650 !important;
    letter-spacing: .04em !important;
}

/* PERCEPTION ONLY is intentionally one step stronger in the same blue family. */
#perceive-btn {
    border-color: rgba(13, 115, 216, .34) !important;
    background-image:
        linear-gradient(90deg, #002fa7 0%, #0969cf 46%, #0d93d8 100%),
        linear-gradient(135deg, #f0f7ff 0%, #dceeff 48%, #c9e5ff 100%) !important;
}

#voice-btn:hover,
#home-left-btn:hover,
#home-right-btn:hover {
    transform: translateY(-1px);
    border-color: rgba(13, 115, 216, .46) !important;
    background-image:
        linear-gradient(90deg, #00278f 0%, #086bd0 50%, #079edc 100%),
        linear-gradient(135deg, #f2f8ff 0%, #dceeff 44%, #c8e4ff 100%) !important;
    box-shadow:
        0 8px 18px rgba(20, 89, 176, .11),
        inset 0 1px 0 rgba(255, 255, 255, .96) !important;
}

#perceive-btn:hover {
    transform: translateY(-1px);
    border-color: rgba(13, 115, 216, .54) !important;
    background-image:
        linear-gradient(90deg, #00278f 0%, #075fc4 46%, #078fd2 100%),
        linear-gradient(135deg, #eaf4ff 0%, #d2e9ff 48%, #bcdcff 100%) !important;
    box-shadow:
        0 8px 18px rgba(20, 89, 176, .13),
        inset 0 1px 0 rgba(255, 255, 255, .96) !important;
}

#home-left-btn,
#home-right-btn {
    min-height: 43px !important;
    border-radius: 12px !important;
    font-size: 10.3px !important;
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
    font-size: 10.3px !important;
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
    border: 1px solid var(--art-line) !important;
    border-radius: 12px !important;
    color: var(--art-ink-soft) !important;
    background: var(--art-surface-soft) !important;
    font-family: var(--sans) !important;
    font-size: 10.3px !important;
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
    color: var(--art-muted) !important;
    font-family: var(--sans) !important;
    font-size: 9.16px !important;
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
    scrollbar-color: #b8c7dc transparent;
}

/* =============================================================================
   Footer
   ============================================================================= */

.compact-footer {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin: 0 5px;
    color: var(--art-muted-2);
    font-family: var(--sans);
    font-size: 8.01px;
    font-weight: 600;
    letter-spacing: .095em;
    text-transform: uppercase;
}

/* =============================================================================
   Responsive — retain the single-workspace concept
   ============================================================================= */

@media (max-width: 1650px) {
    .gradio-container > .main,
    .gradio-container .main {
        width: calc((100vw - 18px) / var(--ui-scale)) !important;
        padding-top: 9px !important;
    }

    #compact-header {
        min-height: 62px;
        border-radius: 18px;
    }

    .brand-logo {
        width: 141.96px;
        max-height: 38.92px;
    }

    .brand-divider {
        height: 32px;
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

@media (max-width: 1230px) {
    body {
        overflow-y: auto;
    }

    .gradio-container > .main,
    .gradio-container .main {
        height: auto;
        min-height: calc(100vh / var(--ui-scale));
    }

    .pipeline {
        display: none;
    }

    #compact-header {
        min-height: 54px;
        padding: 9px 13px;
    }

    .brand-logo {
        width: 123.64px;
        max-height: 34.34px;
    }

    .brand-divider {
        height: 28px;
    }

    .brand-title {
        font-size: 27.48px;
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

    <div class="brand-lockup">

        <div class="brand-logo-wrap">
            <img
                class="brand-logo"
                src="data:image/png;base64,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"
                alt="KernelMind"
            />
        </div>

        <span class="brand-divider"></span>

        <div class="brand-block">

            <div class="brand-top">
                <span class="live-dot"></span>
                MARVINGRASP · EMBODIED AI CONTROL
            </div>

            <h1 class="brand-title">
                Perception into action.
            </h1>

        </div>

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
    voice_enabled: bool = VOICE_ENABLED,
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
                                "VOICE GRASP · 2 S",
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
        *,
        enabled: bool = WEB_ENABLED,
        voice_enabled: bool = VOICE_ENABLED,
    ) -> None:
        self.bridge = bridge
        self.host = host
        self.port = int(port)
        self.enabled = bool(enabled)
        self.voice_enabled = bool(voice_enabled)

        self.demo = None

    def start(self) -> None:
        """Start the tablet web server."""

        if not self.enabled:
            print("[tablet_ui] disabled", flush=True)
            return

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