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


def format_base_target_text(targets: list[object], *, limit: int = 6) -> str:
    """Format target objects for the tablet without coupling it to core types."""
    lines: list[str] = []
    for target in targets[:limit]:
        xyz = getattr(target, "base_xyz", ())
        if len(xyz) < 3:
            continue
        label = getattr(target, "frame_id", getattr(target, "label", "target"))
        lines.append(
            f"{label}: x={float(xyz[0]):.3f} y={float(xyz[1]):.3f} "
            f"z={float(xyz[2]):.3f} m"
        )
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
    --bg-0: #04080d;
    --bg-1: #071219;
    --bg-2: #0a1b23;

    --panel: rgba(9, 25, 32, .91);
    --panel-soft: rgba(10, 30, 38, .78);

    --text: #e9f4f0;
    --text-soft: #9eb8b2;
    --text-dim: #69847f;

    --jade: #6bc8bb;
    --jade-bright: #a5f5e5;
    --jade-dark: #173f3d;
    --jade-deep: #0d302f;

    --blue: #398aa0;
    --blue-dark: #123745;

    --gold: #d3aa60;
    --gold-dark: #49391f;
    --gold-deep: #302715;

    --red: #dd725f;
    --red-dark: #54231f;
    --red-deep: #351715;

    --line: rgba(161, 219, 208, .15);

    --serif:
        "Noto Serif SC",
        "Source Han Serif SC",
        Georgia,
        serif;

    --sans:
        Inter,
        "Noto Sans SC",
        "Source Han Sans SC",
        "PingFang SC",
        system-ui,
        sans-serif;

    --mono:
        "JetBrains Mono",
        "SFMono-Regular",
        Consolas,
        monospace;
}


/* =============================================================================
   Page
============================================================================= */

html,
body {
    margin: 0;
    width: 100%;
    min-height: 100%;
    background: var(--bg-0);
}

body {
    overflow-x: hidden;
}

.gradio-container {
    position: relative;

    min-height: 100vh;

    color: var(--text) !important;
    font-family: var(--sans) !important;

    background:
        radial-gradient(
            circle at 8% -5%,
            rgba(55, 151, 145, .17),
            transparent 26rem
        ),
        radial-gradient(
            circle at 96% 5%,
            rgba(211, 170, 96, .08),
            transparent 24rem
        ),
        linear-gradient(
            145deg,
            var(--bg-0) 0%,
            var(--bg-1) 52%,
            #061017 100%
        ) !important;
}


/* engineering grid */

.gradio-container::before {
    content: "";

    position: fixed;
    inset: 0;

    pointer-events: none;

    opacity: .27;

    background-image:
        linear-gradient(
            rgba(161, 219, 208, .024) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(161, 219, 208, .024) 1px,
            transparent 1px
        );

    background-size:
        52px 52px;

    mask-image:
        linear-gradient(
            to bottom,
            black,
            transparent 92%
        );
}


/* fine grain */

.gradio-container::after {
    content: "";

    position: fixed;
    inset: 0;

    pointer-events: none;

    opacity: .045;

    background-image:
        url(
            "data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.8'/%3E%3C/svg%3E"
        );
}


.gradio-container > .main,
.gradio-container .main {
    position: relative;

    z-index: 1;

    width: min(1380px, calc(100vw - 24px)) !important;
    max-width: 1380px !important;

    margin: 0 auto !important;

    padding:
        10px 0 12px !important;
}


footer {
    display: none !important;
}


#tablet-shell {
    gap: 9px !important;
}


/* =============================================================================
   Compact header
============================================================================= */

#compact-header {
    position: relative;

    display: flex;

    align-items: center;
    justify-content: space-between;

    min-height: 62px;

    overflow: hidden;

    padding:
        11px 18px;

    border:
        1px solid var(--line);

    border-radius:
        17px;

    background:
        linear-gradient(
            120deg,
            rgba(13, 40, 48, .96),
            rgba(6, 17, 23, .94)
        );

    box-shadow:
        0 13px 35px rgba(0, 0, 0, .27),
        inset 0 1px rgba(255, 255, 255, .025);
}


#compact-header::after {
    content: "";

    position: absolute;

    width: 260px;
    height: 260px;

    right: -95px;
    top: -205px;

    border:
        1px solid rgba(165, 245, 229, .13);

    border-radius:
        50%;

    box-shadow:
        0 0 0 30px rgba(165, 245, 229, .017),
        0 0 70px rgba(80, 181, 172, .10);

    animation:
        breathe 7s ease-in-out infinite;

    pointer-events: none;
}


.brand-block {
    min-width: 0;
}


.brand-top {
    display: flex;

    align-items: center;

    gap: 9px;

    color: var(--jade);

    font-family: var(--mono);

    font-size: 9px;

    letter-spacing: .18em;

    white-space: nowrap;
}


.live-dot {
    width: 7px;
    height: 7px;

    flex: 0 0 auto;

    border-radius: 50%;

    background: var(--jade-bright);

    box-shadow:
        0 0 0 5px rgba(165, 245, 229, .06),
        0 0 16px rgba(165, 245, 229, .55);

    animation:
        pulse 2.2s ease-in-out infinite;
}


.brand-title {
    margin:
        3px 0 0;

    color: #f2f8f6;

    font-family: var(--serif);

    font-size:
        clamp(23px, 2.8vw, 36px);

    font-weight:
        500;

    line-height: 1;

    letter-spacing:
        -.025em;
}


.pipeline {
    position: relative;

    z-index: 1;

    display: flex;

    align-items: center;

    gap: 7px;
}


.pipeline span {
    padding:
        6px 9px;

    border:
        1px solid rgba(145, 211, 199, .11);

    border-radius:
        999px;

    color:
        #91aaa5;

    background:
        rgba(3, 14, 19, .55);

    font-family:
        var(--mono);

    font-size:
        8px;

    letter-spacing:
        .06em;

    white-space:
        nowrap;
}


.pipeline b {
    color:
        var(--gold);

    font-size:
        10px;

    font-weight:
        400;
}


/* =============================================================================
   Main workspace: perception left, controls right
============================================================================= */

#workspace {
    gap:
        10px !important;

    align-items:
        stretch !important;
}


/* =============================================================================
   Shared panels
============================================================================= */

.art-panel {
    position: relative;

    overflow: hidden;

    border:
        1px solid var(--line) !important;

    border-radius:
        17px !important;

    background:
        linear-gradient(
            145deg,
            rgba(10, 29, 36, .91),
            rgba(5, 16, 22, .89)
        ) !important;

    box-shadow:
        0 14px 38px rgba(0, 0, 0, .22),
        inset 0 1px rgba(255, 255, 255, .02) !important;

    backdrop-filter:
        blur(15px);
}


.art-panel::before {
    content: "";

    position: absolute;

    left: 0;
    right: 0;
    top: 0;

    height: 1px;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(165, 245, 229, .30),
            transparent
        );
}


/* =============================================================================
   Perception panel
============================================================================= */

#perception-panel {
    padding:
        13px !important;
}


.panel-heading {
    display: flex;

    align-items: flex-end;
    justify-content: space-between;

    gap: 15px;

    margin-bottom:
        8px;
}


.panel-kicker {
    color:
        var(--gold);

    font-family:
        var(--mono);

    font-size:
        8px;

    letter-spacing:
        .17em;

    text-transform:
        uppercase;
}


.panel-title {
    margin:
        2px 0 0;

    color:
        #edf7f3;

    font-family:
        var(--serif);

    font-size:
        23px;

    font-weight:
        500;

    line-height:
        1.05;
}


.panel-meta {
    color:
        #6d8984;

    font-family:
        var(--mono);

    font-size:
        8px;

    letter-spacing:
        .08em;

    text-align:
        right;
}


#perception-results {
    gap:
        9px !important;
}


/* individual perception card */

.result-card {
    min-width: 0 !important;

    padding:
        9px !important;

    border:
        1px solid rgba(153, 214, 203, .085) !important;

    border-radius:
        13px !important;

    background:
        rgba(2, 12, 17, .49) !important;
}


.result-head {
    display: flex;

    align-items: center;
    justify-content: space-between;

    margin-bottom:
        5px;
}


.result-name {
    color:
        #e7f2ee;

    font-family:
        var(--serif);

    font-size:
        17px;

    font-weight:
        500;
}


.result-type {
    color:
        var(--jade);

    font-family:
        var(--mono);

    font-size:
        7px;

    letter-spacing:
        .12em;
}


.result-view {
    border:
        1px solid rgba(145, 211, 199, .09) !important;

    border-radius:
        11px !important;

    background:
        radial-gradient(
            circle at 50% 44%,
            rgba(57, 138, 160, .08),
            transparent 48%
        ),
        #020b10 !important;

    box-shadow:
        inset 0 0 28px rgba(0, 0, 0, .32) !important;
}


.result-view img {
    width:
        100% !important;

    height:
        100% !important;

    max-height:
        300px !important;

    border-radius:
        9px !important;

    object-fit:
        contain !important;

    filter:
        saturate(.95)
        contrast(1.03);
}


.result-view .image-container,
.result-view > div {
    min-height:
        275px !important;

    height:
        275px !important;

    max-height:
        275px !important;
}

#base-targets {
    margin-top: 4px !important;
}

#base-targets textarea {
    color: var(--text-soft) !important;
    background: var(--bg-0) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    line-height: 1.5 !important;
}


/* =============================================================================
   Control panel
============================================================================= */

#control-panel {
    padding:
        13px !important;

    gap:
        7px !important;
}


.control-heading {
    margin-bottom:
        3px;
}


.control-heading .panel-title {
    font-size:
        21px;
}


.flow-strip {
    display: grid;

    grid-template-columns:
        1fr auto 1fr auto 1fr;

    align-items: center;

    gap:
        5px;

    margin:
        4px 0 6px;

    padding:
        7px 8px;

    border:
        1px solid rgba(145, 211, 199, .08);

    border-radius:
        10px;

    background:
        rgba(2, 12, 17, .40);

    color:
        #79938e;

    font-family:
        var(--mono);

    font-size:
        7px;

    letter-spacing:
        .06em;

    text-align:
        center;
}


.flow-strip b {
    color:
        var(--jade);

    font-size:
        10px;

    font-weight:
        400;
}


/* =============================================================================
   Main GRASP button
============================================================================= */

#grasp-btn {
    position: relative;

    min-height:
        78px !important;

    border:
        1px solid #428f86 !important;

    border-radius:
        15px !important;

    color:
        #eafff9 !important;

    background:
        linear-gradient(
            135deg,
            #164b48 0%,
            #1b615b 48%,
            #206e66 100%
        ) !important;

    box-shadow:
        0 12px 28px rgba(16, 98, 91, .24),
        inset 0 1px rgba(204, 255, 244, .08) !important;

    font-size:
        19px !important;

    font-weight:
        700 !important;

    letter-spacing:
        .07em !important;

    text-shadow:
        0 1px 2px rgba(0, 0, 0, .35);

    transition:
        transform .18s ease,
        border-color .18s ease,
        box-shadow .18s ease !important;
}


#grasp-btn:hover {
    transform:
        translateY(-2px);

    border-color:
        #69cabe !important;

    background:
        linear-gradient(
            135deg,
            #18534f,
            #20716a
        ) !important;

    box-shadow:
        0 15px 34px rgba(30, 138, 127, .28),
        0 0 0 1px rgba(150, 245, 229, .06) !important;
}


#grasp-btn:active {
    transform:
        translateY(0)
        scale(.993);
}


#grasp-btn::after {
    content: "";

    position: absolute;

    inset:
        -140% -35%;

    transform:
        translateX(-70%)
        rotate(18deg);

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, .10),
            transparent
        );

    transition:
        transform .60s ease;

    pointer-events:
        none;
}


#grasp-btn:hover::after {
    transform:
        translateX(70%)
        rotate(18deg);
}


/* =============================================================================
   Voice button
============================================================================= */

#voice-btn {
    min-height:
        49px !important;

    border:
        1px solid #826a38 !important;

    border-radius:
        12px !important;

    color:
        #f5dfae !important;

    background:
        linear-gradient(
            135deg,
            #332915,
            #49391f
        ) !important;

    font-size:
        11px !important;

    font-weight:
        700 !important;

    letter-spacing:
        .045em !important;

    box-shadow:
        inset 0 1px rgba(255, 237, 190, .04) !important;
}


#voice-btn:hover {
    transform:
        translateY(-1px);

    border-color:
        #ba9550 !important;

    background:
        linear-gradient(
            135deg,
            #403219,
            #5b4624
        ) !important;
}


/* =============================================================================
   Perception button
============================================================================= */

#perceive-btn {
    min-height:
        49px !important;

    border:
        1px solid #2d6575 !important;

    border-radius:
        12px !important;

    color:
        #cdebf1 !important;

    background:
        linear-gradient(
            135deg,
            #102d38,
            #164351
        ) !important;

    font-size:
        11px !important;

    font-weight:
        700 !important;

    letter-spacing:
        .04em !important;
}


#perceive-btn:hover {
    transform:
        translateY(-1px);

    border-color:
        #488fa2 !important;

    background:
        linear-gradient(
            135deg,
            #153744,
            #195064
        ) !important;
}


/* =============================================================================
   Status
============================================================================= */

#status-area {
    gap:
        7px !important;
}


#robot-status,
#activity-status {
    min-width:
        0 !important;
}


#robot-status textarea,
#activity-status textarea {
    min-height:
        53px !important;

    height:
        53px !important;

    padding:
        8px 10px !important;

    border:
        1px solid rgba(145, 211, 199, .10) !important;

    border-radius:
        10px !important;

    color:
        #c4d8d3 !important;

    background:
        rgba(2, 11, 16, .72) !important;

    font-family:
        var(--mono) !important;

    font-size:
        9px !important;

    line-height:
        1.42 !important;

    resize:
        none !important;

    box-shadow:
        inset 0 1px 13px rgba(0, 0, 0, .19) !important;
}


/* =============================================================================
   Home buttons
============================================================================= */

#home-left-btn,
#home-right-btn {
    min-height:
        43px !important;

    border:
        1px solid #65532f !important;

    border-radius:
        11px !important;

    color:
        #dbc797 !important;

    background:
        linear-gradient(
            135deg,
            #272114,
            #382e19
        ) !important;

    font-size:
        10px !important;

    font-weight:
        700 !important;

    letter-spacing:
        .035em !important;
}


#home-left-btn:hover,
#home-right-btn:hover {
    transform:
        translateY(-1px);

    border-color:
        #92743e !important;

    background:
        linear-gradient(
            135deg,
            #312817,
            #49391f
        ) !important;
}


/* =============================================================================
   Stop button
============================================================================= */

#stop-btn {
    min-height:
        43px !important;

    border:
        1px solid #91483d !important;

    border-radius:
        11px !important;

    color:
        #ffd4cd !important;

    background:
        linear-gradient(
            135deg,
            #3b1a17,
            #59241f
        ) !important;

    font-size:
        10px !important;

    font-weight:
        800 !important;

    letter-spacing:
        .06em !important;
}


#stop-btn:hover {
    transform:
        translateY(-1px);

    border-color:
        #d66b59 !important;

    background:
        linear-gradient(
            135deg,
            #4b1e1a,
            #702c25
        ) !important;
}


/* =============================================================================
   Small action grid
============================================================================= */

#mode-actions {
    gap:
        7px !important;
}


#home-actions {
    gap:
        7px !important;
}


/* =============================================================================
   Gradio native tweaks
============================================================================= */

.gradio-container label,
.gradio-container .label-wrap {
    color:
        #78918c !important;

    font-family:
        var(--sans) !important;

    font-size:
        9px !important;
}


.gradio-container button {
    overflow:
        hidden;

    font-family:
        var(--sans) !important;

    transition:
        transform .18s ease,
        border-color .18s ease,
        background .18s ease !important;
}


.gradio-container textarea {
    scrollbar-width:
        thin;
}


/* =============================================================================
   Bottom signature
============================================================================= */

.compact-footer {
    display: flex;

    justify-content: space-between;

    gap: 12px;

    margin:
        2px 3px 0;

    color:
        #3f5a56;

    font-family:
        var(--mono);

    font-size:
        7px;

    letter-spacing:
        .11em;

    text-transform:
        uppercase;
}


/* =============================================================================
   Animation
============================================================================= */

@keyframes breathe {
    0%,
    100% {
        transform:
            scale(1);

        opacity:
            .60;
    }

    50% {
        transform:
            scale(1.07);

        opacity:
            1;
    }
}


@keyframes pulse {
    0%,
    100% {
        transform:
            scale(.9);

        opacity:
            .62;
    }

    50% {
        transform:
            scale(1.15);

        opacity:
            1;
    }
}


/* =============================================================================
   Landscape tablets / smaller laptops
============================================================================= */

@media (max-width: 1100px) {

    .gradio-container > .main,
    .gradio-container .main {
        width:
            calc(100vw - 18px) !important;
    }

    .pipeline span:nth-of-type(2),
    .pipeline span:nth-of-type(4) {
        display:
            none;
    }

    .result-view .image-container,
    .result-view > div {
        min-height:
            235px !important;

        height:
            235px !important;

        max-height:
            235px !important;
    }

    .result-view img {
        max-height:
            235px !important;
    }
}


/* =============================================================================
   Portrait tablet
============================================================================= */

@media (max-width: 820px) {

    body {
        overflow-y:
            auto;
    }

    .pipeline {
        display:
            none;
    }

    #compact-header {
        min-height:
            54px;

        padding:
            9px 13px;
    }

    .brand-title {
        font-size:
            24px;
    }

    #workspace {
        flex-direction:
            column !important;
    }

    .result-view .image-container,
    .result-view > div {
        min-height:
            205px !important;

        height:
            205px !important;

        max-height:
            205px !important;
    }

    .result-view img {
        max-height:
            205px !important;
    }

    #grasp-btn {
        min-height:
            65px !important;
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
