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

        self._image_rgb: np.ndarray | None = None
        self._sam_rgb: np.ndarray | None = None
        self._flowpose_rgb: np.ndarray | None = None

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
                "Requested: Perception + Autonomous Grasp",

            TabletCommand.VOICE:
                "Requested: 4-second voice command",

            TabletCommand.HOME_LEFT:
                "Requested: Left Arm Home",

            TabletCommand.HOME_RIGHT:
                "Requested: Right Arm Home",

            TabletCommand.STOP:
                "Requested: Stop Current Task",
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
    ) -> None:
        """Publish frames and status for the next browser polling cycle."""

        image_rgb = self._to_browser_rgb(image_bgr)
        sam_rgb = self._to_browser_rgb(sam_bgr)
        flowpose_rgb = self._to_browser_rgb(flowpose_bgr)

        with self._lock:
            self._image_rgb = image_rgb
            self._sam_rgb = sam_rgb
            self._flowpose_rgb = flowpose_rgb
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
                status=self._status,
                activity=self._activity,
            )


# =============================================================================
# Visual style
# =============================================================================


TABLET_CSS = r"""
:root {
    --ink-0: #05090f;
    --ink-1: #08131b;
    --ink-2: #0d202a;

    --mist: #dcece8;
    --muted: rgba(220, 236, 232, .54);

    --jade: #83d8cb;
    --jade-bright: #b7fff0;
    --river: #4fa8bb;

    --gold: #d7b36a;
    --coral: #e68572;

    --glass: rgba(9, 23, 30, .74);
    --glass-strong: rgba(6, 17, 23, .90);
    --line: rgba(173, 225, 216, .13);

    --serif:
        "Noto Serif SC",
        "Source Han Serif SC",
        "Songti SC",
        serif;

    --sans:
        Inter,
        "Noto Sans SC",
        "Source Han Sans SC",
        "PingFang SC",
        sans-serif;

    --mono:
        "JetBrains Mono",
        "SFMono-Regular",
        Consolas,
        monospace;
}


/* =========================================================================
   Base
========================================================================= */

html {
    scroll-behavior: smooth;
}

body {
    margin: 0;
    background: var(--ink-0);
}

.gradio-container {
    position: relative;
    min-height: 100vh;
    overflow-x: hidden;

    color: var(--mist) !important;
    font-family: var(--sans) !important;

    background:
        radial-gradient(
            circle at 12% 3%,
            rgba(87, 173, 170, .17),
            transparent 30rem
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(215, 179, 106, .09),
            transparent 28rem
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(42, 116, 129, .19),
            transparent 36rem
        ),
        linear-gradient(
            180deg,
            #05090f 0%,
            #07141b 45%,
            #061017 100%
        ) !important;
}


/* engineering grid */

.gradio-container::before {
    content: "";

    position: fixed;
    inset: 0;

    z-index: 0;
    pointer-events: none;

    opacity: .30;

    background-image:
        linear-gradient(
            rgba(164, 224, 214, .025) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(164, 224, 214, .025) 1px,
            transparent 1px
        );

    background-size: 54px 54px;

    mask-image:
        linear-gradient(
            to bottom,
            black,
            transparent 88%
        );
}


/* film grain */

.gradio-container::after {
    content: "";

    position: fixed;
    inset: 0;

    pointer-events: none;
    z-index: 0;

    opacity: .055;

    background-image:
        url(
            "data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.9'/%3E%3C/svg%3E"
        );
}

.gradio-container > .main,
.gradio-container .main {
    position: relative;
    z-index: 1;

    max-width: 1400px !important;

    margin: 0 auto !important;
    padding: 18px 26px 36px !important;
}

footer {
    display: none !important;
}


/* =========================================================================
   Main shell
========================================================================= */

#tablet-shell {
    width: 100%;
}


/* =========================================================================
   Hero
========================================================================= */

#tablet-hero {
    position: relative;

    overflow: hidden;

    margin: 2px 0 18px;
    padding: 30px 38px 26px;

    border:
        1px solid rgba(174, 231, 220, .14);

    border-radius: 28px;

    background:
        linear-gradient(
            135deg,
            rgba(14, 35, 43, .94),
            rgba(6, 13, 20, .78)
        ),
        radial-gradient(
            circle at 80% 0%,
            rgba(133, 216, 203, .14),
            transparent 38%
        );

    box-shadow:
        0 28px 80px rgba(0, 0, 0, .32),
        inset 0 1px rgba(255, 255, 255, .035);

    isolation: isolate;
}

#tablet-hero::before {
    content: "";

    position: absolute;

    width: 410px;
    height: 410px;

    right: -90px;
    top: -260px;

    border:
        1px solid rgba(177, 239, 228, .15);

    border-radius: 50%;

    box-shadow:
        0 0 0 38px rgba(177, 239, 228, .02),
        0 0 0 88px rgba(177, 239, 228, .014),
        0 0 100px rgba(96, 186, 183, .11);

    animation:
        hero-breathe 7s ease-in-out infinite;
}

#tablet-hero::after {
    content: "";

    position: absolute;

    left: -8%;
    right: -8%;
    bottom: -54px;

    height: 100px;

    border-radius: 50% 50% 0 0;

    background:
        repeating-radial-gradient(
            ellipse at 50% 100%,
            transparent 0 28px,
            rgba(110, 207, 195, .07) 29px 30px
        );

    animation:
        lake-drift 8s ease-in-out infinite alternate;

    z-index: -1;
}

.hero-topline {
    display: flex;
    align-items: center;

    gap: 11px;

    margin-bottom: 15px;

    color: var(--jade);

    font-family: var(--mono);
    font-size: 10px;

    letter-spacing: .20em;

    text-transform: uppercase;
}

.live-dot {
    width: 7px;
    height: 7px;

    flex: 0 0 auto;

    border-radius: 50%;

    background: var(--jade-bright);

    box-shadow:
        0 0 0 6px rgba(183, 255, 240, .07),
        0 0 20px rgba(183, 255, 240, .62);

    animation:
        pulse-dot 2.2s ease-in-out infinite;
}

.hero-title {
    margin: 0;

    color: #f1f7f4;

    font-family: var(--serif);

    font-size:
        clamp(34px, 5vw, 62px);

    font-weight: 500;

    line-height: 1.08;

    letter-spacing: -.035em;
}

.hero-title em {
    color: var(--jade-bright);

    font-style: normal;

    text-shadow:
        0 0 35px rgba(127, 224, 207, .18);
}

.hero-sub {
    max-width: 700px;

    margin:
        13px 0 20px;

    color:
        rgba(220, 236, 232, .58);

    font-family: var(--serif);

    font-size: 14px;

    line-height: 1.8;

    letter-spacing: .025em;
}

.pipeline {
    display: flex;
    flex-wrap: wrap;

    gap: 8px;
}

.pipeline span {
    padding: 7px 11px;

    border:
        1px solid rgba(160, 225, 213, .11);

    border-radius: 999px;

    color:
        rgba(226, 242, 238, .62);

    background:
        rgba(8, 22, 28, .56);

    font-family: var(--mono);

    font-size: 9px;

    letter-spacing: .08em;

    backdrop-filter: blur(10px);
}


/* =========================================================================
   Shared Panels
========================================================================= */

.art-panel {
    position: relative;

    overflow: hidden;

    padding: 20px !important;

    border:
        1px solid var(--line) !important;

    border-radius: 23px !important;

    background:
        linear-gradient(
            145deg,
            rgba(12, 31, 39, .86),
            rgba(7, 17, 23, .72)
        ) !important;

    box-shadow:
        0 20px 58px rgba(0, 0, 0, .23),
        inset 0 1px rgba(255, 255, 255, .022) !important;

    backdrop-filter: blur(18px);
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
            rgba(183, 255, 240, .34),
            transparent
        );
}

.section-kicker {
    display: flex;
    align-items: center;

    gap: 8px;

    margin-bottom: 6px;

    color: var(--gold);

    font-family: var(--mono);

    font-size: 9px;

    letter-spacing: .18em;

    text-transform: uppercase;
}

.section-kicker::before {
    content: "";

    width: 22px;
    height: 1px;

    background: currentColor;
}

.section-title {
    margin:
        0 0 5px;

    color: #eef8f4;

    font-family: var(--serif);

    font-size:
        clamp(24px, 3vw, 36px);

    font-weight: 500;

    letter-spacing: -.025em;
}

.section-copy {
    margin:
        0 0 14px;

    color:
        rgba(215, 235, 230, .52);

    font-family: var(--serif);

    font-size: 12px;

    line-height: 1.7;
}


/* =========================================================================
   Main Camera
========================================================================= */

#camera-panel {
    padding: 20px !important;
}

#camera-view {
    margin-top: 11px;

    border:
        1px solid rgba(161, 226, 214, .11) !important;

    border-radius: 18px !important;

    background:
        radial-gradient(
            circle at 50% 45%,
            rgba(70, 164, 165, .11),
            transparent 48%
        ),
        rgba(3, 13, 18, .74) !important;

    box-shadow:
        inset 0 0 42px rgba(0, 0, 0, .28) !important;
}

#camera-view img {
    border-radius: 14px !important;

    object-fit: contain !important;

    filter:
        saturate(.93)
        contrast(1.03);
}

#camera-view .image-container,
#camera-view > div {
    min-height: 400px;
}


/* =========================================================================
   SAM3 + FlowPose Results
========================================================================= */

#perception-results {
    gap: 14px !important;
}

.result-card {
    padding: 17px !important;
}

.result-card .image-container {
    min-height: 220px;
}

.result-view img {
    border-radius: 13px !important;

    object-fit: contain !important;
}

.result-caption {
    margin-bottom: 9px;

    color:
        rgba(222, 238, 234, .67);

    font-family: var(--mono);

    font-size: 9px;

    letter-spacing: .14em;
}


/* =========================================================================
   River Divider
========================================================================= */

.river-divider {
    position: relative;

    height: 66px;

    overflow: hidden;

    margin:
        4px 0;
}

.river-divider svg {
    width: 100%;
    height: 100%;
}

.river-main {
    fill: none;

    stroke:
        url(#tabletRiverGradient);

    stroke-width: 1.15;

    stroke-dasharray:
        9 8;

    animation:
        river-run 11s linear infinite;

    filter:
        drop-shadow(
            0 0 7px rgba(96, 199, 192, .23)
        );
}

.river-ghost {
    fill: none;

    stroke:
        rgba(142, 221, 211, .065);

    stroke-width: 6;
}


/* =========================================================================
   Control Area
========================================================================= */

#control-panel {
    padding: 21px !important;
}

.action-intro {
    display: flex;

    align-items: flex-end;
    justify-content: space-between;

    gap: 18px;

    margin-bottom: 18px;
}

.action-note {
    max-width: 430px;

    padding:
        11px 14px;

    border-left:
        1px solid var(--gold);

    color:
        rgba(224, 236, 232, .51);

    font-family: var(--serif);

    font-size: 11px;

    line-height: 1.65;
}


/* =========================================================================
   Main One-touch Grasp Button
========================================================================= */

#grasp-btn {
    position: relative;

    min-height: 104px !important;

    border:
        1px solid rgba(184, 255, 239, .30) !important;

    border-radius: 20px !important;

    color:
        #061719 !important;

    background:
        linear-gradient(
            135deg,
            #c3fff3,
            #76d4c7 52%,
            #58b8b3
        ) !important;

    box-shadow:
        0 18px 45px rgba(62, 179, 166, .20),
        inset 0 1px rgba(255, 255, 255, .55) !important;

    font-size: 22px !important;

    font-weight: 700 !important;

    letter-spacing: .055em !important;

    transition:
        transform .22s ease,
        box-shadow .22s ease !important;
}

#grasp-btn:hover {
    transform:
        translateY(-3px);

    box-shadow:
        0 24px 56px rgba(62, 179, 166, .27),
        inset 0 1px rgba(255, 255, 255, .65) !important;
}

#grasp-btn:active {
    transform:
        translateY(0)
        scale(.992);
}

#grasp-btn::after {
    content: "";

    position: absolute;

    inset: -130% -30%;

    transform:
        translateX(-65%)
        rotate(18deg);

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, .30),
            transparent
        );

    transition:
        transform .7s ease;

    pointer-events: none;
}

#grasp-btn:hover::after {
    transform:
        translateX(65%)
        rotate(18deg);
}


/* =========================================================================
   Perception-only Button
========================================================================= */

#perceive-btn {
    min-height: 64px !important;

    margin-top: 10px;

    border:
        1px solid rgba(145, 219, 207, .15) !important;

    border-radius: 17px !important;

    color:
        rgba(210, 246, 238, .84) !important;

    background:
        rgba(17, 56, 63, .48) !important;

    font-size: 15px !important;

    font-weight: 600 !important;
}

#voice-btn {
    min-height: 64px !important;
    margin-top: 10px;
    border: 1px solid rgba(215, 179, 106, .35) !important;
    border-radius: 17px !important;
    color: #f3e3c3 !important;
    background: rgba(90, 68, 31, .36) !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

#voice-btn:hover {
    transform: translateY(-2px);
}

.voice-note {
    margin: 7px 3px 0;
    color: var(--muted);
    font-size: 11px;
}


/* =========================================================================
   Home / Stop
========================================================================= */

#home-left-btn,
#home-right-btn,
#stop-btn {
    min-height: 57px !important;

    border-radius: 15px !important;

    font-size: 13px !important;

    font-weight: 600 !important;

    transition:
        transform .20s ease,
        border-color .20s ease !important;
}

#home-left-btn,
#home-right-btn {
    border:
        1px solid rgba(215, 179, 106, .20) !important;

    color:
        #ead9b5 !important;

    background:
        rgba(90, 68, 31, .22) !important;
}

#stop-btn {
    border:
        1px solid rgba(230, 133, 114, .26) !important;

    color:
        #ffd9d1 !important;

    background:
        rgba(104, 39, 34, .33) !important;
}

#home-left-btn:hover,
#home-right-btn:hover,
#stop-btn:hover,
#perceive-btn:hover {
    transform:
        translateY(-2px);
}


/* =========================================================================
   Status
========================================================================= */

.status-strip {
    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 8px;

    margin-bottom: 16px;
}

.status-stage {
    padding:
        9px;

    border:
        1px solid rgba(154, 219, 207, .10);

    border-radius: 11px;

    color:
        rgba(225, 241, 237, .56);

    background:
        rgba(4, 16, 21, .42);

    font-family: var(--mono);

    font-size: 8px;

    text-align: center;

    letter-spacing: .07em;
}

#robot-status textarea,
#activity-status textarea {
    border:
        1px solid rgba(151, 214, 203, .10) !important;

    border-radius: 14px !important;

    color:
        rgba(228, 242, 238, .82) !important;

    background:
        rgba(3, 13, 18, .70) !important;

    font-family:
        var(--mono) !important;

    font-size: 11px !important;

    line-height: 1.65 !important;

    box-shadow:
        inset 0 1px 16px rgba(0, 0, 0, .17) !important;
}


/* =========================================================================
   Native Gradio
========================================================================= */

.gradio-container label,
.gradio-container .label-wrap {
    color:
        rgba(225, 241, 237, .64) !important;

    font-family:
        var(--sans) !important;
}

.gradio-container button {
    overflow: hidden;

    font-family:
        var(--sans) !important;
}


/* =========================================================================
   Footer
========================================================================= */

.master-footer {
    display: flex;

    justify-content: space-between;

    gap: 16px;

    margin:
        22px 3px 0;

    padding-top:
        15px;

    border-top:
        1px solid rgba(158, 221, 210, .07);

    color:
        rgba(208, 228, 223, .28);

    font-family:
        var(--mono);

    font-size:
        8px;

    letter-spacing:
        .12em;

    text-transform:
        uppercase;
}


/* =========================================================================
   Animations
========================================================================= */

@keyframes hero-breathe {
    0%,
    100% {
        transform: scale(1);
        opacity: .7;
    }

    50% {
        transform: scale(1.06);
        opacity: 1;
    }
}

@keyframes lake-drift {
    from {
        transform:
            translateX(-1.2%)
            scaleY(.95);
    }

    to {
        transform:
            translateX(1.2%)
            scaleY(1.04);
    }
}

@keyframes pulse-dot {
    0%,
    100% {
        transform: scale(.9);
        opacity: .62;
    }

    50% {
        transform: scale(1.16);
        opacity: 1;
    }
}

@keyframes river-run {
    to {
        stroke-dashoffset: -140;
    }
}


/* =========================================================================
   Tablet Landscape
========================================================================= */

@media (max-width: 1100px) {

    .gradio-container > .main,
    .gradio-container .main {
        padding:
            13px 16px 28px !important;
    }

    #tablet-hero {
        padding:
            26px 28px 23px;
    }

    #camera-view .image-container,
    #camera-view > div {
        min-height: 330px;
    }
}


/* =========================================================================
   Tablet Portrait
========================================================================= */

@media (max-width: 800px) {

    #tablet-hero {
        padding:
            24px 21px 21px;

        border-radius:
            22px;
    }

    .hero-title {
        font-size:
            clamp(31px, 8vw, 48px);
    }

    .action-intro {
        display: block;
    }

    .action-note {
        margin-top:
            12px;
    }

    #grasp-btn {
        min-height:
            92px !important;
    }

    #camera-view .image-container,
    #camera-view > div {
        min-height:
            290px;
    }
}


/* =========================================================================
   Phone Fallback
========================================================================= */

@media (max-width: 560px) {

    .gradio-container > .main,
    .gradio-container .main {
        padding:
            9px 10px 24px !important;
    }

    #tablet-hero {
        padding:
            22px 17px 19px;
    }

    .hero-title {
        font-size:
            34px;
    }

    .hero-sub {
        font-size:
            12px;
    }

    .art-panel {
        padding:
            15px !important;
    }

    .status-strip {
        grid-template-columns:
            1fr;
    }

    .master-footer {
        flex-direction:
            column;
    }
}
"""


# =============================================================================
# HTML Sections
# =============================================================================


TABLET_HERO_HTML = r"""
<section id="tablet-hero">

    <div class="hero-topline">
        <span class="live-dot"></span>
        MARVINGRASP · EMBODIED AI CONTROL
    </div>

    <h1 class="hero-title">
        See the world.<br>
        Then let the robot <em>reach out.</em>
    </h1>

    <p class="hero-sub">
        Light enters through RealSense.
        SAM3 reveals the object boundary.
        FlowPose recovers its spatial orientation,
        and the robot turns perception into motion.
    </p>

    <div class="pipeline">
        <span>RGB-D</span>
        <span>SAM3</span>
        <span>FLOWPOSE 6D</span>
        <span>TF</span>
        <span>MOTION</span>
        <span>GRASP</span>
    </div>

</section>
"""


CAMERA_HEAD_HTML = r"""
<div class="section-kicker">
    LIVE PERCEPTION
</div>

<h2 class="section-title">
    What the robot sees
</h2>

<p class="section-copy">
    Live vision is the entry point of the grasping pipeline.
    Every change in the physical world begins here.
</p>
"""


RESULT_HEAD_HTML = r"""
<div class="section-kicker">
    PERCEPTION STREAM
</div>

<h2 class="section-title">
    From boundary to spatial pose
</h2>

<p class="section-copy">
    SAM3 extracts the object boundary.
    FlowPose reconstructs its 6D pose in 3D space.
</p>
"""


RIVER_HTML = r"""
<div class="river-divider" aria-hidden="true">

    <svg
        viewBox="0 0 1200 70"
        preserveAspectRatio="none"
    >

        <defs>

            <linearGradient
                id="tabletRiverGradient"
                x1="0"
                x2="1"
            >

                <stop
                    offset="0"
                    stop-color="#83d8cb"
                    stop-opacity="0"
                />

                <stop
                    offset=".40"
                    stop-color="#83d8cb"
                />

                <stop
                    offset=".72"
                    stop-color="#d7b36a"
                />

                <stop
                    offset="1"
                    stop-color="#d7b36a"
                    stop-opacity="0"
                />

            </linearGradient>

        </defs>

        <path
            class="river-ghost"
            d="
                M0,42
                C180,3 320,68 470,36
                C650,2 745,70 900,34
                C1030,8 1115,52 1200,28
            "
        />

        <path
            class="river-main"
            d="
                M0,42
                C180,3 320,68 470,36
                C650,2 745,70 900,34
                C1030,8 1115,52 1200,28
            "
        />

    </svg>

</div>
"""


CONTROL_HEAD_HTML = r"""
<div class="action-intro">

    <div>

        <div class="section-kicker">
            ONE TOUCH GRASP
        </div>

        <h2 class="section-title">
            Turn perception into action
        </h2>

        <p class="section-copy">
            One touch runs perception, pose estimation,
            motion planning and robotic grasp execution.
        </p>

    </div>

    <div class="action-note">
        When the physical world changes,
        the system observes it again.
    </div>

</div>

<div class="status-strip">

    <div class="status-stage">
        01 · SEE
    </div>

    <div class="status-stage">
        02 · UNDERSTAND
    </div>

    <div class="status-stage">
        03 · ACT
    </div>

</div>
"""


FOOTER_HTML = r"""
<div class="master-footer">

    <span>
        MARVINGRASP · TABLET CONTROL
    </span>

    <span>
        PERCEPTION → POSE → MOTION → GRASP
    </span>

</div>
"""


# =============================================================================
# Gradio Tablet UI
# =============================================================================


def create_tablet_interface(
    bridge: TabletTaskLoopBridge,
    *,
    voice_enabled: bool = False,
):
    """Build the tablet browser interface.

    This function contains only UI and browser callback bindings.
    Perception and robot execution stay inside the existing application loop.
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
        """Poll the latest bridge snapshot for browser display."""

        snap = bridge.snapshot()

        return (
            snap.image_rgb,
            snap.sam_rgb,
            snap.flowpose_rgb,
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
            # Hero
            # =================================================================

            gr.HTML(
                TABLET_HERO_HTML
            )

            # =================================================================
            # Live Camera
            # =================================================================

            with gr.Group(
                elem_id="camera-panel",
                elem_classes=["art-panel"],
            ):

                gr.HTML(
                    CAMERA_HEAD_HTML
                )

                camera = gr.Image(
                    label="RealSense · Live RGB",
                    type="numpy",
                    interactive=False,
                    elem_id="camera-view",
                )

            # =================================================================
            # SAM3 + FlowPose
            # =================================================================

            with gr.Group(
                elem_classes=["art-panel"],
            ):

                gr.HTML(
                    RESULT_HEAD_HTML
                )

                with gr.Row(
                    equal_height=True,
                    elem_id="perception-results",
                ):

                    # ---------------------------------------------------------
                    # SAM3
                    # ---------------------------------------------------------

                    with gr.Column(
                        elem_classes=["result-card"],
                    ):

                        gr.HTML(
                            """
                            <div class="result-caption">
                                SAM3 · OBJECT BOUNDARY
                            </div>
                            """
                        )

                        sam_result = gr.Image(
                            label="SAM3 Segmentation",
                            type="numpy",
                            interactive=False,
                            elem_classes=["result-view"],
                        )

                    # ---------------------------------------------------------
                    # FlowPose
                    # ---------------------------------------------------------

                    with gr.Column(
                        elem_classes=["result-card"],
                    ):

                        gr.HTML(
                            """
                            <div class="result-caption">
                                FLOWPOSE · 6D ORIENTATION
                            </div>
                            """
                        )

                        flowpose_result = gr.Image(
                            label="FlowPose 6D Pose",
                            type="numpy",
                            interactive=False,
                            elem_classes=["result-view"],
                        )

            # =================================================================
            # River Transition
            # =================================================================

            gr.HTML(
                RIVER_HTML
            )

            # =================================================================
            # Control Area
            # =================================================================

            with gr.Group(
                elem_id="control-panel",
                elem_classes=["art-panel"],
            ):

                gr.HTML(
                    CONTROL_HEAD_HTML
                )

                # -------------------------------------------------------------
                # Main One-touch Grasp
                # -------------------------------------------------------------

                grasp = gr.Button(
                    "START GRASP",
                    variant="primary",
                    size="lg",
                    elem_id="grasp-btn",
                )

                if voice_enabled:
                    voice = gr.Button(
                        "VOICE GRASP · RECORD 4 S",
                        elem_id="voice-btn",
                    )
                    gr.HTML(
                        '<div class="voice-note">Uses the microphone on the robot computer.</div>'
                    )

                # -------------------------------------------------------------
                # Perception Only
                # -------------------------------------------------------------

                perceive = gr.Button(
                    "PERCEPTION ONLY · SEE",
                    elem_id="perceive-btn",
                )

                # -------------------------------------------------------------
                # Robot Status
                # -------------------------------------------------------------

                with gr.Row(
                    equal_height=True,
                ):

                    status = gr.Textbox(
                        label="ROBOT STATUS",
                        value="Connecting to robot vision...",
                        interactive=False,
                        lines=3,
                        elem_id="robot-status",
                    )

                    activity = gr.Textbox(
                        label="CURRENT ACTIVITY",
                        value="Tablet control ready",
                        interactive=False,
                        lines=3,
                        elem_id="activity-status",
                    )

                # -------------------------------------------------------------
                # Secondary Controls
                # -------------------------------------------------------------

                with gr.Row():

                    home_left = gr.Button(
                        "LEFT ARM · HOME",
                        elem_id="home-left-btn",
                    )

                    home_right = gr.Button(
                        "RIGHT ARM · HOME",
                        elem_id="home-right-btn",
                    )

                    stop = gr.Button(
                        "STOP",
                        variant="stop",
                        elem_id="stop-btn",
                    )

            # =================================================================
            # Footer
            # =================================================================

            gr.HTML(
                FOOTER_HTML
            )

        # =====================================================================
        # Browser Polling
        # =====================================================================

        timer = gr.Timer(
            value=refresh_sec,
            active=True,
        )

        timer.tick(
            refresh,
            outputs=[
                camera,
                sam_result,
                flowpose_result,
                status,
                activity,
            ],
        )

        # =====================================================================
        # Command Bindings
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
                lambda: bridge.request(TabletCommand.VOICE),
                outputs=[status, activity],
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
