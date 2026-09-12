"""Map short Chinese grasp commands to object names used by the pipeline."""

from __future__ import annotations

import re


def parse_voice_target(text: str) -> str | None:
    """Return a supported canonical target, or None for an unknown command."""
    command = re.sub(r"[\s，。！？、,!.?]", "", text).lower()
    if (
        not command
        or any(word in command for word in ("不要", "别", "取消", "停止"))
        or not any(verb in command for verb in ("夹", "抓", "拿", "取"))
    ):
        return None

    if "螺丝刀" in command or "改锥" in command:
        return "screwdriver_handle"
    if ("黄色" in command or "黄" in command) and any(
        word in command for word in ("方块", "积木", "立方体")
    ):
        return "yellow_cube"
    if "笔" in command:
        return "pen"
    if "玩具" in command:
        return "toy"
    return None
