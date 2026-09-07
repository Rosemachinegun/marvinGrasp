#!/usr/bin/env python3
"""Compatibility entry point for the Daimon tactile dashboard."""
"""调用tactile dashboard的兼容入口, 主要用于在命令行中直接运行tactile dashboard, 方便调试和测试.具体grasp职能待确定"""


from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from daimon_stuff.dm_gripper_tac_py.tac import (  # noqa: F401
    SensorManager,
    depth_to_u8,
    disable_proxy_for_host,
    draw_flow,
    main,
    make_dashboard,
    make_options,
    parse_args,
    resize_keep,
)


if __name__ == "__main__":
    main()
