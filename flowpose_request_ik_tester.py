#!/usr/bin/env python3
"""Entry point for the RealSense + SAM3 + FlowPose + IK grasp demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Gradio/httpx does not accept the generic ``socks://`` scheme commonly
# exported by desktop proxy clients.  Clear only those invalid proxy values
# before importing the application (which imports Gradio); valid HTTP(S)
# proxy settings are left untouched.
for _proxy_var in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    if os.environ.get(_proxy_var, "").lower().startswith("socks://"):
        os.environ.pop(_proxy_var, None)

from grasp_core.tools.flowpose_request_ik_app import main


if __name__ == "__main__":
    raise SystemExit(main())
