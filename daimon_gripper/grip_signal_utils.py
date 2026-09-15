from __future__ import annotations

import os
from urllib.parse import urlsplit


def clamp(value: int | float, low: int | float, high: int | float):
    return max(low, min(high, value))


def disable_proxy_for_host(addr: str) -> None:
    """Avoid grpc using HTTP proxy for the local gripper server."""
    host = urlsplit(addr if "://" in addr else "//" + addr).hostname
    if not host:
        return
    for key in ("NO_PROXY", "no_proxy"):
        values = os.environ.get(key, "")
        hosts = [item.strip() for item in values.split(",") if item.strip()]
        if host not in hosts:
            hosts.append(host)
        os.environ[key] = ",".join(hosts)


def read_feedback_sample(grip, *, source: str = "direct") -> str:
    pos = grip.read_pos()
    current = grip.read_cur_current()
    temp = grip.read_cur_tempture()
    return f"feedback source={source} pos={pos} current={current} temp={temp}"


def read_check_sample(grip) -> str:
    pos = grip.read_pos()
    current = grip.read_cur_current()
    temp = grip.read_cur_tempture()
    return f"CHECK OK: pos={pos}, current={current}, temp={temp}"


def set_motion_profile(grip, *, speed: int, torque: int) -> None:
    grip.set_speed(speed)
    grip.set_torque_limit(torque)
