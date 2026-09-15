"""Shared stop-state helpers for skills and publishers."""

from __future__ import annotations


def publisher_stop_requested(publisher) -> bool:
    stop_requested = getattr(publisher, "stop_requested", None)
    if callable(stop_requested):
        return bool(stop_requested())
    client = getattr(publisher, "client", None)
    ok = getattr(client, "ok", None)
    return callable(ok) and not bool(ok())
