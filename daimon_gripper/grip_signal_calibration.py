from __future__ import annotations

import argparse
import json

try:
    from .grip_signal_config import (
        CALIBRATION_CACHE_PATH,
        CALIBRATION_FIELDS,
        CALIBRATION_INVALIDATED_ATTR,
        CALIBRATION_SOURCE_ATTR,
    )
except ImportError:
    from grip_signal_config import (
        CALIBRATION_CACHE_PATH,
        CALIBRATION_FIELDS,
        CALIBRATION_INVALIDATED_ATTR,
        CALIBRATION_SOURCE_ATTR,
    )


def cache_key_for_server(server: str) -> str:
    return str(server).strip()


def load_calibration_cache() -> dict:
    try:
        with CALIBRATION_CACHE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"读取夹爪标定缓存失败: {exc}", flush=True)
        return {}
    return data if isinstance(data, dict) else {}


def apply_cached_calibration(args: argparse.Namespace) -> None:
    cache = load_calibration_cache()
    entry = cache.get(cache_key_for_server(args.server))
    if not isinstance(entry, dict):
        return
    try:
        values = {field: int(entry[field]) for field in CALIBRATION_FIELDS}
    except (KeyError, TypeError, ValueError):
        print(f"忽略无效夹爪标定缓存: server={args.server}", flush=True)
        return
    args.clamp_pos = values["clamp_pos"]
    args.open_pos = values["open_pos"]
    args.max_itinerary = values["max_itinerary"]
    args.speed_coe = values["speed_coe"]
    setattr(args, CALIBRATION_SOURCE_ATTR, "cache")
    print(
        "加载夹爪标定缓存: "
        f"hand={args.hand} server={args.server} clamp_pos={args.clamp_pos} "
        f"open_pos={args.open_pos} max_itinerary={args.max_itinerary} "
        f"speed_coe={args.speed_coe}",
        flush=True,
    )


def save_cached_calibration(args: argparse.Namespace) -> None:
    cache = load_calibration_cache()
    cache[cache_key_for_server(args.server)] = {
        field: int(getattr(args, field)) for field in CALIBRATION_FIELDS
    }
    try:
        with CALIBRATION_CACHE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        print(f"保存夹爪标定缓存失败: {exc}", flush=True)
        return
    print(f"已保存夹爪标定缓存: {CALIBRATION_CACHE_PATH}", flush=True)


def invalidate_cached_calibration(args: argparse.Namespace, reason: str) -> None:
    if getattr(args, CALIBRATION_INVALIDATED_ATTR, False):
        return
    setattr(args, CALIBRATION_INVALIDATED_ATTR, True)
    server = cache_key_for_server(args.server)
    cache = load_calibration_cache()
    if server not in cache:
        return
    cache.pop(server, None)
    try:
        with CALIBRATION_CACHE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        print(
            f"删除失效夹爪标定缓存失败: server={server} reason={reason}: {exc}",
            flush=True,
        )
        return
    print(f"已删除失效夹爪标定缓存: server={server} reason={reason}", flush=True)
