#!/usr/bin/env python3
"""双目相机查看器主入口：启动左右远程相机并显示拼接画面。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from daimon_stuff.dm_gripper_cam_py.viewer_config import (  # noqa: E402
    build_argparser,
    camera_specs_from_args,
)
from daimon_stuff.dm_gripper_cam_py.dashboard import make_camera_dashboard  # noqa: E402
from daimon_stuff.dm_gripper_cam_py.worker import CameraWorker  # noqa: E402

DASHBOARD_WINDOW = "Dual Camera Dashboard"
QUIT_KEYS = {ord("q"), 27}


def open_workers(workers: list[CameraWorker], read_timeout: float) -> None:
    """依次打开所有相机，并启动后台读取线程。"""

    for worker in workers:
        worker.open()
        worker.start(read_timeout)


def collect_frames(
    workers: list[CameraWorker],
    stats_interval: float,
) -> tuple[bool, list]:
    """收集每路相机最新帧，同时返回是否还有相机处于打开状态。"""

    any_open = False
    frames = []
    for worker in workers:
        if worker.cap.isOpened():
            any_open = True
        frames.append(worker.latest_frame())
        worker.maybe_print_stats(stats_interval)
    return any_open, frames


def close_workers(workers: list[CameraWorker]) -> None:
    """释放所有相机 worker，保证异常退出时也能关闭远程流。"""

    for worker in workers:
        worker.close()


def main() -> int:
    """程序入口：解析参数、运行显示循环、处理退出。"""

    args = build_argparser().parse_args()
    workers = [CameraWorker(spec, args) for spec in camera_specs_from_args(args)]

    try:
        open_workers(workers, args.read_timeout)
        cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)

        while True:
            any_open, frames = collect_frames(workers, args.stats_interval)
            if not any_open:
                break

            cv2.imshow(DASHBOARD_WINDOW, make_camera_dashboard(frames))
            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                break
    except KeyboardInterrupt:
        pass
    finally:
        close_workers(workers)
        cv2.destroyAllWindows()
        print("[dual-camera] released", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
