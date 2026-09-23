#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

from dm_lingkong_grip_sdk import LingkongGrip


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Dual gripper interactive position control. "
            "One slider controls both hands: 0=closed, 1000=open."
        )
    )
    parser.add_argument("--left-server", default="192.168.14.11:55551")
    parser.add_argument("--right-server", default="192.168.10.11:55551")
    parser.add_argument("--speed", type=int, default=50, help="10..100")
    parser.add_argument("--torque", type=int, default=50, help="10..100")
    parser.add_argument("--send-interval", type=float, default=0.08, help="minimum seconds between move commands")
    parser.add_argument("--poll-interval", type=float, default=0.2, help="seconds between position polls")
    parser.add_argument("--no-init", action="store_true", help="skip grip_init, only connect and send commands")
    return parser.parse_args()


class HandController:
    def __init__(self, name, server, speed, torque, poll_interval, no_init):
        self.name = name
        self.server = server
        self.speed = speed
        self.torque = torque
        self.poll_interval = poll_interval
        self.no_init = no_init

        self.grip = None
        self.ready = False
        self.status = "等待初始化"
        self.position = None
        self.error = None

        self._targets = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def set_target(self, position):
        if not self.ready:
            return
        while True:
            try:
                self._targets.get_nowait()
            except queue.Empty:
                break
        self._targets.put(position)

    def stop(self):
        self._stop.set()
        self._targets.put(None)

    def join(self, timeout=2.0):
        self._thread.join(timeout)

    def _run(self):
        try:
            self.status = f"连接 {self.server}"
            self.grip = LingkongGrip(server_address=self.server)

            if not self.no_init:
                self.status = "初始化中，请保持夹爪区域安全"
                if not self.grip.grip_init():
                    self.status = "初始化失败"
                    self.error = "grip_init failed"
                    return

            if not self.grip.set_torque_limit(self.torque):
                self.status = "设置力矩失败"
                self.error = "set_torque_limit failed"
                return

            if not self.grip.set_speed(self.speed):
                self.status = "设置速度失败"
                self.error = "set_speed failed"
                return

            self.ready = True
            self.status = "就绪"
            next_poll = 0.0

            while not self._stop.is_set():
                try:
                    target = self._targets.get(timeout=0.03)
                except queue.Empty:
                    target = None

                if target is not None:
                    self.status = f"移动到 {target}"
                    if not self.grip.move_to_pos(target):
                        self.status = "移动命令失败"
                    else:
                        self.status = f"目标 {target}"

                now = time.monotonic()
                if now >= next_poll:
                    self.position = self.grip.read_pos()
                    next_poll = now + self.poll_interval

        except Exception as exc:
            self.status = "异常"
            self.error = str(exc)
        finally:
            self.ready = False
            if self.grip is not None:
                self.grip.close()
            if self.error:
                self.status = f"{self.status}: {self.error}"
            else:
                self.status = "已关闭"


class DualGripApp:
    def __init__(self, root, left, right, send_interval):
        self.root = root
        self.left = left
        self.right = right
        self.send_interval = send_interval
        self.last_sent = 0.0
        self.pending_after = None

        self.target_var = tk.IntVar(value=500)
        self.left_status = tk.StringVar(value="左手: 等待启动")
        self.right_status = tk.StringVar(value="右手: 等待启动")
        self.left_pos = tk.StringVar(value="左手位置: --")
        self.right_pos = tk.StringVar(value="右手位置: --")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.left.start()
        self.right.start()
        self.refresh_status()

    def _build_ui(self):
        self.root.title("Dual Lingkong Gripper Position")
        self.root.geometry("620x260")

        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(frame, text="双手夹爪位置控制", font=("", 16, "bold"))
        title.pack(anchor=tk.W)

        value_row = ttk.Frame(frame)
        value_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(value_row, text="目标位置").pack(side=tk.LEFT)
        self.value_label = ttk.Label(value_row, text="500", width=6, anchor=tk.E)
        self.value_label.pack(side=tk.RIGHT)

        self.scale = ttk.Scale(
            frame,
            from_=0,
            to=1000,
            orient=tk.HORIZONTAL,
            command=self.on_slider,
        )
        self.scale.set(self.target_var.get())
        self.scale.pack(fill=tk.X, pady=(8, 6))

        end_labels = ttk.Frame(frame)
        end_labels.pack(fill=tk.X)
        ttk.Label(end_labels, text="0 闭合").pack(side=tk.LEFT)
        ttk.Label(end_labels, text="1000 张开").pack(side=tk.RIGHT)

        button_row = ttk.Frame(frame)
        button_row.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(button_row, text="闭合", command=lambda: self.set_position(0)).pack(side=tk.LEFT)
        ttk.Button(button_row, text="半开", command=lambda: self.set_position(500)).pack(side=tk.LEFT, padx=8)
        ttk.Button(button_row, text="张开", command=lambda: self.set_position(1000)).pack(side=tk.LEFT)
        ttk.Button(button_row, text="退出", command=self.on_close).pack(side=tk.RIGHT)

        status_grid = ttk.Frame(frame)
        status_grid.pack(fill=tk.X, pady=(16, 0))
        ttk.Label(status_grid, textvariable=self.left_status).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_grid, textvariable=self.right_status).grid(row=0, column=1, sticky=tk.W, padx=(24, 0))
        ttk.Label(status_grid, textvariable=self.left_pos).grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Label(status_grid, textvariable=self.right_pos).grid(row=1, column=1, sticky=tk.W, padx=(24, 0), pady=(6, 0))

    def on_slider(self, raw_value):
        target = int(float(raw_value))
        self.target_var.set(target)
        self.value_label.configure(text=str(target))
        self.schedule_send(target)

    def set_position(self, target):
        self.scale.set(target)
        self.send_target(target)

    def schedule_send(self, target):
        now = time.monotonic()
        if now - self.last_sent >= self.send_interval:
            self.send_target(target)
            return

        if self.pending_after is not None:
            self.root.after_cancel(self.pending_after)
        delay_ms = int((self.send_interval - (now - self.last_sent)) * 1000)
        self.pending_after = self.root.after(delay_ms, lambda: self.send_target(self.target_var.get()))

    def send_target(self, target):
        self.pending_after = None
        target = max(0, min(1000, int(target)))
        self.last_sent = time.monotonic()
        self.left.set_target(target)
        self.right.set_target(target)

    def refresh_status(self):
        self.left_status.set(f"左手: {self.left.status}")
        self.right_status.set(f"右手: {self.right.status}")
        self.left_pos.set(f"左手位置: {self._format_pos(self.left.position)}")
        self.right_pos.set(f"右手位置: {self._format_pos(self.right.position)}")
        self.root.after(200, self.refresh_status)

    def on_close(self):
        if self.pending_after is not None:
            self.root.after_cancel(self.pending_after)
            self.pending_after = None
        self.left.stop()
        self.right.stop()
        self.left.join()
        self.right.join()
        self.root.destroy()

    @staticmethod
    def _format_pos(position):
        if position is None:
            return "--"
        return str(position)


def main():
    args = parse_args()
    left = HandController("left", args.left_server, args.speed, args.torque, args.poll_interval, args.no_init)
    right = HandController("right", args.right_server, args.speed, args.torque, args.poll_interval, args.no_init)

    root = tk.Tk()
    DualGripApp(root, left, right, args.send_interval)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
