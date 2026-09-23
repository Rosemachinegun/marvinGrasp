#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time

from dm_lingkong_grip_sdk import LingkongGrip


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive gripper position control: SDK native 0=closed, 1000=open."
    )
    parser.add_argument("--server", default="192.168.14.11:55551")
    parser.add_argument("--speed", type=int, default=50, help="10..100")
    parser.add_argument("--torque", type=int, default=50, help="10..100")
    parser.add_argument("--settle", type=float, default=0.8, help="seconds after each move")
    return parser.parse_args()


def main():
    args = parse_args()

    grip = LingkongGrip(server_address=args.server)
    try:
        print("正在初始化夹爪，会执行找零动作，请保持夹爪区域安全...")
        if not grip.grip_init():
            print("夹爪初始化失败")
            return 1

        grip.set_torque_limit(args.torque)
        grip.set_speed(args.speed)

        print("初始化成功。输入 0-1000 控制位置：0=闭合，1000=张开。输入 q 退出。")
        while True:
            value = input("目标位置(0-1000)> ").strip()
            if value.lower() in {"q", "quit", "exit"}:
                break

            try:
                sdk_pos = int(value)
            except ValueError:
                print("请输入整数 0-1000，或 q 退出")
                continue

            if sdk_pos < 0 or sdk_pos > 1000:
                print("超出范围：请输入 0-1000")
                continue

            print(f"移动到 SDK 位置 {sdk_pos}")
            if not grip.move_to_pos(sdk_pos):
                print("移动命令发送失败")
                continue

            time.sleep(args.settle)
            current = grip.read_pos()
            print(f"当前位置：SDK={current}")

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，退出")
    finally:
        grip.close()
        print("夹爪连接已关闭")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
