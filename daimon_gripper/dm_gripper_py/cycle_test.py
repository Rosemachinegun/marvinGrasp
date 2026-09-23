from dm_lingkong_grip_sdk import LingkongGrip
import time
import threading
        
if __name__ == "__main__":
    def wait_for_position(grip, target_pos: int, *, label: str, sample_hz: float = 20.0, timeout_s: float = 60.0, tolerance: int = 10, stable_count: int = 3) -> bool:
        period = 1.0 / float(sample_hz)
        start = time.monotonic()
        next_tick = start
        ok_count = 0

        print(f"[{label}] 等待到位 target={target_pos}")
        while True:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
                now = time.monotonic()
            else:
                next_tick = now

            pos = grip.read_pos()
            temp = grip.read_cur_tempture()
            
            print(f"[{label}] 位置={pos}, 目标位置={target_pos}, 温度={temp}")

            if pos != -1 and abs(pos - target_pos) <= tolerance:
                ok_count += 1
                if ok_count >= stable_count:
                    return True
            else:
                ok_count = 0

            if now - start >= timeout_s:
                print(f"[{label}] 到位超时，最后位置={pos}")
                return False

            next_tick += period

    def run_speed_sweep():
        grip = LingkongGrip(server_address="192.168.127.10:55551")
        try:
            if not grip.grip_init():
                return

            # 3.6倍减速比的电机无法设置下面力矩
            # 设置力矩电流10~100, 更改下面值从10到100，当前100对应的是500
            grip.set_torque_limit(100)

            speeds = [20, 40, 60, 80, 100]
            cycle = 0
            while cycle < 3:
                cycle += 1
                print(f"\n========== 循环测试 cycle={cycle} (Ctrl+C 退出) ==========")

                for speed in speeds:
                    print(f"\n========== 速度测试 {speed} ==========")
                    if not grip.set_speed(speed):
                        print(f"[speed={speed}] 设置速度失败")
                        continue

                    time.sleep(0.2)

                    print(f"[speed={speed}] 开到 1000")
                    if not grip.move_to_pos(1000):
                        print(f"[speed={speed}] open 指令失败")
                        continue
                    if not wait_for_position(grip, 1000, label=f"cycle={cycle}/speed={speed}/open"):
                        print(f"[speed={speed}] open 等待失败")
                        continue

                    time.sleep(0.2)

                    print(f"[speed={speed}] 关到 0")
                    if not grip.move_to_pos(0):
                        print(f"[speed={speed}] close 指令失败")
                        continue
                    if not wait_for_position(grip, 0, label=f"cycle={cycle}/speed={speed}/close"):
                        print(f"[speed={speed}] close 等待失败")
                        continue

                print(f"\n[cycle={cycle}] 一轮测试完成，继续下一轮... (Ctrl+C 退出)")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt: 用户退出测试。")
        finally:
            grip.close()

    run_speed_sweep()
