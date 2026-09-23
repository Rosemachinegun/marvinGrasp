# 导入 SDK
import time
from dm_lingkong_grip_sdk import LingkongGrip

# 1. 创建夹爪实例
# 参数：server_address - gRPC CAN 服务地址（根据实际情况修改）
grip = LingkongGrip(server_address="192.168.127.10:55551")

# 2. 初始化夹爪（会执行夹紧、找零位等动作）
if grip.grip_init():
    print("夹爪初始化成功")
else:
    print("夹爪初始化失败")
    exit(1)

# 3. 设置力矩限制（10~100）,速度(10~100)
grip.set_torque_limit(50)
grip.set_speed(50)

# # 4. 移动夹爪到目标位置（0~1000，0=闭合，1000=张开）
grip.move_to_pos(500)   # 移动到一半

time.sleep(1.0)
# 5. 查询当前位置（返回值 0~1000）
pos = grip.read_pos()
print(f"当前位置: {pos}")

# 6. 读取当前温度、力矩参数、速度参数
temp = grip.read_cur_tempture()
print(f"当前温度: {temp}")
torque = grip.read_torque_limit()
print(f"力矩参数: {torque}")
speed = grip.read_speed()
print(f"速度参数: {speed}")

# 7. 关闭夹爪
grip.close()
