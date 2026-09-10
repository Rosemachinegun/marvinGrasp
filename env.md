# marvinGrasp 新机器人 / 新主机部署指南

> 项目仓库：`https://github.com/Rosemachinegun/marvinGrasp`
>
> 目标：在新的 Ubuntu 主机和新的 Marvin 机器人上，完整恢复 `RealSense → SAM3 → FlowPose → Grasp Policy → IK / Trajectory → Gripper → Pick / Put / Recovery` 实机抓取链路。

---

## 1. 迁移时需要准备什么

这次迁移不是单纯 `git clone`。

完整系统实际上包含：

```text
代码
+
Python / AI 环境
+
模型权重
+
ROS2 / 机器人控制环境
+
RealSense
+
相机外参
+
机器人 TF / URDF / TCP
+
夹爪通信与标定
+
网络
+
Home / Put / Table 等工作空间参数
```

其中可以直接继承的通常是：

```text
SAM3 提示词逻辑
FlowPose 推理逻辑
Cube symmetry policy
Long-object / pen policy
Trajectory smoothing
Drop recovery state machine
Pause / resume logic
```

而新的机器人上必须重新确认或重新标定的通常是：

```text
T_base_camera
RealSense serial
Robot URDF / TCP
Robot / Gripper IP
Gripper calibration
Home
PUT positions
Table Z
Workspace boundary
```

---

# 2. 推荐的软件栈

为了优先把 Demo 跑通，建议尽量保持和旧系统一致。

| 模块 | 推荐 |
|---|---|
| OS | Ubuntu 22.04 |
| ROS | ROS2 Humble |
| Python | Python 3.10 |
| Conda env | `flowpose` |
| GPU | NVIDIA GPU |
| AI Framework | PyTorch + CUDA |
| Camera | Intel RealSense D455 |
| Segmentation | SAM3 |
| 6D Pose | FlowPose |
| Feature Backbone | DINOv2 |
| Robot | Marvin 7DoF |
| Gripper | Daimon / DM-DATACLAW |
| Communication | ROS2 + gRPC |

不建议迁移过程中同时升级：

```text
Ubuntu
ROS
Python
PyTorch
CUDA
FlowPose
SAM3
```

先复现旧环境，再单独升级。

---

# 3. 旧主机先做备份

如果旧主机还能运行，迁移之前先保存当前真实可运行状态。

```bash
cd ~/graspdemo

git status
git branch --show-current
git rev-parse HEAD
git remote -v
```

记录：

```text
当前 branch
当前 commit SHA
是否存在未提交修改
远程仓库地址
```

建议新机器最终 checkout **完全相同的 commit**，不要单纯按照 README 的默认 branch。

---

## 3.1 导出 Python 环境

```bash
conda activate flowpose

conda env export > flowpose_environment_full.yml
conda env export --from-history > flowpose_environment_history.yml
pip freeze > pip_freeze.txt
```

记录系统环境：

```bash
python --version > environment_info.txt
pip --version >> environment_info.txt
nvidia-smi >> environment_info.txt
nvcc --version >> environment_info.txt
```

ROS 信息：

```bash
ros2 doctor --report > ros2_doctor.txt
```

---

## 3.2 备份模型

Git 通常不会保存大型模型权重。

重点确认：

```text
sam3.pt
FlowNet3.pth
ScaleNet3.pth
DINOv2 repo / checkpoint / cache
其他 FlowPose checkpoint
```

例如：

```bash
cp -a ~/graspdemo/model ~/graspdemo_model_backup
```

还建议查看：

```bash
ls ~/.cache/torch/hub/
```

如果 DINOv2 是通过 Torch Hub 加载的，旧机器可能依赖这里的缓存。

---

# 4. 新主机基础环境

## 4.1 Ubuntu

推荐：

```text
Ubuntu 22.04
```

检查：

```bash
lsb_release -a
```

---

## 4.2 NVIDIA Driver

检查：

```bash
nvidia-smi
```

必须能够识别 GPU。

---

## 4.3 CUDA / PyTorch

真正需要确认的是 PyTorch 能使用 CUDA。

安装好 PyTorch 后：

```bash
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

预期：

```text
CUDA available: True
```

---

# 5. Conda / Python 环境

建议：

```text
Conda env: flowpose
Python: 3.10
```

创建：

```bash
conda create -n flowpose python=3.10 -y
conda activate flowpose
```

随时检查：

```bash
which python
which pip
python --version
```

预期路径应该类似：

```text
.../envs/flowpose/bin/python
.../envs/flowpose/bin/pip
```

避免出现：

```text
sam3 安装在 base
但项目使用 flowpose
```

这种环境错位。

---

# 6. Clone marvinGrasp

```bash
cd ~
git clone https://github.com/Rosemachinegun/marvinGrasp.git
cd marvinGrasp
```

如果知道旧主机 commit：

```bash
git fetch --all
git checkout <OLD_COMMIT_SHA>
```

确认：

```bash
git rev-parse HEAD
```

和旧主机一致。

---

# 7. Python 基础依赖

进入：

```bash
conda activate flowpose
cd ~/marvinGrasp
```

先确认基础包：

```bash
python - <<'PY'
import numpy
import cv2
import scipy
import torch

print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("SciPy:", scipy.__version__)
print("Torch:", torch.__version__)
print("Basic AI environment OK")
PY
```

项目还可能使用：

```text
numpy
scipy
opencv-python
torch
torchvision
grpcio
protobuf
pyyaml
tqdm
Pillow
```

具体版本优先参考旧电脑的：

```text
pip_freeze.txt
```

---

# 8. SAM3

如果出现：

```text
ModuleNotFoundError: No module named 'sam3'
```

说明缺的是 **SAM3 Python package**，而不只是权重。

SAM3 需要：

```text
SAM3 source code
+
sam3.pt
```

---

## 8.1 安装 SAM3

```bash
conda activate flowpose

cd ~
git clone https://github.com/facebookresearch/sam3.git
cd sam3

pip install -e .
```

检查：

```bash
python -c "import sam3; print(sam3.__file__)"
```

进一步检查：

```bash
python - <<'PY'
from sam3.model_builder import build_sam3_image_model
print("SAM3 import OK")
PY
```

---

## 8.2 SAM3 权重

准备：

```text
sam3.pt
```

推荐放到：

```text
~/marvinGrasp/model/sam3.pt
```

最终结构例如：

```text
marvinGrasp/
└── model/
    └── sam3.pt
```

注意：

```text
sam3 package != sam3.pt
```

一个是网络代码，一个是模型参数，两者都需要。

---

# 9. FlowPose

FlowPose 至少涉及：

```text
FlowPose source
FlowNet3.pth
ScaleNet3.pth
DINOv2
PyTorch
NumPy
OpenCV
SciPy
```

旧主机需要复制：

```text
FlowNet3.pth
ScaleNet3.pth
```

推荐统一存到：

```text
marvinGrasp/
└── model/
    ├── FlowNet3.pth
    ├── ScaleNet3.pth
    └── sam3.pt
```

---

## 9.1 检查硬编码旧路径

特别注意旧主机可能使用：

```text
/home/kewei/...
/home/kewei/.cache/torch/hub/...
```

新主机用户名或目录变化后会失效。

建议逐渐把模型路径统一放进：

```text
config/models.yaml
```

例如：

```yaml
sam3_checkpoint: /home/<USER>/marvinGrasp/model/sam3.pt
flownet_checkpoint: /home/<USER>/marvinGrasp/model/FlowNet3.pth
scalenet_checkpoint: /home/<USER>/marvinGrasp/model/ScaleNet3.pth
dinov2_root: /home/<USER>/dinov2
```

不要长期依赖散落在 Python 文件里的绝对路径。

---

# 10. DINOv2

FlowPose 使用 DINO 特征，因此 DINO 也必须可用。

旧主机检查：

```bash
ls ~/.cache/torch/hub/
```

如果存在：

```text
facebookresearch_dinov2...
```

说明旧系统可能使用 Torch Hub cache。

新机器可以：

```text
重新下载
```

或：

```text
复制旧 cache
```

最终目标是 FlowPose 初始化 DINO 时不会出现：

```text
repository not found
checkpoint not found
No module named ...
```

---

# 11. RealSense D455

需要：

```text
librealsense2
pyrealsense2
```

如果需要 ROS node：

```text
realsense2_camera
```

---

## 11.1 检查 D455

连接相机：

```bash
rs-enumerate-devices
```

确认：

```text
Intel RealSense D455
Serial Number
Firmware Version
```

---

## 11.2 Python 检查

```bash
python - <<'PY'
import pyrealsense2 as rs

print("pyrealsense2 OK")

ctx = rs.context()

for dev in ctx.devices:
    print("Name:", dev.get_info(rs.camera_info.name))
    print("Serial:", dev.get_info(rs.camera_info.serial_number))
PY
```

---

# 12. 修改 RealSense Serial

旧代码中如果存在固定 serial，例如：

```python
DEFAULT_SERIAL = "..."
```

新 D455 必须重新查询：

```bash
rs-enumerate-devices
```

然后：

```text
修改默认 serial
```

或运行时：

```bash
--serial <NEW_SERIAL>
```

建议以后放入：

```text
config/hardware.yaml
```

例如：

```yaml
camera:
  type: D455
  serial: "XXXXXXXXXXXX"
```

---

# 13. ROS2 Humble

推荐：

```text
ROS2 Humble
```

确认：

```bash
source /opt/ros/humble/setup.bash
ros2 --help
```

Python 需要至少访问：

```text
rclpy
tf2_ros
geometry_msgs
sensor_msgs
trajectory_msgs
```

检查：

```bash
python - <<'PY'
import rclpy
import tf2_ros

print("ROS Python OK")
PY
```

如果：

```text
No module named rclpy
```

不要优先尝试：

```bash
pip install rclpy
```

应首先确认：

```bash
source /opt/ros/humble/setup.bash
```

以及当前 Python / ROS 环境关系。

---

# 14. Marvin 机器人 ROS Workspace

`marvinGrasp` 是上层抓取项目，并不等于完整机器人控制系统。

整体链路：

```text
RealSense
    ↓
SAM3
    ↓
FlowPose
    ↓
Grasp Policy
    ↓
marvinGrasp
    ↓
Target Pose / Cartesian Trajectory
    ↓
IK / QP
    ↓
Robot Controller
    ↓
Robot Driver
    ↓
7DoF Arm
```

因此需要从旧主机或机器人团队处准备：

```text
marvin_description
URDF / Xacro
IK
QP Controller
Robot Controller
Robot Driver
request_ik_tester
```

---

## 14.1 Build ROS Workspace

例如：

```bash
cd ~/robot_ws
source /opt/ros/humble/setup.bash

colcon build

source install/setup.bash
```

建议写入终端启动脚本：

```bash
source /opt/ros/humble/setup.bash
source ~/robot_ws/install/setup.bash
conda activate flowpose
```

---

# 15. 检查 ROS Topics

项目可能依赖：

```text
/joint_states

/control/request_ik_tester/target_poseL
/control/request_ik_tester/target_poseR

/control/request_ik_tester/target_trajectoryL
/control/request_ik_tester/target_trajectoryR
```

检查：

```bash
ros2 node list
ros2 topic list
```

确认目标 topic 都存在。

---

# 16. TF / Robot Frames

至少确认：

```text
base_link
camera_rgb_link
left_tool / left_tcp
right_tool / right_tcp
```

查看：

```bash
ros2 run tf2_ros tf2_echo base_link camera_rgb_link
```

```bash
ros2 run tf2_ros tf2_echo base_link left_tool
```

```bash
ros2 run tf2_ros tf2_echo base_link right_tool
```

也可以：

```bash
ros2 run tf2_tools view_frames
```

---

# 17. 新机器人必须重新标定相机外参

这是迁移中最重要的一项之一。

视觉输出最终转换：

```text
camera frame object pose
        ↓
T_base_camera
        ↓
base_link object pose
```

数学上：

```text
T_base_object
=
T_base_camera
×
T_camera_object
```

因此如果 D455 在新机器人上的安装位置不同：

```text
旧 T_base_camera
```

不能继续直接使用。

---

## 17.1 重新标定

可以继续使用项目已有标定程序，例如：

```bash
python3 calibrate_camera_extrinsic.py \
    --live-realsense \
    --save-samples calib_samples.json \
    --output camera_extrinsic_result.json \
    --write-xacro stand_v3.urf.xacro
```

然后验证：

```bash
ros2 run tf2_ros tf2_echo base_link camera_rgb_link
```

---

## 17.2 RViz 验证

在真正运动机械臂之前：

```text
Real object
     ↓
FlowPose estimated frame
     ↓
base_link transformed marker
```

必须在 RViz 里基本重合。

如果这里不对：

```text
不要继续自动抓取
```

---

# 18. Robot URDF / TCP / IK

如果新机器人是：

```text
同型号
同尺寸
同 TCP
同 base 定义
```

一般可以继续复用大部分 IK。

但仍需验证。

如果新的机器人存在：

```text
机械臂长度不同
关节零位不同
URDF 不同
Joint limit 不同
TCP 不同
工具安装不同
```

则必须重新确认：

```text
URDF
TCP Transform
Joint Limits
Collision Geometry
IK / QP Model
Home Joint Configuration
```

否则可能出现：

```text
视觉目标正确
但 IK 使用旧机器人模型
```

导致实机运动错误。

---

# 19. Daimon / DM-DATACLAW 夹爪环境

Python 通信至少涉及：

```text
grpcio
protobuf
```

检查：

```bash
python -c "import grpc; print(grpc.__version__)"
```

项目本身还包含：

```text
daimon_stuff
```

---

# 20. 新机器人夹爪 IP

旧项目可能存在：

```text
left_server  = 192.168.14.11:55551
right_server = 192.168.10.11:55551
```

新机器人不要直接默认这些地址仍然有效。

需要确认：

```text
Left Gripper IP
Right Gripper IP
Robot Controller IP
Left Camera IP
Right Camera IP
```

---

# 21. 网络

旧系统可能同时涉及：

```text
192.168.10.x
192.168.14.x
```

检查：

```bash
ip addr
ip route
```

确认目标设备：

```bash
ping <LEFT_GRIPPER_IP>
ping <RIGHT_GRIPPER_IP>
ping <ROBOT_IP>
```

必要时为一个网卡添加多个 subnet 地址，例如：

```bash
sudo ip addr add 192.168.10.123/24 dev enp3s0
sudo ip addr add 192.168.14.123/24 dev enp3s0
```

但具体 PC IP 应该按新机器人的网络规划设置，不要直接照抄旧机器。

---

# 22. 新夹爪重新标定

如果实际换了新夹爪，不建议直接复制旧：

```text
.gripper_calibration_cache.json
```

需要重新确认：

```text
fully open position
fully closed position
empty close limit
object gripping position
torque
hold torque
current
stall threshold
drop threshold
```

否则下面逻辑都可能错误：

```text
抓取成功判定
空夹判定
掉落检测
失败恢复
```

---

# 23. tool.yaml

这类参数不能认为换机器人以后仍然成立。

重点检查：

```text
force_object_z
forced_object_z_m
pregrasp_distance_m
lift_distance_m
pose_relative
rotation_constraint
IK downward tilt
gripper parameters
```

例如如果存在：

```yaml
force_object_z: true
forced_object_z_m: 0.67
```

意味着项目对世界坐标中的物体 Z 有强假设。

换机器人 / 换桌子后，需要重新确认：

```text
base_link Z
table Z
object Z
safe Z
```

---

# 24. Home

Home 不应该第一次就高速测试。

即使是同型号机器人，也建议：

```text
10% Speed
→ No Object
→ One Arm
→ Home
```

确认：

```text
Left Home
Right Home
```

不会：

```text
跨越中心线
腕部内翻
IK 跳解
撞桌
撞身体
```

---

# 25. PUT 坐标

PUT 坐标通常和：

```text
机器人 base
工作台位置
桌面高度
机器人安装位置
```

强相关。

例如：

```python
FIXED_PUT_RIGHT_XYZ = (...)
FIXED_PUT_LEFT_XYZ  = (...)
```

换机器人后建议重新测。

特别确认：

```text
right put x/y/z
left put x/y/z
release height
arc height
workspace safety
```

---

# 26. 工作空间几何

新机器人建议重新测量：

```text
base_link origin
table plane Z
minimum safe Z
maximum usable Z
body centerline y=0
left workspace
right workspace
camera workspace
```

项目中的抓取 policy 可以继续使用类似：

```text
Left arm → y > 0
Right arm → y < 0
Do not cross y = 0
Approach inward
Elbow stays on own side
Wrist stays on own side
Avoid joint limits
Prefer continuous IK solution
```

但这些规则依赖新机器人的 base frame 定义是否一致。

---

# 27. 推荐的新机器人配置结构

长期建议把硬件绑定参数从 Python 中拆出来。

```text
config/
├── hardware.yaml
├── calibration.yaml
├── models.yaml
├── tool.yaml
├── workspace.yaml
└── profiles/
    ├── robot_old.yaml
    └── robot_new.yaml
```

---

## 27.1 hardware.yaml

```yaml
robot:
  ip: "ROBOT_IP"

camera:
  serial: "D455_SERIAL"

gripper:
  left_server: "192.168.x.x:55551"
  right_server: "192.168.x.x:55551"

frames:
  base: "base_link"
  camera: "camera_rgb_link"
  left_tcp: "left_tool"
  right_tcp: "right_tool"
```

---

## 27.2 models.yaml

```yaml
sam3_checkpoint: /home/<USER>/marvinGrasp/model/sam3.pt

flowpose:
  flownet_checkpoint: /home/<USER>/marvinGrasp/model/FlowNet3.pth
  scalenet_checkpoint: /home/<USER>/marvinGrasp/model/ScaleNet3.pth

dinov2_root: /home/<USER>/dinov2
```

---

## 27.3 calibration.yaml

```yaml
camera_extrinsic:
  parent: base_link
  child: camera_rgb_link

  translation:
    x: 0.0
    y: 0.0
    z: 0.0

  quaternion:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
```

---

## 27.4 workspace.yaml

```yaml
table_z: 0.0

home:
  left: []
  right: []

put:
  left: [0.0, 0.0, 0.0]
  right: [0.0, 0.0, 0.0]

workspace:
  centerline_y: 0.0
  min_z: 0.0
  max_z: 0.0
```

---

# 28. 推荐运行顺序

不要第一次就启动完整自动抓取。

推荐：

```text
① Ubuntu / Driver / CUDA
        ↓
② Conda / Python
        ↓
③ PyTorch CUDA
        ↓
④ marvinGrasp
        ↓
⑤ SAM3
        ↓
⑥ FlowPose
        ↓
⑦ DINOv2
        ↓
⑧ RealSense
        ↓
⑨ Vision Only
        ↓
⑩ ROS2 / Robot Workspace
        ↓
⑪ TF
        ↓
⑫ Camera Extrinsic Calibration
        ↓
⑬ RViz Pose Validation
        ↓
⑭ IK Only
        ↓
⑮ Gripper Only
        ↓
⑯ Home
        ↓
⑰ Cube Pick
        ↓
⑱ PUT
        ↓
⑲ Pen / Screwdriver Policy
        ↓
⑳ Drop / Retry / Pause Recovery
        ↓
㉑ Full Auto Pipeline
```

---

# 29. 分阶段验收 Checklist

## Stage A：GPU

```bash
nvidia-smi
```

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

要求：

```text
True
```

---

## Stage B：SAM3

```bash
python -c "import sam3; print(sam3.__file__)"
```

要求：

```text
Import success
```

并确认：

```text
model/sam3.pt
```

存在。

---

## Stage C：FlowPose

确认：

```text
FlowNet3.pth
ScaleNet3.pth
DINOv2
```

全部能加载。

先做：

```text
offline image
```

或：

```text
camera inference only
```

不要控制机械臂。

---

## Stage D：RealSense

```bash
rs-enumerate-devices
```

```bash
python -c "import pyrealsense2"
```

要求：

```text
RGB
Depth
Intrinsics
Depth Scale
Serial
```

全部正常。

---

## Stage E：ROS

```bash
ros2 node list
ros2 topic list
```

确认：

```text
joint_states
IK node
controller
robot driver
```

正常。

---

## Stage F：TF

```bash
ros2 run tf2_ros tf2_echo base_link camera_rgb_link
```

```bash
ros2 run tf2_ros tf2_echo base_link left_tool
```

```bash
ros2 run tf2_ros tf2_echo base_link right_tool
```

要求：

```text
TF continuous
No missing transform
Frame names correct
```

---

## Stage G：Camera Calibration

重新得到：

```text
T_base_camera
```

然后通过 RViz 验证：

```text
estimated object frame
≈
real object
```

---

## Stage H：IK

先发送一个：

```text
known
safe
reachable
slow
```

目标。

不要从 FlowPose 直接控制实机。

---

## Stage I：Gripper

分别测试：

```text
Left open
Left close
Left feedback

Right open
Right close
Right feedback
```

重新标定：

```text
empty-close
object-grip
drop
```

反馈范围。

---

## Stage J：Home

```text
Single arm
Low speed
No object
```

确认：

```text
No violent jump
No IK flip
No collision
```

---

## Stage K：Cube

Cube 是最适合作为第一件测试物体的对象。

先验证：

```text
Detection
Segmentation
FlowPose
TF
Policy
IK
Pick
```

不要第一件就测试 pen / screwdriver。

---

## Stage L：PUT

重新标定：

```text
RIGHT PUT
LEFT PUT
```

确认：

```text
Final XYZ
Release Height
Trajectory Height
End Orientation
```

---

## Stage M：Long Object

再测试：

```text
pen
yellow_screwdriver_handle
red_screwdriver_handle
```

确认：

```text
Policy Y = physical long axis
Policy Z = world +Z
Policy X = Y × Z
```

以及：

```text
+Y sign points to correct robot side
```

再验证 YAML 中类似：

```text
Y -0.04 / -0.05
```

的偏移是否始终加到正确的一侧。

---

## Stage N：Recovery

最后测试：

```text
S pause
S resume
Home
Grip failure
Object drop
Retry
Return Home
```

因为这些功能都会真正触发机器人运动，不应该在基础坐标和 IK 尚未确认前测试。

---

# 30. 常见问题

## `No module named sam3`

说明：

```text
SAM3 package 未安装
```

而不是：

```text
sam3.pt 缺失
```

处理：

```bash
cd ~/sam3
pip install -e .
```

验证：

```bash
python -c "import sam3"
```

---

## `sam3.pt not found`

说明：

```text
Python package 已安装
但 checkpoint 路径错误
```

检查：

```bash
ls ~/marvinGrasp/model/
```

---

## `torch.cuda.is_available() == False`

优先检查：

```text
NVIDIA Driver
PyTorch CUDA build
Conda environment
```

而不是 FlowPose。

---

## `No module named rclpy`

优先检查：

```bash
source /opt/ros/humble/setup.bash
```

不要直接认为需要 `pip install rclpy`。

---

## RealSense 找不到

检查：

```bash
rs-enumerate-devices
lsusb
```

然后再排查：

```text
USB
librealsense
permissions
serial
```

---

## Vision 正确，但机械臂抓偏

优先检查：

```text
T_base_camera
camera frame convention
base_link
TCP
object Z override
```

不要第一时间怀疑 SAM3。

---

## 末端走到错误姿态

优先区分：

```text
object pose orientation error
policy error
TCP convention
IK multiple-solution jump
trajectory discontinuity
```

---

## 抓取正确但 PUT 错

重点检查：

```text
FIXED_PUT_LEFT_XYZ
FIXED_PUT_RIGHT_XYZ
table Z
new robot base
release height
```

---

# 31. 新机器人必改项总表

| 配置 | 是否可以直接复制旧值 |
|---|---|
| Git code | 可以 |
| SAM3 logic | 可以 |
| FlowPose policy | 可以 |
| Cube policy | 可以 |
| Long-object policy | 可以 |
| Model checkpoints | 可以复制 |
| Python package versions | 建议复现 |
| RealSense serial | 不可以 |
| Camera extrinsic | 不可以 |
| Robot IP | 需确认 |
| Gripper IP | 需确认 |
| Gripper calibration | 建议重新做 |
| URDF | 需确认 |
| TCP transform | 需确认 |
| Joint limits | 需确认 |
| Home | 必须验证 |
| PUT XYZ | 建议重新标定 |
| Table Z | 必须重新确认 |
| Workspace | 必须重新确认 |

---

# 32. 最终环境树

```text
Ubuntu 22.04
│
├── NVIDIA Driver
│   └── CUDA / PyTorch
│
├── Conda
│   └── flowpose / Python 3.10
│       │
│       ├── PyTorch
│       ├── NumPy
│       ├── OpenCV
│       ├── SciPy
│       │
│       ├── SAM3
│       │   └── sam3.pt
│       │
│       ├── FlowPose
│       │   ├── FlowNet3.pth
│       │   ├── ScaleNet3.pth
│       │   └── DINOv2
│       │
│       ├── pyrealsense2
│       └── grpc / protobuf
│
├── ROS2 Humble
│   │
│   ├── rclpy
│   ├── tf2
│   ├── marvin_description
│   ├── URDF
│   ├── IK / QP
│   ├── controller
│   └── robot driver
│
├── RealSense
│   ├── librealsense
│   └── D455
│
├── Network
│   ├── Robot
│   ├── Left Gripper
│   └── Right Gripper
│
└── Calibration
    ├── T_base_camera
    ├── Gripper calibration
    ├── Home
    ├── PUT
    ├── Table Z
    └── Workspace
```

---

# 33. 建议最终目标

新主机迁移完成后，应该达到下面的状态：

```text
Camera
  ↓
RGB-D
  ↓
SAM3
  ↓
Mask
  ↓
FlowPose
  ↓
6D Pose
  ↓
T_base_camera
  ↓
Object Pose in base_link
  ↓
Grasp Policy
  ↓
IK / QP
  ↓
Smooth Cartesian Trajectory
  ↓
Robot
  ↓
Gripper
  ↓
Pick
  ↓
Put
  ↓
Drop Detection / Retry / Home
```

不要把：

```text
程序能启动
```

当成迁移成功。

真正的迁移完成标准应该是：

```text
视觉坐标正确
TF 正确
IK 正确
夹爪正确
路径正确
异常恢复正确
```

并且在新机器人上完成至少一次稳定的：

```text
Detect
→ Pose
→ Pick
→ Put
→ Home
```

闭环。

---

# 34. 推荐迁移原则

一句话总结：

> **代码和算法策略可以继承；模型权重要复制；软件环境要复现；相机外参、机器人运动学/TCP、网络、夹爪标定以及 Home / Put / Table / Workspace 参数要针对新机器人重新确认。**
