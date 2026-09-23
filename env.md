# marvinGrasp / FlowPose 部署指南

本文档以当前仓库为准，目标是恢复：

~~~
RealSense → SAM3 → FlowPose → 抓取规划 → ROS2 / request_ik → 夹爪
~~~

推荐软件栈：

| 项目 | 版本 / 要求 |
|---|---|
| Ubuntu | 22.04 |
| ROS2 | Humble |
| Python | 3.10 |
| Conda 环境 | flowpose |
| PyTorch | 2.5.1 |
| PyTorch CUDA | 12.1 |
| GPU | NVIDIA CUDA GPU |
| 相机 | Intel RealSense D455 / 兼容型号 |

## 0. 安装原则

仓库中有两套旧安装说明：

- 根目录 environment-flowpose.yml 是当前推荐环境。
- perception/flowpose/README.md 中的 genpose2 环境名和旧版 PyTorch/CUDA 版本已经过时。

不要混用两套 PyTorch/CUDA 安装方式。本文档统一使用：

~~~
Python 3.10
PyTorch 2.5.1
PyTorch CUDA 12.1
CUDA Toolkit 12.1
Conda 环境 flowpose
~~~

当前主应用默认模型路径：

~~~
perception/models/FlowNet3.pth
perception/models/ScaleNet3.pth
perception/models/sam3.pt
perception/models/facebookresearch_dinov2_main/
perception/models/dinov2_vits14_pretrain.pth
~~~

## 1. 迁移前备份旧主机

在旧主机执行：

~~~
cd /path/to/marvinGrasp

git status
git branch --show-current
git rev-parse HEAD
git remote -v

conda activate flowpose
conda env export > flowpose_environment_full.yml
conda env export --from-history > flowpose_environment_history.yml
python -m pip freeze > pip_freeze.txt

python --version > environment_info.txt
python -m pip --version >> environment_info.txt
nvidia-smi >> environment_info.txt
nvcc --version >> environment_info.txt
ros2 doctor --report > ros2_doctor.txt
~~~

重点备份：

~~~
FlowNet3.pth
ScaleNet3.pth
sam3.pt
DINOv2 源码目录
DINOv2 checkpoint
旧主机的相机外参
RealSense 序列号
机器人 URDF / xacro / TCP 配置
夹爪地址和标定参数
~~~

新主机建议 checkout 完全相同的 commit：

~~~
git fetch --all
git checkout <OLD_COMMIT_SHA>
git rev-parse HEAD
~~~

## 2. 检查 Ubuntu、NVIDIA 和 ROS2

~~~
lsb_release -a
nvidia-smi
nvcc --version
~~~

nvidia-smi 必须可以识别 GPU。仅有 nvcc 不代表 NVIDIA 驱动正常。

如果驱动未安装或 nvidia-smi 失败：

~~~
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
~~~

重启后再次检查：

~~~
nvidia-smi
~~~

检查 ROS2：

~~~
source /opt/ros/humble/setup.bash
ros2 doctor --report
~~~

FlowPose 可以单独运行，但完整机器人应用需要 ROS2 Humble。

## 3. 安装系统依赖

~~~
sudo apt update

sudo apt install -y \
  build-essential gcc g++ make cmake ninja-build git pkg-config \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  libusb-1.0-0
~~~

RealSense 建议安装系统工具和开发包：

~~~
sudo apt install -y librealsense2-utils librealsense2-dev
~~~

如果找不到 librealsense2-*，需要先按 Intel librealsense 对应 Ubuntu 22.04 的软件源说明添加仓库。

检查相机工具：

~~~
rs-enumerate-devices
~~~

## 4. 获取仓库

~~~
cd /home/jjj/code
git clone https://github.com/Rosemachinegun/marvinGrasp.git
cd marvinGrasp
~~~

如果仓库已经存在：

~~~
cd /home/jjj/code/marvinGrasp
git status
git rev-parse HEAD
~~~

设置仓库根目录变量：

~~~
export REPO_ROOT=/home/jjj/code/marvinGrasp
~~~

## 5. 创建或更新 Conda 环境

必须在仓库根目录执行，因为 environment-flowpose.yml 中包含本地 SAM3 editable 安装：

~~~
cd "$REPO_ROOT"
conda env create -f environment-flowpose.yml
conda activate flowpose
~~~

如果 flowpose 环境已经存在：

~~~
conda activate flowpose
conda env update -n flowpose -f "$REPO_ROOT/environment-flowpose.yml"
~~~

确认 Python 和 pip 来自同一个环境：

~~~
which python
which pip
python --version
~~~

补充安装兼容依赖和旧脚本依赖：

~~~
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<81"
python -m pip install cutoop ultralytics
~~~

说明：

- cutoop 主要用于旧版数据集评估脚本。
- ultralytics 主要用于旧版 infer_rs.py / infer_rs_kp.py。
- 当前主应用的 SAM3 → FlowPose 流程不依赖 YOLO，但建议完整安装。
- 使用 environment-flowpose.yml 后，不要再无条件执行旧 requirements.txt，因为其中的 OpenCV 版本可能覆盖当前环境。

## 6. 安装匹配 CUDA Toolkit

环境文件中的 PyTorch 使用 CUDA 12.1。编译 PointNet2 前安装匹配的 Toolkit：

~~~
conda activate flowpose
conda install -n flowpose -c nvidia cuda-toolkit=12.1 -y
~~~

设置编译环境：

~~~
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

which nvcc
nvcc --version
~~~

CUDA Toolkit 版本应与 PyTorch 的 CUDA 版本保持一致或兼容。不要仅依赖系统 CUDA 12.8 编译当前 PyTorch 12.1 环境。

## 7. 安装 / 验证 SAM3

environment-flowpose.yml 已经包含：

~~~
-e ./perception/sam3
~~~

如果是手工创建环境，则执行：

~~~
cd "$REPO_ROOT"
python -m pip install -e perception/sam3
~~~

验证：

~~~
python -c "import sam3; print('SAM3:', sam3.__file__)"
~~~

SAM3 源码和模型权重是两件不同的东西。检查权重：

~~~
test -s "$REPO_ROOT/perception/models/sam3.pt"
~~~

如果缺少 sam3.pt，需要从旧主机或模型提供方复制，不能只安装 Python package。

## 8. 编译 PointNet2 CUDA 扩展

FlowPose 的 PointNet2 CUDA 扩展是必须组件：

~~~
conda activate flowpose

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO_ROOT/perception/flowpose/networks/pts_encoder/pointnet2_utils/pointnet2"
python setup.py install
~~~

验证时必须先导入 torch：

~~~
cd "$REPO_ROOT"
python -c "import torch; import pointnet2_cuda; print('PointNet2 CUDA OK')"
~~~

如果出现 libc10.so 找不到，通常是：

1. 没有先 import torch；
2. LD_LIBRARY_PATH 没有包含 PyTorch 的 torch/lib；
3. 扩展是用另一套 PyTorch 或 CUDA 编译的。

修复后重新执行 python setup.py install。

## 9. 准备 FlowPose 模型

检查当前模型：

~~~
test -s "$REPO_ROOT/perception/models/FlowNet3.pth"
test -s "$REPO_ROOT/perception/models/ScaleNet3.pth"
~~~

如果缺失，从旧主机复制：

~~~
cp /path/to/FlowNet3.pth "$REPO_ROOT/perception/models/FlowNet3.pth"
cp /path/to/ScaleNet3.pth "$REPO_ROOT/perception/models/ScaleNet3.pth"
~~~

旧版脚本中的 results/ckpts/... 路径不等于当前主应用默认使用的 perception/models/... 路径。

## 10. 准备 DINOv2

FlowPose 使用 DINOv2 提取 RGB 特征。推荐使用本地源码和 checkpoint：

~~~
cd "$REPO_ROOT/perception/models"

git clone https://github.com/facebookresearch/dinov2.git \
  facebookresearch_dinov2_main

wget -O dinov2_vits14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14_pretrain.pth
~~~

检查：

~~~
test -d "$REPO_ROOT/perception/models/facebookresearch_dinov2_main"
test -s "$REPO_ROOT/perception/models/dinov2_vits14_pretrain.pth"
~~~

也可以从旧主机复制 DINOv2 源码和 checkpoint。

注意：当前 DinoLoader 使用本地仓库时会以 pretrained=False 加载模型，然后再读取 checkpoint。如果本地 checkpoint 不存在，可能使用未训练的 DINO 权重，导致 FlowPose 结果不可用。

## 11. 全部依赖验证

~~~
conda activate flowpose
cd "$REPO_ROOT"

python -c "import torch; print('Torch:', torch.__version__); print('Torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
python -c "import cv2, scipy, open3d, pyrealsense2, webdataset, timm; print('FlowPose Python dependencies OK')"
python -c "import sam3; print('SAM3 OK')"
python -c "import torch; import pointnet2_cuda; print('PointNet2 CUDA OK')"
~~~

检查 FlowPose 源码导入：

~~~
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/perception/flowpose:${PYTHONPATH:-}"

python -c "from networks.pts_encoder.pointnet2 import Pointnet2ClsMSGFus; print('FlowPose network import OK')"
python -c "from networks.dino.dino import DinoLoader; print('DINO loader import OK')"
~~~

检查模型文件：

~~~
test -s "$REPO_ROOT/perception/models/FlowNet3.pth" && echo FlowNet3 OK
test -s "$REPO_ROOT/perception/models/ScaleNet3.pth" && echo ScaleNet3 OK
test -s "$REPO_ROOT/perception/models/sam3.pt" && echo SAM3 checkpoint OK
test -s "$REPO_ROOT/perception/models/dinov2_vits14_pretrain.pth" && echo DINO checkpoint OK
~~~

## 12. 验证 RealSense

~~~
rs-enumerate-devices
python -c "import pyrealsense2 as rs; print('RealSense devices:', rs.context().query_devices().size())"
~~~

如果没有设备，检查：

~~~
USB 连接
RealSense udev 规则
相机是否被其他进程占用
当前用户是否具有 USB 设备访问权限
~~~

新相机必须查询真实序列号，启动时传入：

~~~
--serial <REAL_CAMERA_SERIAL>
~~~

## 13. 完整应用依赖

完整机器人应用还需要夹爪 SDK 和 ROS2。

夹爪 SDK：

~~~
conda activate flowpose
cd "$REPO_ROOT"

python -m pip install -r daimon_gripper/dm_gripper_py/requirement.txt
python -m pip install -e daimon_gripper/dm_gripper_py
~~~

ROS2 验证：

~~~
source /opt/ros/humble/setup.bash
python -c "import rclpy; print('rclpy OK')"
python -c "from geometry_msgs.msg import PoseStamped; print('geometry_msgs OK')"
python -c "from sensor_msgs.msg import JointState; print('sensor_msgs OK')"
~~~

完整应用还需要：

~~~
request_ik 节点
机器人 URDF / xacro
base_link / marker frame
左右臂 HOME 姿态
机器人 IP
夹爪 gRPC 地址
~~~

## 14. 启动完整 FlowPose 应用

先启动 ROS2 和机器人控制节点，再执行：

~~~
source /opt/ros/humble/setup.bash
conda activate flowpose
cd "$REPO_ROOT"

python flowpose_request_ik_tester.py \
  --serial <REAL_CAMERA_SERIAL> \
  --sam3-root "$REPO_ROOT/perception/sam3" \
  --sam3-checkpoint-path "$REPO_ROOT/perception/models/sam3.pt" \
  --flow-model-path "$REPO_ROOT/perception/models/FlowNet3.pth" \
  --scale-model-path "$REPO_ROOT/perception/models/ScaleNet3.pth" \
  --dino-repo-path "$REPO_ROOT/perception/models/facebookresearch_dinov2_main" \
  --dino-ckpt-path "$REPO_ROOT/perception/models/dinov2_vits14_pretrain.pth" \
  --flowpose-device cuda
~~~

实际抓取前必须重新确认：

~~~
T_base_camera
RealSense serial
camera_joint
robot URDF / TCP
Home
Put position
Table Z
workspace boundary
gripper IP
gripper calibration
~~~

## 15. 旧版 FlowPose 脚本

旧脚本包括：

~~~
perception/flowpose/scripts/infer_rs.sh
perception/flowpose/scripts/infer_rs_kp.sh
perception/flowpose/scripts/infer.sh
~~~

这些脚本可能依赖：

~~~
results/ckpts/YOLO/mixed.pt
results/ckpts/...
~~~

使用前必须确认对应 YOLO 和 FlowPose checkpoint 存在。否则优先使用第 14 节的当前主应用入口。

旧版脚本的基本启动形式：

~~~
conda activate flowpose
cd "$REPO_ROOT/perception/flowpose"

PYTHONPATH=. python py_runners/infer_rs.py \
  --tracking \
  --realsense \
  --pretrained_flow_model_path "$REPO_ROOT/perception/models/FlowNet3.pth" \
  --pretrained_scale_model_path "$REPO_ROOT/perception/models/ScaleNet3.pth" \
  --device cuda \
  --data_mode rs
~~~

该入口仍可能要求旧版 YOLO 权重，不代表当前主应用的完整验证已经通过。

## 16. 常见问题

### 16.1 环境名错误

旧 README 中类似：

~~~
conda create -n genpose2 ...
conda activate flowpose
~~~

这是错误的。统一使用 flowpose。

### 16.2 CUDA available: False

依次检查：

~~~
nvidia-smi
which nvcc
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
~~~

nvidia-smi 失败时先修复 NVIDIA 驱动；安装 CUDA Toolkit 不能替代驱动。

### 16.3 PointNet2 编译失败

检查：

~~~
conda activate flowpose
echo "$CUDA_HOME"
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
~~~

确认 CUDA_HOME 指向与 PyTorch 匹配的 Toolkit，然后重新安装：

~~~
cd "$REPO_ROOT/perception/flowpose/networks/pts_encoder/pointnet2_utils/pointnet2"
python setup.py clean --all
python setup.py install
~~~

### 16.4 No module named sam3

~~~
conda activate flowpose
cd "$REPO_ROOT"
python -m pip install -e perception/sam3
python -c "import sam3; print(sam3.__file__)"
~~~

### 16.5 sam3.pt 找不到

~~~
test -s "$REPO_ROOT/perception/models/sam3.pt"
~~~

源码安装成功不等于模型权重存在。

### 16.6 DINO 没有权重

必须同时存在：

~~~
facebookresearch_dinov2_main/
dinov2_vits14_pretrain.pth
~~~

不要只创建空目录，否则可能加载未训练的特征提取器。

### 16.7 ultralytics 缺失

仅当使用旧版 RealSense YOLO 脚本时安装：

~~~
conda activate flowpose
python -m pip install ultralytics
~~~

### 16.8 rclpy 导入失败

先 source ROS2：

~~~
source /opt/ros/humble/setup.bash
conda activate flowpose
python -c "import rclpy; print('rclpy OK')"
~~~

如果仍失败，检查 ROS2 Python 版本与 Conda Python 是否都是 Python 3.10，并确认没有被其他 Conda 环境覆盖。

## 17. 最终验收清单

主机和软件：

~~~
[ ] Ubuntu 22.04
[ ] nvidia-smi 正常
[ ] CUDA Toolkit 12.1 可用
[ ] flowpose 环境存在
[ ] Python 3.10
[ ] PyTorch 2.5.1
[ ] torch.cuda.is_available() 为 True
[ ] PointNet2 CUDA 扩展可导入
[ ] SAM3 package 可导入
[ ] sam3.pt 存在
[ ] FlowNet3.pth 存在
[ ] ScaleNet3.pth 存在
[ ] DINOv2 源码和 checkpoint 存在
[ ] pyrealsense2 可导入
[ ] RealSense 可以被检测
[ ] ROS2 Humble 可用
[ ] rclpy / geometry_msgs / sensor_msgs 可导入
~~~

机器人：

~~~
[ ] 相机序列号已确认
[ ] 相机外参已确认
[ ] robot URDF / xacro 正确
[ ] camera_joint 正确
[ ] base_link / marker frame 正确
[ ] request_ik 节点正常
[ ] HOME 姿态已确认
[ ] Put 位姿已确认
[ ] Table Z 已确认
[ ] 夹爪服务地址正确
[ ] 夹爪标定参数正确
[ ] 机器人和夹爪网络连通
~~~

完成以上检查后，再执行完整抓取流程。
