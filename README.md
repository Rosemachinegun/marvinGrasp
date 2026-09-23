# marvinGrasp

基于 RealSense、SAM3、FlowPose、ROS2 和 request_ik 的 Marvin 机器人视觉抓取系统。

~~~text
RealSense → SAM3 分割 → FlowPose 6D 位姿 → 抓取规划 → request_ik → 夹爪
~~~

## 功能

- RealSense RGB-D 图像采集
- SAM3 开放词汇目标分割
- FlowPose 6D 位姿和尺寸估计
- 目标过滤、聚类和抓取排序
- Box、Cuboid、长物体抓取策略
- 平滑笛卡尔轨迹规划
- ROS2 request_ik 目标发布
- 夹爪控制、放置和失败恢复

## 文件架构

核心模块之间的调用关系如下：

~~~mermaid
flowchart TD
    U[flowpose_request_ik_tester.py] --> A[grasp_core/tools/flowpose_request_ik_app.py]

    A --> C[grasp_core/perception/]
    A --> P[grasp_core/planning/]
    A --> E[grasp_core/execution/]
    A --> M[grasp_core/communication/]

    C --> R[RealSense]
    C --> S[perception/sam3/]
    C --> F[perception/flowpose/]
    C --> W[perception/models/]

    S --> SM[sam3.pt]
    F --> FM[FlowNet3.pth / ScaleNet3.pth]
    F --> D[DINOv2]
    W --> SM
    W --> FM
    W --> D

    C --> T[TargetObjectPose]
    T --> P
    P --> E
    E --> M
    M --> I[ROS2 / request_ik]
    M --> G[夹爪 gRPC / TCP]
~~~

关键目录：

~~~text
marvinGrasp/
├── flowpose_request_ik_tester.py       主入口
├── grasp_core/
│   ├── perception/                     RealSense、SAM3、FlowPose
│   ├── planning/                       抓取目标和轨迹规划
│   ├── execution/                      抓取、放置、HOME、恢复
│   ├── communication/                  ROS2、IK、夹爪通信
│   ├── core/                           位姿、数学和数据结构
│   └── config/                         参数和路径配置
├── perception/
│   ├── sam3/                           SAM3 源码
│   ├── flowpose/                       FlowPose 源码和 PointNet2 CUDA 源码
│   └── models/                         模型权重，默认不上传 Git
├── daimon_gripper/                     夹爪 SDK 和接收器
├── environment-flowpose.yml            Conda 环境
├── env.md                              完整部署和故障排查
└── README.md                           安装和使用说明
~~~

## 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Ubuntu 22.04 |
| Python | 3.10 |
| Conda 环境 | flowpose |
| PyTorch | 2.5.1 |
| CUDA | 12.1 |
| ROS2 | Humble |
| GPU | NVIDIA CUDA GPU |
| 相机 | Intel RealSense D455 或兼容型号 |

详细迁移、驱动、标定和故障排查说明见 [env.md](env.md)。

## 安装

### 1. 获取代码

~~~bash
cd /home/jjj/code
git clone https://github.com/Rosemachinegun/marvinGrasp.git
cd marvinGrasp
~~~

### 2. 安装系统依赖

~~~bash
sudo apt update
sudo apt install -y \
  build-essential gcc g++ make cmake ninja-build git pkg-config \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  libusb-1.0-0
~~~

确认 NVIDIA 驱动和 ROS2：

~~~bash
nvidia-smi
source /opt/ros/humble/setup.bash
ros2 doctor --report
~~~

RealSense 工具：

~~~bash
sudo apt install -y librealsense2-utils librealsense2-dev
rs-enumerate-devices
~~~

### 3. 创建 Conda 环境

必须在仓库根目录执行：

~~~bash
conda env create -f environment-flowpose.yml
conda activate flowpose
~~~

如果环境已经存在：

~~~bash
conda activate flowpose
conda env update -n flowpose -f environment-flowpose.yml
~~~

补充依赖：

~~~bash
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<81"
python -m pip install cutoop ultralytics
~~~

### 4. 安装 CUDA Toolkit 并编译 PointNet2

~~~bash
conda activate flowpose
conda install -n flowpose -c nvidia cuda-toolkit=12.1 -y

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"

cd perception/flowpose/networks/pts_encoder/pointnet2_utils/pointnet2
python setup.py install
cd /home/jjj/code/marvinGrasp
~~~

验证：

~~~bash
cd /home/jjj/code/marvinGrasp
python -c "import torch; import pointnet2_cuda; print('PointNet2 CUDA OK')"
~~~

### 5. 安装 SAM3

environment-flowpose.yml 已包含仓库内的 SAM3 editable 安装。手工安装时执行：

~~~bash
python -m pip install -e perception/sam3
python -c "import sam3; print(sam3.__file__)"
~~~

### 6. 准备模型

模型目录：

~~~text
perception/models/
├── FlowNet3.pth
├── ScaleNet3.pth
├── sam3.pt
├── facebookresearch_dinov2_main/
└── dinov2_vits14_pretrain.pth
~~~

仓库中已有 FlowNet3、ScaleNet3 和 sam3 时，检查：

~~~bash
test -s perception/models/FlowNet3.pth
test -s perception/models/ScaleNet3.pth
test -s perception/models/sam3.pt
~~~

准备 DINOv2：

~~~bash
cd perception/models
git clone https://github.com/facebookresearch/dinov2.git facebookresearch_dinov2_main
wget -O dinov2_vits14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14_pretrain.pth
cd ../..
~~~

DINOv2 源码和 checkpoint 都必须存在，否则 FlowPose 可能使用未训练的特征提取器。

### 7. 验证安装

~~~bash
conda activate flowpose
cd /home/jjj/code/marvinGrasp

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import cv2, scipy, open3d, pyrealsense2, webdataset, timm; print('Python dependencies OK')"
python -c "import sam3; print('SAM3 OK')"
python -c "import torch; import pointnet2_cuda; print('PointNet2 CUDA OK')"
rs-enumerate-devices
~~~

GPU 推理前，torch.cuda.is_available() 必须为 True。

### 8. 安装夹爪 SDK

完整机器人应用还需要夹爪 gRPC SDK：

~~~bash
conda activate flowpose
cd /home/jjj/code/marvinGrasp

python -m pip install -r daimon_gripper/dm_gripper_py/requirement.txt
python -m pip install -e daimon_gripper/dm_gripper_py
~~~

验证 ROS2 Python 接口：

~~~bash
source /opt/ros/humble/setup.bash
python -c "import rclpy; from geometry_msgs.msg import PoseStamped; print('ROS2 Python OK')"
~~~

## 使用

### 启动完整应用

启动前确认：

- ROS2 Humble 已 source
- request_ik 节点已启动
- RealSense 已连接
- 模型文件已准备
- 相机序列号和相机外参正确
- 机器人 URDF、TCP、HOME 和夹爪参数正确

启动：

~~~bash
source /opt/ros/humble/setup.bash
conda activate flowpose
cd /home/jjj/code/marvinGrasp

python flowpose_request_ik_tester.py \
  --serial CAMERA_SERIAL \
  --sam3-root /home/jjj/code/marvinGrasp/perception/sam3 \
  --sam3-checkpoint-path /home/jjj/code/marvinGrasp/perception/models/sam3.pt \
  --flow-model-path /home/jjj/code/marvinGrasp/perception/models/FlowNet3.pth \
  --scale-model-path /home/jjj/code/marvinGrasp/perception/models/ScaleNet3.pth \
  --dino-repo-path /home/jjj/code/marvinGrasp/perception/models/facebookresearch_dinov2_main \
  --dino-ckpt-path /home/jjj/code/marvinGrasp/perception/models/dinov2_vits14_pretrain.pth \
  --flowpose-device cuda
~~~

将 CAMERA_SERIAL 替换成 rs-enumerate-devices 查询到的真实序列号。

### 界面按键

| 按键 | 操作 |
|---|---|
| A | 自动执行采集、SAM3、FlowPose 和抓取 |
| Z | 执行 SAM3 + FlowPose 感知流程 |
| B | 对最近一次 SAM3 结果执行 FlowPose |
| C | 发布当前目标 |
| S | 暂停或停止当前动作 |
| H | 左右机械臂回 HOME |
| L | 夹爪闭合 |
| P | 夹爪释放 |
| Q / Esc | 退出程序 |

默认情况下，A 执行完整自动流程。手动分步操作时增加：

~~~bash
--auto-pipeline-on-a FALSE
~~~

此时使用 Z、B、C 分步执行感知和目标发布。

### 常用参数

~~~bash
--prompts "pen, screwdriver, toy"
--sam3-device cuda
--flowpose-device cuda
--score-threshold 0.1
--no-sam3-roi-filter
--target-order-max-volume-m3 0.0005
~~~

查看全部参数：

~~~bash
python flowpose_request_ik_tester.py --help
~~~

### 旧版 FlowPose 脚本

旧版脚本可能依赖 YOLO 和 results/ckpts/ 下的旧模型，不建议作为当前主流程入口：

~~~bash
conda activate flowpose
cd perception/flowpose

PYTHONPATH=. python py_runners/infer_rs.py \
  --tracking \
  --realsense \
  --pretrained_flow_model_path ../../models/FlowNet3.pth \
  --pretrained_scale_model_path ../../models/ScaleNet3.pth \
  --device cuda \
  --data_mode rs
~~~

## 安全注意事项

首次运行必须确认：

- 机器人工作空间和桌面高度
- 相机到机器人 base_link 的外参
- 机器人 HOME / Put 位姿
- TCP 和抓取方向
- request_ik topic
- 夹爪 IP、位置范围和力矩参数

建议先关闭自动抓取和夹爪动作，仅验证相机、SAM3 和 FlowPose 输出；确认位姿坐标和轨迹正确后，再连接真实机器人执行抓取。
