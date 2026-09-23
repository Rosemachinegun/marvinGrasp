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
├── flowpose_request_ik_tester.py           主入口：解析参数并启动完整抓取流程
├── grasp_core/                             主应用代码
│   ├── tools/                              启动器、交互 UI、轨迹工具
│   │   ├── flowpose_request_ik_app.py      完整 FlowPose + request_ik 应用
│   │   ├── request_ik_ui.py                键盘交互和动作触发
│   │   └── trajectory_*.py                 轨迹录制、检查和 CSV 工具
│   ├── perception/                          感知运行时
│   │   ├── realsense_sam3.py                RealSense 采集和 SAM3 分割
│   │   ├── flowpose_pipeline.py             FlowPose 推理、位姿和尺寸估计
│   │   └── perception_runtime.py            感知流程封装和结果输出
│   ├── planning/                            抓取规划
│   │   ├── grasp/planner.py                 抓取姿态生成
│   │   ├── grasp/target_order.py            多目标抓取顺序
│   │   └── trajectory/                      笛卡尔轨迹、插值和时间规划
│   ├── execution/                           机器人动作执行
│   │   ├── skills/grasp.py                  抓取动作
│   │   ├── skills/place.py                  放置动作
│   │   ├── skills/home.py                   HOME 动作
│   │   ├── motion_executor.py               运动执行和轨迹下发
│   │   ├── drop_monitor.py                  掉落检测
│   │   └── robot_skill_service.py            动作服务和失败恢复
│   ├── communication/                       外部设备和 ROS2 通信
│   │   ├── request_ik_transport.py          request_ik 通信传输
│   │   ├── request_ik_feedback.py           IK 执行反馈
│   │   └── gripper_signal.py                夹爪信号通信
│   ├── core/                                通用数据结构和数学运算
│   │   ├── types/                           相机、目标和机器人位姿类型
│   │   ├── math/                            向量、位姿、轴向和轨迹数学
│   │   └── io/                              目标位姿等数据读写
│   ├── config/                              参数、路径和 YAML 配置解析
│   │   ├── defaults.py                      默认参数
│   │   ├── resource_paths.py                模型和资源路径
│   │   └── yaml_loader.py                   YAML 配置加载
│   └── resources/                           机器人工具和 URDF/Xacro 资源
│       ├── tool.yaml                        工具/夹具参数
│       └── stand_v3.urf.xacro               机器人描述资源
├── perception/                              感知模型和算法源码
│   ├── sam3/                                SAM3 源码，作为普通目录纳入仓库
│   │   ├── sam3/                            SAM3 Python 包
│   │   ├── examples/                        SAM3 示例
│   │   └── pyproject.toml                   SAM3 安装配置
│   ├── flowpose/                            FlowPose 源码
│   │   ├── dataset/                         数据集和推理数据加载
│   │   ├── inference/                       推理辅助函数和 mask 处理
│   │   ├── py_runners/                      训练、验证和 RealSense 推理入口
│   │   ├── networks/                        网络结构和 PointNet2 CUDA 算子
│   │   ├── utils/                           训练、可视化和推理工具
│   │   └── configuration.yaml               FlowPose 配置
│   └── models/                              模型权重目录，权重默认不上传 Git
│       ├── FlowNet3.pth                     FlowNet 权重
│       ├── ScaleNet3.pth                   尺寸估计权重
│       ├── sam3.pt                          SAM3 checkpoint
│       ├── facebookresearch_dinov2_main/    DINOv2 源码
│       └── dinov2_vits14_pretrain.pth       DINOv2 checkpoint
├── daimon_gripper/                          夹爪相关代码
│   ├── dm_gripper_py/                       Lingkong 夹爪 gRPC SDK 源码
│   │   ├── dm_lingkong_grip_sdk/            SDK Python 包和 protobuf 接口
│   │   ├── setup.py                          SDK 安装配置
│   │   └── requirement.txt                   SDK 依赖
│   ├── dm_gripper_cam_py/                   远程相机 gRPC 客户端
│   ├── dm_gripper_tac_py/                   触觉夹爪相关代码
│   ├── grip_signal_receiver.py              夹爪信号接收
│   └── grip_signal_calibration.py           夹爪信号标定
├── eye2hand_calibration/                    Eye-to-Hand 外参标定工具
│   └── calibrate_camera_extrinsic.py        相机外参标定脚本
├── tests/                                   单元测试和动作流程测试
├── environment-flowpose.yml                 Conda 环境定义
├── env.md                                   完整部署、标定和故障排查
└── README.md                                安装、目录和使用说明
~~~

目录使用关系：

- 修改抓取策略时，优先查看 `grasp_core/planning/grasp/` 和 `grasp_core/execution/skills/`。
- 修改相机、分割或位姿推理时，查看 `grasp_core/perception/`；底层模型算法位于 `perception/`。
- 修改机器人运动或 `request_ik` 通信时，查看 `grasp_core/communication/`、`grasp_core/execution/` 和 `grasp_core/resources/`。
- 修改夹爪 SDK 时，查看 `daimon_gripper/dm_gripper_py/`；该目录包含可安装的 Python 源码。
- 修改模型路径或运行参数时，查看 `grasp_core/config/`，模型权重放在 `perception/models/`。
- `__pycache__/`、`*.egg-info/`、编译生成文件和模型权重属于运行产物，不是业务源码。

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

#### 6.1 从 ModelScope 下载模型

项目模型文件统一发布在 [ModelScope：hwkanHW/Marvin_Grasp](https://modelscope.cn/models/hwkanHW/Marvin_Grasp/files)。在项目根目录执行下面的命令，会将该模型仓库中的文件下载到程序默认使用的 `perception/models` 目录：

~~~bash
conda activate flowpose
cd /home/jjj/code/marvinGrasp

python -m pip install modelscope
modelscope download \
  --model hwkanHW/Marvin_Grasp \
  --local_dir "$PWD/perception/models"
~~~

下载后检查模型文件：

~~~bash
test -s perception/models/FlowNet3.pth
test -s perception/models/ScaleNet3.pth
test -s perception/models/sam3.pt
test -d perception/models/facebookresearch_dinov2_main
test -s perception/models/dinov2_vits14_pretrain.pth
~~~

模型权重不纳入 Git 仓库，新的机器或全新克隆的工作区都需要重新下载。若 ModelScope 仓库中的文件目录发生变化，请保持最终文件路径与上面的模型目录结构一致。

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
