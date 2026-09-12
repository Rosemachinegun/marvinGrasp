# `flowpose_request_ik_tester.py` 环境文件与依赖清单

## 本项目生成的环境文件

| 文件 | 用途 |
| --- | --- |
| `environment.yml` | 创建 Python 3.10、CUDA 11.8、编译工具齐全的 Conda 环境 |
| `requirements.txt` | 安装入口程序、SAM3、FlowPose、RealSense、Daimon 夹爪和测试所需的 Python 包 |
| `FlowPose/requirements.txt` | FlowPose 上游的原始依赖参考；根目录文件已经包含本入口所需的子集 |
| `config/tool.yaml` | 抓取策略、轨迹和工具模板运行配置，不是 Python 包依赖文件 |
| `config/stand_v3.urf.xacro` | 相机相对机器人基座的外参和机器人结构配置 |

## 创建环境

```bash
conda env create -f environment.yml
conda activate graspdemo
```

更新已有环境时使用：

```bash
conda env update -f environment.yml --prune
```

## 必须单独准备的组件

以下内容不能可靠地写成普通 PyPI 依赖，需在运行前单独准备。

### 1. ROS 2 Humble

程序默认启用 ROS 2 发布和 IK 请求，使用以下 ROS 包：

- `rclpy`
- `geometry_msgs`
- `sensor_msgs`
- `visualization_msgs`
- `tf2_ros`
- `builtin_interfaces`
- `rcl_interfaces`
- `ament_index_python`
- `pinocchio`（中断后从实测关节状态计算工具端 FK）

Ubuntu 22.04 安装 ROS 2 Humble 后，每个运行终端先执行：

```bash
conda activate graspdemo
source /opt/ros/humble/setup.bash
```

如果只调试感知流程，可传入 `--no-ros2-publish`；完整的 IK/机械臂流程仍需要项目对应的 ROS 工作区和节点。

### 2. SAM3 源码

`sam3` 不是当前项目的一部分。准备 SAM3 源码后，可任选一种方式：

```bash
python -m pip install -e /path/to/sam3
```

或运行时指定源码根目录：

```bash
python flowpose_request_ik_tester.py --sam3-root /path/to/sam3
```

SAM3 权重默认从 `model/sam3.pt` 读取。

### 3. FlowPose PointNet2 CUDA 扩展

FlowPose 会在导入时加载 `pointnet2_cuda`，创建 Conda 环境后编译一次：

```bash
conda activate graspdemo
export CUDA_HOME="$CONDA_PREFIX"
python -m pip install -v ./FlowPose/networks/pts_encoder/pointnet2_utils/pointnet2
```

### 4. DINOv2 源码和权重

FlowPose 使用 DINOv2 特征。推荐下载 `facebookresearch/dinov2` 源码和 `dinov2_vits14_pretrain.pth`，然后显式传入：

```bash
python flowpose_request_ik_tester.py \
  --dino-repo-path /path/to/dinov2 \
  --dino-ckpt-path /path/to/dinov2_vits14_pretrain.pth
```

未指定本地源码时会尝试通过 `torch.hub` 下载，离线设备会失败。

### 5. 模型、硬件与系统库

运行前确认：

- 模型文件：`model/sam3.pt`、`model/FlowNet3.pth`、`model/ScaleNet3.pth`
- Intel RealSense D435 及系统级 `librealsense`/udev 规则
- 支持 CUDA 11.8 运行时的 NVIDIA 驱动
- 可用的图形桌面或 X11 转发，因为主程序使用 OpenCV 窗口
- Daimon 双夹爪网络可达；夹爪 Python SDK 会由根目录 `requirements.txt` 以可编辑模式安装

## 快速检查

```bash
conda activate graspdemo
source /opt/ros/humble/setup.bash

python -c "import cv2, gradio, grpc, numpy, pyrealsense2, torch, yaml; print(torch.__version__, torch.cuda.is_available())"
python -c "import rclpy; from geometry_msgs.msg import PoseStamped; print('ROS 2 OK')"
python -c "import pointnet2_cuda; print('PointNet2 CUDA OK')"
python flowpose_request_ik_tester.py --help
```

注意：Conda 环境必须使用 Python 3.10，才能加载 Ubuntu 22.04/ROS 2 Humble 提供的 `rclpy` 二进制模块。
