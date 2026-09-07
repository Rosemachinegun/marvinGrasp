# Daimon Stuff

`daimon_stuff` 收集了 Daimon 夹爪、远程相机和触觉检测相关的 Python 工具。这个目录主要用于本地调试、双夹爪控制、双相机预览，以及触觉/滑移检测实验。
![System Architecture](pipeline.png)

## 目录结构

| 路径 | 说明 |
|---|---|
| `dm_gripper_py/` | 夹爪控制 SDK 与基础测试脚本 |
| `dm_gripper_cam_py/` | 远程鱼眼相机客户端、双相机查看器和 gRPC/UDP 视频流逻辑 |
| `dm_gripper_tac_py/` | 触觉感知、力估计、滑移检测和 TensorRT/CUDA 推理相关工具 |
| `dual_camera_viewer.py` | 顶层双相机查看入口 |
| `grip_signal_receiver.py` | 夹爪信号接收脚本 |
| `gripper.sh` | 夹爪相关启动脚本 |
| `README_cam.md` | 相机查看器与夹爪参数的补充说明 |

## 快速开始

建议在虚拟环境中按需安装依赖。下面的命令默认从项目根目录执行。顶层 `requirement.txt` 覆盖夹爪、相机和触觉基础流程；GPU/TensorRT 专用依赖单独安装。

安装常用基础依赖：

```bash
cd daimon_stuff
python -m pip install -r requirement.txt
```

安装夹爪 SDK：

```bash
cd daimon_stuff/dm_gripper_py
python -m pip install .
```

安装相机客户端依赖：

```bash
cd daimon_stuff/dm_gripper_cam_py
python -m pip install -r requirements.txt
```

如果需要触觉模块：

```bash
cd daimon_stuff/dm_gripper_tac_py
python -m pip install .
```

如果需要触觉 GPU 推理：

```bash
cd daimon_stuff/dm_gripper_tac_py
python -m pip install ".[gpu]"
dmrobotics trt build
```

GPU 环境也可参考 `dm_gripper_tac_py/requirements-gpu.txt` 和 `dm_gripper_tac_py/README.md`。

## 依赖文件说明

| 文件 | 覆盖范围 |
|---|---|
| `requirement.txt` | 夹爪、相机和触觉 CPU/基础流程的常用依赖 |
| `dm_gripper_py/requirement.txt` | 夹爪 SDK 的 gRPC/protobuf 依赖 |
| `dm_gripper_cam_py/requirements.txt` | 相机客户端依赖，包含 `numpy`、`opencv-python` 等 |
| `dm_gripper_tac_py/setup.py` | 触觉模块 CPU/基础依赖 |
| `dm_gripper_tac_py/requirements-gpu.txt` | 触觉 GPU 环境的固定版本依赖 |

## 常用命令

夹爪基础测试：

```bash
cd daimon_stuff/dm_gripper_py
python test_grip.py
```

双夹爪交互控制：

```bash
cd daimon_stuff/dm_gripper_py
python dual_interactive_position.py
```

双相机预览：

```bash
cd daimon_stuff
python dual_camera_viewer.py
```

相机默认地址：

| 设备 | 地址 |
|---|---|
| 左手 | `192.168.10.10` |
| 右手 | `192.168.10.11` |

## 调参入口

夹爪闭合行为常用参数包括：

| 参数 | 作用 | 建议 |
|---|---|---|
| `--speed` | 闭合速度 | 软物体可用 `30-50`，快速抓取可用 `60` |
| `--torque` | 最大夹取力矩 | 夹得过紧时降到 `20` 或 `15` |
| `--hold-torque` | 接触后的保持力 | 通常保持 `10` |
| `--min-pos` | 最小闭合位置 | 默认 `300`，夹不到时可试 `250` |
| `--poll-interval` | 状态检测周期 | 推荐 `0.03-0.08` 秒 |
| `--stall-samples` | 停滞检测次数 | 响应慢可降到 `3`，误停止可升到 `6-8` |

更多相机链路和夹爪参数说明见 `README_cam.md`。

## 注意事项

- 运行前确认夹爪和相机 IP 与脚本配置一致。
- 远程相机依赖 OpenCV、gRPC 和视频解码环境。
- GPU 触觉推理需要 NVIDIA driver、CUDA、cuDNN 和 TensorRT 环境匹配。
- 子目录中各自保留了更详细的安装与测试说明。
# DaimonGeneral
# DaimonGeneral
