#!/usr/bin/env python3
"""Check software imports and a CUDA kernel without connecting to the robot."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    failures = []
    modules = (
        "numpy", "scipy", "cv2", "yaml", "torch", "torchvision", "open3d",
        "pyrealsense2", "sam3.model.sam3_image_processor", "sam3.model_builder",
        "tensorboardX", "webdataset", "sklearn", "gradio", "faster_whisper",
        "sounddevice", "grpc", "dm_lingkong_grip_sdk", "dmrobotics",
        "mujoco", "mink", "qpsolvers", "pointnet2_cuda", "rclpy", "tf2_ros",
        "geometry_msgs.msg", "sensor_msgs.msg", "trajectory_msgs.msg",
        "visualization_msgs.msg", "xacro",
    )
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"[OK] {name}", flush=True)
        except Exception as exc:
            failures.append(name)
            print(f"[失败] {name}: {exc}", flush=True)
    try:
        sys.path.insert(0, str(root / "perception" / "flowpose"))
        importlib.import_module("inference.inference_helper")
        print("[OK] FlowPose 推理入口")
    except Exception as exc:
        failures.append("FlowPose")
        print(f"[失败] FlowPose: {exc}")
    try:
        import torch
        import pointnet2_cuda

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch 无法访问 GPU，请检查驱动")
        points = torch.arange(24, device="cuda", dtype=torch.float32).reshape(1, 8, 3)
        temp = torch.full((1, 8), 1e10, device="cuda")
        indices = torch.zeros((1, 2), dtype=torch.int32, device="cuda")
        pointnet2_cuda.furthest_point_sampling_wrapper(1, 8, 2, points, temp, indices)
        torch.cuda.synchronize()
        if indices.cpu().tolist() != [[0, 7]]:
            raise RuntimeError(f"PointNet2 采样结果异常：{indices}")
        print(f"[OK] CUDA 实际运算 / PointNet2 / {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        failures.append("CUDA / PointNet2")
        print(f"[失败] CUDA / PointNet2: {exc}")

    models = root / "perception" / "models"
    for name in ("sam3.pt", "FlowNet3.pth", "ScaleNet3.pth", "dinov2_vits14_pretrain.pth"):
        path = models / name
        if path.is_file() and path.stat().st_size > 0:
            print(f"[存在] 模型 {path}（未加载验证）")
        else:
            print(f"[待准备] 模型 {path}；从原主机复制或使用应用参数指定路径")
    if not (models / "facebookresearch_dinov2_main" / "hubconf.py").is_file():
        print("[待准备] DINOv2 源码/缓存；缺失时应用可能需要联网下载")
    print("[待确认] RealSense USB 权限、机器人控制工作空间、网络和手眼/夹爪标定；参见 env.md")
    if failures:
        print(f"软件检查失败（{len(failures)} 项），请修复以上错误后重跑。")
        return 1
    print("软件检查通过；模型及硬件就绪情况需按以上提示确认。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
