# 生产模型与 CUDA 资源审计

本审计依据实际初始化和推理调用链，而非项目 README。

## 主感知链路

| 组件 | 实际入口 | 权重/资源 | 关键资源行为 |
|---|---|---|---|
| SAM3 image model | `grasp_core/perception/realsense_sam3.py::Sam3Runner` | `perception/models/sam3.pt`（约 3.3 GiB 文件） | 整模型放入选定 device；CUDA 上启用 TF32，推理使用 BF16 autocast。每帧先算 1008×1008 image backbone，再算文本 grounding 和 mask。|
| FlowNet / MeanFlow | `grasp_core/perception/flowpose_pipeline.py::FlowPoseRunner` → `inference/inference_helper.py::Flow` | `FlowNet3.pth` | 构造时直接 `.to(device)`；同时创建 Adam optimizer 和模型参数 EMA shadow copy，虽是推理路径仍常驻，增加 GPU/CPU 资源。|
| ScaleNet | 同上 | `ScaleNet3.pth` | 模型、Adam optimizer、scheduler 和 EMA 一起构造；推理时复用 FlowNet 点特征。|
| DINOv2 ViT-S/14 | `networks/dino/dino.py::DinoLoader` | 本地 DINOv2 repo + `dinov2_vits14_pretrain.pth` | `torch.hub.load(..., source='local')`，权重加载后常驻同一 GPU；每个实例 ROI 提取 intermediate layer features。|
| PointNet2 | `networks/pts_encoder/pointnet2.py` | Python model + `pointnet2_cuda` 扩展 | FlowNet 点云编码实际调用自定义 CUDA op。迁移到 RTX 5060 后必须在目标架构上验证/重新编译，不能只验证 PyTorch 能看到 GPU。|
| CPU 后处理 | `inference_helper.py`, `api_runner.py`, `flowpose_pipeline.py` | scikit-learn DBSCAN、NumPy/OpenCV | 姿态聚合包含 GPU→CPU 数据传输和 DBSCAN；生产完整链路还生成两张可视化，因此完整链路延迟会高于裸 FlowPose。|

生产应用在 `GraspDemoApp.load_models()` 中先后填充 `sam_cache` 与
`flowpose_cache`，两套模型会同时常驻，直至进程退出。没有逐次 unload 或
`empty_cache()`。这正是 `resident` 和 `pipeline` 场景的测量对象。

## 不属于默认感知 benchmark 的资源

- `add/voice.py` 的 Faster-Whisper `small` 是延迟加载：只有触发语音识别时才创建，
  `FASTER_WHISPER_DEVICE=auto` 可能令它也常驻 CUDA。若部署时会同时使用语音，需
  另跑一次完整 benchmark，并在测量前触发语音模型加载；当前四项验收不包含它。
- 夹爪触觉目录带 TensorRT/CUDA plugin，但主应用的 `grip_signal_receiver.py` 是独立
  子进程/设备通信路径，不在 SAM3→FlowPose 感知进程中自动初始化。若目标部署同时
  启用本地触觉 GPU 推理，应作为额外整机并发负载测量。
- ROS2、MuJoCo/mink、request_ik 和 RealSense 会消耗 RAM/CPU，但没有在这条感知
  进程中加载额外的 PyTorch CUDA 模型。最终整机验收仍应在这些服务同时运行时关注
  报告里的 machine GPU/RAM 指标。

## 已识别的部署风险

1. FlowPose 的推理构造器保留训练用 optimizer、scheduler 和 EMA；如果 8 GiB 预算
   不足，这是最值得先做的资源精简点，但 benchmark 不修改生产行为。
2. SAM3 checkpoint 文件大小不是显存占用；加载时参数 dtype、临时 checkpoint
   反序列化、CUDA allocator reserved 和推理 activation 都必须实测。
3. `pointnet2_cuda` 是 ABI/架构敏感的本地扩展。目标笔记本应执行
   `scripts/check_env.py`，然后再跑本 benchmark。
4. RTX 4090 等开发机的容量结果可以用于筛查，但不能替代 RTX 5060 笔记本上的
   温度、功耗限制、compute capability 和 P95 延迟验收。

