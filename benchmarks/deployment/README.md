# Deployment Benchmark

这套 benchmark 直接调用项目正在使用的 `Sam3Runner`、`FlowPoseRunner`、
`run_sam3_job` 和 `run_flowpose_job`。四个场景各自在全新进程中执行，避免
PyTorch CUDA allocator 缓存污染“加载前”基线：

1. `sam3`：只加载 SAM3，执行图像 + 文本提示推理。
2. `flowpose`：加载 FlowNet、ScaleNet、DINOv2 和 PointNet2 CUDA 扩展，使用
   录制 RGB-D 与实例 mask 推理。
3. `resident`：SAM3 与 FlowPose 全部模型同时常驻，并依次执行两段推理。
4. `pipeline`：每次都执行生产路径的 SAM3 后处理、SAM3 mask → FlowPose、
   位姿后处理和两次 FlowPose 可视化；默认连续测量 100 次。

## 准备输入

推荐从实际 D435 工作位录制一组对齐数据：彩色 PNG、uint16 深度 PNG（或
NPY）、相机元数据 JSON，以及供 FlowPose 单测使用的 binary mask。元数据格式：

```json
{
  "depth_scale": 0.001,
  "color_intrinsics": {"fx": 604.3, "fy": 604.0, "cx": 327.2, "cy": 248.9}
}
```

`pipeline` 必须使用 SAM3 的真实输出，不会用人工 mask 兜底。如果提示词没有检出
物体，该场景明确失败，避免产生虚假的低显存/低延迟结果。`flowpose` 和
`resident` 未提供 `--mask` 时会用中央矩形 mask，并在报告中标记
`synthetic_mask=true`；它适合资源冒烟测试，不适合作为最终部署验收数据。

也可以让每个隔离场景直接采集一帧：

```bash
conda run -n flowpose python benchmarks/deployment/run.py \
  --camera --camera-serial 419622073235 --prompt pen
```

## 执行

在目标 RTX 5060 笔记本、与生产程序相同的 conda 环境中运行：

```bash
conda activate flowpose
python benchmarks/deployment/run.py \
  --color fixtures/workcell_color.png \
  --depth fixtures/workcell_depth.png \
  --mask fixtures/workcell_mask.png \
  --metadata fixtures/workcell_meta.json \
  --prompt pen \
  --iterations 100 --warmup 3
```

快速验证单个场景：

```bash
python benchmarks/deployment/run.py --scenarios sam3 \
  --color fixtures/workcell_color.png --iterations 3 --warmup 1
```

默认输出到 `benchmark_results/<时间>/`：每个场景一个原始 JSON，加上合并的
`report.json` 和 `report.md`。`benchmark_results/` 应作为本机测量产物，不提交
模型输入中可能包含的现场图像。

## 指标口径

- GPU：`nvidia-smi` 采样整机显存、进程显存、利用率；PyTorch 同时记录精确的
  allocator allocated/reserved 峰值。
- RAM：采样进程 RSS、系统 used/available；每轮额外记录 RSS 用于增长回归。
- CPU：进程和整机利用率均为时间窗百分比，多核进程值可能超过 100%。
- 延迟：计时边界前后均执行 `torch.cuda.synchronize()`；warmup 不进入统计。
- 泄漏：对 100 轮首尾窗口中位数和线性斜率同时判断。阈值为 GPU 64 MiB、
  RSS 128 MiB。PyTorch `reserved` 增长会单独报告，不能只看 `allocated`。
- 整机峰值是周期采样值；很短的尖峰以 PyTorch peak allocator 为准。

默认按 8192 MiB VRAM、至少预留 512 MiB 做容量判断。可通过
`--target-vram-mib` 和 `--vram-headroom-mib` 调整。RAM 默认按 16384 MiB 总量并
预留 2048 MiB，可用 `--target-ram-mib`、`--ram-headroom-mib` 调整。若不是在 RTX 5060 上执行，
结论最多为 `capacity_candidate_only`：它不能证明目标机的 CUDA 扩展兼容性、
功耗降频或真实延迟。

详细的生产代码审计见 [MODEL_AUDIT.md](MODEL_AUDIT.md)。

