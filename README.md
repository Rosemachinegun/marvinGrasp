# marvinGrasp 代码架构

marvinGrasp 是一个基于 RealSense、SAM3、FlowPose、ROS2 和 request_ik 的机器人抓取系统。代码按“感知、目标排序、抓取规划、动作执行、底层通信”分层组织。

## 一键配置环境

适用于 **Ubuntu 22.04 x86_64 + 已安装可用 NVIDIA 驱动**。在项目根目录运行：

```bash
bash setup_env.sh
source .marvin-env/activate.sh
```

首次安装需要联网和 sudo 权限（可能提示输入系统密码），CUDA / PyTorch 下载较大。
脚本自动安装系统依赖、ROS2 Humble 和缺失的 RealSense USB 权限规则（首次需重新插拔相机）；没有 Conda 时安装 Miniforge；默认创建独立的
`marvin-grasp` 环境，安装 `environment-flowpose.yml` 中的依赖及本地 SAM3 / 夹爪 SDK，
编译 PointNet2，并验证 Python 导入和实际 CUDA 运算。重跑会更新同名环境；不会自动启动机器人。
新终端使用上面的 `source` 命令即可激活，不需要手动拼接 ROS / Conda 配置。

常用选项：

```bash
# 接入已经编译好的机器人控制工作空间
bash setup_env.sh --robot-setup ~/robot_ws/install/setup.bash

# 系统依赖和 ROS 已就绪时跳过 apt
bash setup_env.sh --skip-system

# 只检查，不安装；自定义环境名时，安装和检查都传同一个名字
bash setup_env.sh --check
bash setup_env.sh --env-name my-grasp
bash setup_env.sh --help
```

安装日志和激活脚本位于 `.marvin-env/`（不提交 Git）。默认环境名与历史 YAML 中的
`flowpose` 不同，以避免修改原有运行环境；可通过 `--env-name flowpose` 显式更新旧环境。
YAML 面向主抓取应用，触觉 SDK 安装基础依赖；可选 TensorRT / CuPy 后端及其他实验、训练脚本的
依赖按对应子项目说明安装。

模型权重不在 Git 中：请将 `sam3.pt`、`FlowNet3.pth`、`ScaleNet3.pth`、
`dinov2_vits14_pretrain.pth` 及 DINOv2 源码目录放到 `perception/models/`，
或用应用参数指定已有路径。脚本会报告缺失模型，但软件检查通过不代表实机已就绪。
机器人控制工作空间、RealSense USB 权限、IP、手眼标定、夹爪标定及 HOME / PUT 等参数
仍需按 [部署指南](env.md) 准备。确认完成后再运行：

```bash
python flowpose_request_ik_tester.py
```

ROS 软件源配置采用 [ROS2 官方安装流程](https://github.com/ros2/ros2_documentation/blob/humble/source/Installation/_Apt-Repositories.rst)。

## 目录结构

```text
marvinGrasp/
├── grasp_core/
│   ├── tools/
│   │   └── flowpose_request_ik_app.py   主应用和状态机
│   ├── perception/
│   │   ├── realsense_sam3.py             RealSense和SAM3基础接口
│   │   ├── flowpose_pipeline.py          SAM3/FlowPose推理流程
│   │   └── perception_runtime.py         异步结果收集、坐标转换、目标排序入口
│   ├── planning/
│   │   ├── grasp/
│   │   │   ├── planner.py                唯一抓取规划入口
│   │   │   ├── target_order.py            DBSCAN聚类和抓取顺序
│   │   │   └── policies/
│   │   │       ├── box_symmetry.py       Box/Cuboid姿态策略
│   │   │       ├── long_object.py         长物体/长方体抓取策略
│   │   │       └── ribbon.py              Ribbon特殊成功判定
│   │   └── trajectory/
│   │       ├── planner.py                笛卡尔轨迹规划
│   │       ├── interpolation.py          轨迹插值
│   │       └── orientation.py            姿态插值
│   ├── execution/
│   │   ├── robot_skill_service.py        抓取、放置、HOME统一服务
│   │   ├── motion_executor.py            IK轨迹发布和执行
│   │   ├── config.py                     执行层配置
│   │   ├── drop_monitor.py               掉落检测
│   │   └── skills/
│   │       ├── grasp.py                  抓取动作执行和反馈处理
│   │       ├── place.py                  放置动作执行
│   │       └── home.py                   HOME和失败恢复
│   ├── core/
│   │   ├── math/                         位姿、四元数、坐标和物体轴线
│   │   ├── types/                        TargetObjectPose等数据结构
│   │   └── io/                           感知结果和目标加载
│   ├── communication/                    ROS2、IK和夹爪通信
│   ├── config/                           参数解析和配置归一化
│   └── resources/                        tool.yaml、机器人模型等资源
├── perception/
│   ├── sam3/                             SAM3源码
│   ├── flowpose/                         FlowPose源码
│   └── models/                           模型权重目录
├── daimon_gripper/                       夹爪SDK和夹爪信号接收器
├── eye2hand_calibration/                 手眼标定工具
├── tests/                                单元测试和流程测试
└── flowpose_request_ik_tester.py         启动脚本
```

## 主应用调用链

```text
flowpose_request_ik_tester.py
        ↓
flowpose_request_ik_app.py
        ↓
启动时左右机械臂回 HOME
        ↓
RealSense采集图像
        ↓
SAM3检测实例 mask
        ↓
FlowPose计算物体 6D pose 和尺寸
        ↓
转换到 base_link 坐标系
        ↓
target_order.py 过滤、DBSCAN聚类、目标排序
        ↓
UnifiedGraspPlanner生成抓取计划
        ↓
motion_executor发布IK轨迹
        ↓
夹爪执行 grip
        ↓
确认抓取、Place、失败恢复或重抓
```

## 感知和目标排序

`perception_runtime.py` 负责将 FlowPose 输出转换成 `TargetObjectPose`，并调用 `target_order.py`。

```text
FlowPose目标
    ↓
转换到 base_link
    ↓
按体积过滤
    ├── volume > 0.0005 m³：丢弃
    └── 其他目标：保留
    ↓
使用 XY 中心点执行 DBSCAN
    ↓
单物体聚类优先
    ↓
单物体按距离 base_link 原点最近排序
    ↓
多物体聚类内部选择距离聚类中心最远的物体
    ↓
剩余物体按距离机器人排序
    ↓
排序结果的 targets[0] 作为首个抓取目标
```

DBSCAN 默认参数：

```text
eps = 0.12 m
min_samples = 1
聚类坐标 = base_link 下的 XY
```

每个目标的名称使用统一实例编号，例如 `pen_0`、`pen_1`、`pen_2`。SAM3、FlowPose、`TargetObjectPose` 和 OBB 可视化应使用同一个实例名称，不能在中间重新编号。

## 抓取规划架构

所有抓取姿态和抓取轨迹规划统一由 `UnifiedGraspPlanner` 负责，包括：

- Box/Cuboid策略选择
- 长物体姿态计算
- 普通目标姿态计算
- `tool.yaml` 模板加载和展开
- pregrasp/grasp姿态生成
- TCP、旋转和 downward tilt 修正
- 抓取轨迹和夹爪触发点生成

`execution/skills/grasp.py` 中的 `GraspPlanner` 只保留执行层适配功能，主要用于读取机械臂当前起始姿态，并调用 `UnifiedGraspPlanner`。它不再维护另一套规划算法。

抓取执行链：

```text
RobotActionService.publish_grasp()
        ↓
GraspSkill.execute()
        ↓
execute_grasp()
        ↓
读取当前机械臂姿态
        ↓
UnifiedGraspPlanner
        ↓
模板轨迹或计算轨迹
        ↓
发布request_ik轨迹
        ↓
最终点触发夹爪
        ↓
处理夹爪结果
```

规划层不负责 ROS 发布和夹爪通信；执行层不重复实现抓取姿态算法。

## Box/Cuboid策略

Box 和 Cuboid 统一进入抓取规划流程。尺寸分类由 `core/math/object_axes.py` 集中处理：

```text
最大边 / 最小边 ≤ 1.2：cube
最大边 / 最小边 > 1.2：cuboid
```

`box_symmetry.py` 负责 Box/Cuboid 的等价旋转和夹爪方向选择；`long_object.py` 负责长轴方向、闭合轴和长物体专用抓取轨迹。

## Place、HOME和失败恢复

应用启动时先执行左右臂 HOME，成功后才进入任务循环。

抓取成功后的流程为：

```text
grasp confirmed
    ↓
Place移动
    ↓
夹爪release
    ↓
Place完成
    ↓
读取最新相机帧
    ↓
重新执行SAM3 + FlowPose
    ↓
重新排序
    ↓
下一次抓取
```

Place 期间不会提前使用旧感知结果；下一次抓取必须使用松夹爪后的最新检测结果。

抓取失败时由 `home.py` 和 `flowpose_request_ik_app.py` 协同处理：停止旧轨迹、释放夹爪、根据实测姿态恢复，再决定是否重新感知和重抓。

## 并发边界

```text
inference_executor  → SAM3、FlowPose、目标处理
action_executor     → 机械臂轨迹、抓取、Place、HOME
gripper_executor    → 夹爪TCP命令
```

同一个 IK 发布器不能被多个动作并发驱动，因此机械臂动作始终通过 `action_executor` 串行执行。感知可以异步运行，但只有在动作状态允许时才能触发抓取。

## 输出和调试

当前主应用默认不保存图像文件；SAM3 和 FlowPose 可视化主要保留在内存中用于界面显示。`target_order.py` 仍支持通过 `output_dir` 保存 OBB 图，外部调用时可以显式开启。

重要调试信息包括：

- 当前目标名称和 base_link 坐标
- 体积过滤结果
- 抓取模板是否命中
- 起始姿态来源
- pregrasp/grasp轨迹状态
- 夹爪位置、电流和失败原因
- Place和掉落恢复状态

## 扩展规则

- 新增感知模型：放入 `grasp_core/perception/`
- 新增物体排序或过滤规则：放入 `planning/grasp/target_order.py`
- 新增抓取姿态策略：放入 `planning/grasp/policies/`
- 新增模板处理：放入 `planning/grasp/planner.py`
- 新增轨迹算法：放入 `planning/trajectory/`
- 新增抓取、Place或HOME动作：放入 `execution/skills/`
- 新增 ROS2、IK或夹爪接口：放入 `communication/`
- 新增通用数学和数据结构：放入 `core/`

同一功能只保留一个真实实现。其他层通过公开接口调用，避免重复的姿态计算、配置处理和硬件控制逻辑。
