# MarvinGrasp

> **Perception-driven robotic grasping pipeline for real-world dual-arm manipulation.**  
> From RGB-D perception and 6D object pose estimation to grasp policy, motion generation, gripper control, and failure recovery.

`MarvinGrasp` 是一套面向真实机器人抓取任务的端到端实验框架。

项目围绕一个核心问题展开：

> **机器人如何从“看见桌面上的物体”，最终得到一个稳定、自然并且可执行的抓取动作？**

系统将视觉感知、6D Pose、抓取姿态策略、机械臂运动规划、夹爪控制以及失败恢复连接为一条完整链路，并针对真实机器人中经常出现的 **姿态对称、多解 IK、轨迹不连续、抓取失败、物体掉落以及未知物体泛化** 等问题进行了工程化处理。

---

## Overview

```text
RGB-D Camera
     │
     ▼
Object Detection / Segmentation
     │
     │  SAM3
     ▼
Object Mask + Point Cloud
     │
     ▼
FlowPose
     │
     │  6D Pose
     ▼
Pose Normalization
     │
     │  Cube symmetry
     │  Long-object axis policy
     │  Arm-side constraints
     ▼
Grasp Target Generation
     │
     ▼
IK / Reachability
     │
     ▼
Smooth Cartesian Trajectory
     │
     ▼
Robot Arm
     │
     ▼
Daimon Gripper
     │
     ├── Grasp
     ├── Object detection
     ├── Drop detection
     └── Failure recovery
```
![System Architecture](frame.png)
整个系统不是简单地将视觉模型输出直接发送给 IK。

在 **Perception → Robot Execution** 之间增加了一层抓取策略与姿态规范化，使视觉模型给出的“几何上正确姿态”进一步转换成“机械臂真正适合执行的姿态”。

---

# System Pipeline

## 1. RGB-D Perception

系统首先从 RealSense RGB-D 相机获取：

- RGB image
- Depth image
- Camera intrinsics
- Point cloud

感知模块主要位于：

```text
grasp_core/perception/
```

当前视觉链路以目标分割和 6D Pose estimation 为核心。

```text
RGB
 │
 ├── Object Prompt
 │
 ▼
SAM3
 │
 ▼
Object Mask
 │
 ├──────── Depth
 │           │
 ▼           ▼
Masked RGB-D / Point Cloud
             │
             ▼
          FlowPose
```

---

## 2. Object Segmentation

SAM3 用于从场景中提取目标物体 mask。

Mask 的主要作用并不只是得到物体轮廓，而是：

> **从复杂 RGB-D 场景中隔离属于目标物体的几何信息。**

这样能够避免桌面、背景以及邻近物体的点云进入后续 Pose estimation。

当前系统支持针对已知类别配置 prompt，同时正在面向：

- unknown objects
- cluttered scenes
- adjacent objects
- stacked objects

扩展更通用的目标发现策略。

---

## 3. FlowPose 6D Pose Estimation

FlowPose 负责从目标物体的视觉与几何信息中预测物体的：

```text
Translation
x, y, z

Rotation
R ∈ SO(3)
```

最终得到：

```text
T_camera_object ∈ SE(3)
```

再通过相机外参转换至机器人基坐标系：

```text
T_base_object
=
T_base_camera
×
T_camera_object
```

从而完成：

```text
Image Object
        ↓
Camera-frame 3D Geometry
        ↓
6D Object Pose
        ↓
base_link Object Pose
```

`FlowPose/` 保存 FlowPose 相关代码与模型接口。

> 模型权重及大型训练数据不包含在 Git 仓库中，需要单独配置。

---

# Pose Is Not Yet a Grasp

6D Pose 正确，并不意味着机器人一定能够自然抓取。

这是 MarvinGrasp 中非常重要的一层设计。

例如一个立方体旋转：

```text
0°
90°
180°
270°
```

对于人来说可能是完全等价的。

但是对于机器人，它们可能对应完全不同的：

- wrist configuration
- IK solution
- elbow posture
- reachability
- collision risk

因此系统不会直接：

```text
FlowPose → IK
```

而是：

```text
FlowPose
   ↓
Pose Policy
   ↓
Canonical Grasp Pose
   ↓
IK
```

---

# Grasp Pose Policy

抓取姿态策略主要位于：

```text
grasp_core/planning/
grasp_core/policyhere/
grasp_core/tasks/
```

不同几何类型使用不同的 normalization policy。

---

## Cube / Symmetric Objects

对于具有旋转对称性的物体，会生成多个等价姿态候选：

```text
R
R · Rz(90°)
R · Rz(180°)
R · Rz(270°)
```

随后根据机械臂侧、抓取方向以及可执行性选择更合理的候选。

系统希望满足：

```text
Left arm  → grasp from left / outside toward inside
Right arm → grasp from right / outside toward inside
```

而不是单纯选择视觉模型输出的第一个姿态。

---

## Long Objects

对于：

- pen
- screwdriver handle
- elongated tools
- stick-like objects

抓取问题更加依赖物体的物理长轴。

系统根据预测尺寸动态寻找：

```python
long_axis_index = argmax(size)
```

而不是假设 FlowPose 的固定 local X/Y/Z 一定对应物体长轴。

随后构造统一的 Policy Frame：

```text
Policy Y = physical long axis
Policy Z = world +Z
Policy X = Y × Z
```

并保证：

```text
X × Y = Z
det(R) = +1
```

这样即使 FlowPose 在不同帧中将长轴编码到不同 local axis，抓取策略依然能够保持一致。

对于细长物体，夹爪 closing direction 会进一步约束在物体短轴方向附近，以避免沿长轴错误闭合。

---

# Arm-Side Constraints

双臂系统不仅需要目标末端位姿，还需要限制机械臂采用合理的身体构型。

当前抓取策略倾向于满足：

### Left Arm

```text
workspace: y > 0
wrist: left side
elbow: left side
approach: outside → inside
```

### Right Arm

```text
workspace: y < 0
wrist: right side
elbow: right side
approach: outside → inside
```

核心目标是减少：

- wrist folding
- arm crossing
- center-line crossing
- large joint jumps
- unnatural IK branches

---

# Grasp Target Generation

最终抓取目标不是一个孤立 pose，而是一系列具有语义的阶段：

```text
Object Pose
    │
    ▼
Pre-Grasp
    │
    ▼
Approach
    │
    ▼
Grasp
    │
    ▼
Retreat
```

对象级 waypoint 和抓取参数主要通过：

```text
config/tool.yaml
```

管理。

配置可以描述：

- relative position offset
- relative orientation
- grasp height
- approach distance
- retreat distance
- object-specific policy

从而避免将大量抓取参数硬编码在任务代码中。

---

# Motion Generation

运动相关实现位于：

```text
grasp_core/motion/
```

MarvinGrasp 不希望机械臂简单地：

```text
Waypoint A
    ↓
Waypoint B
    ↓
Waypoint C
```

形成明显折线。

轨迹层负责将任务级 waypoint 转换成连续采样轨迹，并考虑：

- Cartesian position interpolation
- orientation interpolation
- local path density
- speed continuity
- smooth start / stop
- waypoint transition continuity

目标是从：

```text
robotic-looking segmented motion
```

逐渐转向：

```text
continuous human-readable motion
```

特别是在：

- pick → lift
- grasp → put
- pause → home
- failure → home

这些长距离运动阶段中减少突变。

---

# IK and Robot Execution

通信层位于：

```text
grasp_core/communication/
```

负责：

- ROS2 target publishing
- request IK
- Cartesian trajectory publishing
- gripper command communication

主要执行链路：

```text
flowpose_request_ik_tester.py
        │
        ▼
grasp_core/apps/
flowpose_request_ik_app.py
        │
        ▼
grasp_core/tasks/
robot_actions.py
        │
        ▼
grasp_core/tasks/
grasp_request_ik.py
        │
        ├── planning/
        ├── motion/
        └── communication/
```

---

# Daimon Gripper

`daimon_stuff/` 包含 Daimon 末端设备相关功能。

```text
daimon_stuff/
│
├── dm_gripper_py/
│   └── gripper SDK / position control
│
├── dm_gripper_cam_py/
│   └── wrist camera / remote video stream
│
├── dm_gripper_tac_py/
│   └── tactile / force / slip detection
│
├── dual_camera_viewer.py
│
├── grip_signal_receiver.py
│
└── tac.py
```

系统不只是发送：

```text
close gripper
```

而是希望通过夹爪反馈判断：

```text
Did we actually grasp an object?
```

---

# Grasp Failure Detection

真实机器人中，“夹爪执行了闭合”不等于“抓取成功”。

系统通过夹爪位置、状态以及反馈信息区分：

```text
Gripper Closing
      │
      ▼
Reached empty-grip limit?
      │
 ┌────┴────┐
Yes        No
 │          │
 ▼          ▼
Empty     Object
Grasp     Detected
```

因此抓取流程能够从单向执行：

```text
Perception → Grasp → Done
```

升级为闭环：

```text
Perception
   ↓
Grasp
   ↓
Feedback
   ↓
Success?
 ┌─┴─┐
Yes  No
 │    │
Put  Recover / Retry
```

---

# Drop Detection & Recovery

物体在抓取后仍可能因为：

- insufficient gripping force
- soft object deformation
- unstable contact
- motion acceleration

发生掉落。

系统包含运行期间的抓取状态监控：

```text
Carrying Object
      │
      ▼
Gripper Feedback
      │
      ▼
Object Still Present?
   ┌──┴──┐
  Yes    No
   │      │
Continue  Stop Motion
          │
          ▼
       Recovery
          │
          ▼
         Home
```

恢复流程特别关注：

- 立即停止当前任务
- 避免旧轨迹继续执行
- 从真实当前位置重新开始
- 避免 IK branch jump
- 平滑返回 Home

---

# Project Structure

```text
marvinGrasp/
│
├── FlowPose/
│   └── 6D pose estimation
│
├── config/
│   ├── tool.yaml
│   └── robot / camera configuration
│
├── grasp_core/
│   │
│   ├── apps/
│   │   └── application entry / main loop
│   │
│   ├── perception/
│   │   └── RealSense / SAM3 / FlowPose
│   │
│   ├── core/
│   │   └── pose / transform / quaternion utilities
│   │
│   ├── planning/
│   │   └── grasp target and waypoint planning
│   │
│   ├── policyhere/
│   │   └── object-specific pose policies
│   │
│   ├── motion/
│   │   └── trajectory interpolation
│   │
│   ├── communication/
│   │   └── ROS2 / IK / gripper communication
│   │
│   ├── tasks/
│   │   └── grasp / put / home / recovery
│   │
│   └── ui/
│       └── runtime dashboard
│
├── daimon_stuff/
│   ├── dm_gripper_py/
│   ├── dm_gripper_cam_py/
│   └── dm_gripper_tac_py/
│
├── tests/
│
├── flowpose_request_ik_tester.py
│
└── README.md
```

---

# Environment

Primary development environment:

```text
Ubuntu 22.04
ROS2 Humble
Python 3.10
CUDA-capable NVIDIA GPU
Intel RealSense RGB-D Camera
Daimon Gripper
```

Recommended Python environment:

```bash
conda activate flowpose
```

Exact model dependencies may depend on the local FlowPose / SAM3 deployment.

---

# Quick Start

## 1. Clone

```bash
git clone https://github.com/Rosemachinegun/marvinGrasp.git
cd marvinGrasp
```

Checkout the development branch when needed:

```bash
git checkout failloop
```

---

## 2. Check Configuration

Main robot grasp configuration:

```text
config/tool.yaml
```

Before running on a real robot, verify:

- camera extrinsic
- robot network
- left / right arm configuration
- gripper IP
- object policy
- home pose
- grasp / put coordinates

---

## 3. Daimon Gripper Dependencies

```bash
cd daimon_stuff
python -m pip install -r requirement.txt
```

Install the gripper SDK:

```bash
cd dm_gripper_py
python -m pip install .
```

Optional tactile module:

```bash
cd ../dm_gripper_tac_py
python -m pip install .
```

---

## 4. Run Main Grasp Pipeline

From the repository root:

```bash
python flowpose_request_ik_tester.py
```

The runtime application coordinates:

```text
Camera
→ Segmentation
→ FlowPose
→ Pose Policy
→ Grasp Planning
→ IK
→ Trajectory
→ Gripper
→ Recovery
```

---

# Development Philosophy

MarvinGrasp separates the system into several layers:

```text
Perception
    ↓
Geometry
    ↓
Policy
    ↓
Planning
    ↓
Motion
    ↓
Execution
    ↓
Feedback
```

A model should answer:

> **Where is the object?**

A grasp policy should answer:

> **How should this robot grasp it?**

A motion planner should answer:

> **How should the robot move there?**

Feedback should answer:

> **Did the physical action actually succeed?**

Keeping these problems separate makes the system easier to debug and extend.

---

# Extension Guide

### Add a new perception model

```text
grasp_core/perception/
```

### Add a new grasp / put / recovery task

```text
grasp_core/tasks/
```

### Add an object-specific grasp strategy

```text
grasp_core/planning/
or
grasp_core/policyhere/
```

### Add trajectory algorithms

```text
grasp_core/motion/
```

### Add ROS / device communication

```text
grasp_core/communication/
```

### Add geometry / pose utilities

```text
grasp_core/core/
```

The general rule is:

> **Each reusable capability should have one canonical implementation and one clear interface.**

Avoid duplicating pose conversion, trajectory generation or device communication logic across task modules.

---

# Current Focus

Current engineering work focuses on improving robustness in real-world manipulation:

- [x] RGB-D perception pipeline
- [x] SAM-based object segmentation
- [x] FlowPose 6D pose estimation
- [x] Camera → `base_link` pose transformation
- [x] Object-specific grasp pose normalization
- [x] Cube symmetry handling
- [x] Long-object axis normalization
- [x] Dual-arm side-aware grasp policy
- [x] Smooth Cartesian trajectory generation
- [x] Daimon gripper integration
- [x] Empty-grasp detection
- [x] Object-drop detection
- [x] Failure recovery
- [ ] Unknown-object automatic discovery
- [ ] Clutter / stacked-object segmentation
- [ ] Online grasp quality scoring
- [ ] Collision-aware approach planning
- [ ] Closed-loop visual re-planning
- [ ] Wrist-camera fine alignment
- [ ] More general category-free grasping

---

# Research Direction

The long-term goal of MarvinGrasp is not to build a collection of hard-coded pick scripts.

The goal is to gradually move from:

```text
Object-specific handcrafted grasp
```

toward:

```text
Open-world perception
        +
Geometry-aware grasp policy
        +
Continuous robot motion
        +
Closed-loop physical feedback
```

so that the system can handle objects that were not explicitly programmed beforehand while still producing stable and physically meaningful robot behavior.

---

## Notes

This repository is primarily intended for robotics research and real-robot experimentation.

Large model weights, datasets, calibration files and machine-specific runtime assets may not be included in the repository and should be configured separately for each deployment environment.
