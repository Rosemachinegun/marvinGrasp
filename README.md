# Project Structure

仓库当前采用按职责分层的结构。运行入口位于 `grasp_core/tools/`，动作编排、技能执行和硬件通信分别由不同模块负责。

```text
marvinGrasp/
│
├── perception/
│   ├── flowpose/                 FlowPose 模型源码与推理脚本
│   ├── sam3/                     SAM3 源码（独立仓库指针）
│   └── models/                   本地模型权重；仅追踪 .gitkeep
│
├── grasp_core/
│   ├── perception/               相机、SAM3、FlowPose 运行时封装
│   │   ├── realsense_sam3.py
│   │   ├── perception_runtime.py
│   │   └── flowpose_pipeline.py
│   │
│   ├── core/                     通用数学、数据类型和文件读写
│   │   ├── math/                 位姿、旋转、向量、物体轴线
│   │   ├── types/                相机、目标和末端目标位姿
│   │   └── io/                   目标位姿加载
│   │
│   ├── config/                   运行配置和参数解析
│   │   ├── request_ik_config.py  CLI 兼容入口
│   │   ├── defaults.py / normalizers.py
│   │   ├── gripper_config.py
│   │   └── trajectory_config.py
│   │
│   ├── planning/
│   │   ├── grasp/                抓取姿态策略和模板
│   │   └── trajectory/           笛卡尔轨迹、插值和时间参数
│   │
│   ├── execution/                机器人动作执行层
│   │   ├── config.py             执行层统一配置
│   │   ├── motion_executor.py    轨迹下发与执行循环
│   │   ├── robot_skill_service.py 技能服务和动作接口
│   │   ├── drop_monitor.py       掉落检测
│   │   └── skills/
│   │       ├── grasp.py          抓取规划与执行
│   │       ├── place.py          放置规划与执行
│   │       └── home.py           Home 和恢复动作
│   │
│   ├── communication/            ROS2、IK、夹爪和反馈通信
│   │   ├── request_ik_transport.py
│   │   ├── request_ik_feedback.py
│   │   └── gripper_signal.py
│   │
│   ├── resources/                仓库内默认资源
│   │   ├── tool.yaml
│   │   └── stand_v3.urf.xacro
│   │
│   └── tools/                    应用入口、UI 和轨迹诊断
│       └── flowpose_request_ik_app.py
│
├── daimon_gripper/
│   ├── dm_gripper_py/            夹爪 SDK（独立仓库指针）
│   ├── dm_gripper_cam_py/        相机夹爪接口（独立仓库指针）
│   ├── dm_gripper_tac_py/        触觉夹爪接口（独立仓库指针）
│   └── grip_signal_*.py          夹爪信号服务和设备封装
│
├── eye2hand_calibration/         手眼标定工具
├── tests/                        单元测试和执行流程测试
│
├── flowpose_request_ik_tester.py 主程序启动脚本
│
└── README.md
```
