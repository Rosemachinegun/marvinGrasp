# Magic 目录说明

`grasp_core/` 是当前机器人抓取项目的主体代码包。一级目录只保留顶层包文件和说明文档，所有业务代码都放在二级功能目录中。

## 功能目录

| 目录 | 职责 |
|---|---|
| `tools/` | 应用入口、操作界面、轨迹记录和运行诊断 |
| `ui/` | 平板 Web UI 及其线程安全命令/画面桥接 |
| `apps/` | 语音等可选应用输入适配器 |
| `perception/` | RealSense、SAM3、FlowPose、感知结果收集 |
| `core/` | 位姿数据结构、坐标转换、四元数和几何工具 |
| `config/` | 命令行参数、默认值、YAML 配置读取 |
| `planning/grasp/` | 抓取目标位姿规划、resources/tool.yaml waypoint 模板展开 |
| `execution/` | 抓取、放置、回 home、夹爪动作、失败恢复等动作执行 |
| `planning/trajectory/` | 轨迹插值和轨迹步长计算 |
| `communication/` | ROS2、IK target、夹爪 socket 等底层通信 |

## 入口链路

```text
flowpose_request_ik_tester.py
  -> grasp_core/tools/flowpose_request_ik_app.py
  -> grasp_core/ui/tablet_ui.py / grasp_core/apps/voice_grasp.py
  -> grasp_core/execution/robot_skill_service.py
  -> grasp_core/execution/skills/grasp.py
  -> grasp_core/execution/motion_executor.py
  -> grasp_core/communication/request_ik_transport.py
```

## 扩展规则

- 新增抓取、放置、分类任务：放到 `execution/`
- 新增抓取姿态策略或模板解析：放到 `planning/grasp/`
- 新增轨迹插值、速度曲线、路径采样：放到 `planning/trajectory/`
- 新增相机、分割模型、位姿估计模型：放到 `perception/`
- 新增 ROS topic、夹爪、机械臂通信：放到 `communication/`
- 新增浏览器界面：放到 `ui/`；新增可选应用输入：放到 `apps/`
- 新增共享位姿结构和坐标数学：放到 `core/`

每个通用能力只保留一个入口，其他模块只能调用，不再复制实现。
