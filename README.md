# everygrasp 目录说明

`grasp_core/` 是当前机器人抓取项目的主体代码包。一级目录只保留顶层包文件和说明文档，所有业务代码都放在二级功能目录中。
![System Architecture](frame.png)

## 功能目录

| 目录 | 职责 |
|---|---|
| `apps/` | 应用入口、主循环、按键事件、资源生命周期 |
| `perception/` | RealSense、SAM3、FlowPose、感知结果收集 |
| `core/` | 位姿数据结构、坐标转换、四元数和几何工具 |
| `config/` | 命令行参数、默认值、YAML 配置读取 |
| `planning/` | 抓取目标位姿规划、tool.yaml waypoint 模板展开 |
| `motion/` | 轨迹插值和轨迹步长计算 |
| `communication/` | ROS2、IK target、夹爪 socket 等底层通信 |
| `tasks/` | 抓取、回 home、夹爪动作、失败恢复等任务执行 |
| `ui/` | OpenCV dashboard 和状态显示 |

## 入口链路

```text
flowpose_request_ik_tester.py
  -> grasp_core/apps/flowpose_request_ik_app.py
  -> grasp_core/tasks/robot_actions.py
  -> grasp_core/tasks/grasp_request_ik.py
  -> grasp_core/planning/ + grasp_core/motion/ + grasp_core/communication/
```

## 扩展规则

- 新增抓取、放置、分类任务：放到 `tasks/`
- 新增抓取姿态策略或模板解析：放到 `planning/`
- 新增轨迹插值、速度曲线、路径采样：放到 `motion/`
- 新增相机、分割模型、位姿估计模型：放到 `perception/`
- 新增 ROS topic、夹爪、机械臂通信：放到 `communication/`
- 新增共享位姿结构和坐标数学：放到 `core/`

每个通用能力只保留一个入口，其他模块只能调用，不再复制实现。

## 平板控制台

运行主入口后会同时启动浅蓝色平板 Web 控制台，服务默认监听
`0.0.0.0:7860`。平板和机器人处于同一局域网时，访问：

```text
http://机器人IP:7860/
```

例如机器人 IP 为 `192.168.10.123` 时，地址为
`http://192.168.10.123:7860/`。页面持续显示 RealSense 原始彩色画面，
“识别与定位”复用热键 Z 的 SAM3 + FlowPose 流程，“识别并执行抓取”复用
热键 A 的 SAM3 + FlowPose + 机械臂执行流程；Home 和停止也沿用主循环现有接口。

可用 TRUE/FALSE 参数控制是否启动平板服务，并可覆盖监听地址和端口：

```bash
python flowpose_request_ik_tester.py --tablet-ui TRUE
python flowpose_request_ik_tester.py --tablet-ui FALSE
python flowpose_request_ik_tester.py --tablet-ui TRUE --tablet-ui-port 7861
```

页面刷新间隔默认 0.25 秒，可通过环境变量
`TASK_LOOP_UI_REFRESH_SEC` 调整。

## 夹爪部分

- `daimon_stuff/dm_gripper_cam_py/` 腕部相机相关功能包
- `daimon_stuff/dm_gripper_tac_py/` 触觉传感器相关功能包；`daimon_stuff/tac.py` 是触觉入口。4个传感器启动规则如下
    |1
    |python daimon_stuff/tac.py --remote-addr 192.168.10.11:50052 --dev-id 2 --pc-host 192.168.10.123 --pc-port 60031
    |2
    |python daimon_stuff/tac.py --remote-addr 192.168.10.11:50051 --dev-id 0 --pc-host 192.168.10.123 --pc-port 60030
    |3
    |python daimon_stuff/tac.py --remote-addr 192.168.10.10:50052 --dev-id 2 --pc-host 192.168.10.123 --pc-port 60033
    |4
    |python daimon_stuff/tac.py --remote-addr 192.168.10.10:50051 --dev-id 0 --pc-host 192.168.10.123 --pc-port 60032
- `gripper/` 不再放 Python 文件，Daimon 相关入口统一放在 `daimon_stuff/`

直接按 L：双夹爪一起闭合。
先按 J 再按 L：只闭合左夹爪。
先按 H 再按 L：只闭合右夹爪。
# DaimonGeneral
