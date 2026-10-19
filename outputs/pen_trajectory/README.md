# pen 抓取轨迹学术汇报图

输出：`pen_grasp_trajectory.png`（4800 × 2700，300 dpi）、`pen_grasp_trajectory.svg`（矢量）、`pen_trajectory_samples.csv`（全部位置及四元数样本）、`metadata.json`。

重现命令：在仓库根目录运行 `python outputs/pen_trajectory/generate_figure.py`。脚本只调用离线规划模块，不连接或控制机器人。

## 假设与边界

- 用户给定位置作为机器人基坐标系下的规划输入，单位 m：物体中心 (0.385, -0.154, 0.670)。
- 缺少实测起始位姿，假设右臂从 `HomeConfig` 的 (0.250, -0.250, 0.830) 出发，姿态取配置 Home 姿态。
- 假设 pen 水平放置，长轴沿基坐标系 +X；采用示意尺寸 (0.140, 0.012, 0.012) m 触发长物体策略。尺寸不是测量值。当前 pen 模板横向偏移为零，位置航点不随水平偏航变化；姿态和采样数可能变化。
- `GraspConfig.controlled_args()` 覆盖抓取参数，当前强制物体高度为 0.670 m，不能仅按 tool.yaml 的旧默认值 0.650 m 推导。
- 负 Y 对应自动选择右臂。pen 模板预抓取与抓取偏移分别为世界 +Z 方向 0.100 和 0.030 m。
- 当前抓取函数只运动至抓取点并闭合夹爪，不含后续抬升、放置或回 Home；图未添加这些动作。
- 图中是末端目标轨迹，不是关节轨迹、实测轨迹或通过 IK/碰撞检验的运动。
- 横轴为归一化采样序号，不代表实测时间。发布循环中存在固定 sleep，故没有根据标称 75 Hz 将横轴宣称为真实执行时间。

## 代码依据

- `grasp_core/execution/config.py`：Home 起点、抓取参数及其覆盖规则。
- `grasp_core/resources/tool.yaml`：pen 抓取模板。
- `grasp_core/planning/grasp/planner.py`：模板展开和连续接近路径；此抓取路径 lift_arc=False。
- `grasp_core/planning/trajectory/planner.py`、`interpolation.py`：分段三次 Bézier、两轮采样链平滑、四元数 SLERP。
- `grasp_core/planning/trajectory/constraints.py`：有效步长限制。
- `grasp_core/execution/skills/grasp.py`：与执行一致的 min_steps=1、final_slowdown_ratio=0.03，默认 smoothing_passes=2。

本例共 353 个样本，路径长度约 0.2473 m。已校验起点、预抓取点和抓取终点均与规划结果一致。平滑后的下降段有轻微 X/Y 超调，图保留原始规划结果。

建议图注：在物体中心为 (0.385, -0.154, 0.670) m、右臂由配置 Home 位姿出发的假设下，采用当前抓取模板及分段三次 Bézier 插值生成末端目标轨迹。左图展示三维路径，右图展示坐标分量随归一化采样进度的变化；预抓取点与抓取点分别位于物体中心上方 0.10 m 和 0.03 m。结果为离线规划示例，非实测轨迹。
