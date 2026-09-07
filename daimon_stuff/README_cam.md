# Gripper Camera 目录说明 + how to grip

`daimon_stuff/dual_camera_viewer.py` 是双目远程相机查看器入口。

## 目录职责

| 目录 | 职责 |
|---|---|
| `daimon_stuff/dm_gripper_cam_py/viewer_config.py` | 双目查看器参数 |
| `daimon_stuff/dm_gripper_cam_py/worker.py` | 后台读取 worker |
| `daimon_stuff/dm_gripper_cam_py/dashboard.py` | dashboard 拼接 |
| `daimon_stuff/dm_gripper_cam_py/remote_camera.py` | 远程相机客户端，负责 gRPC 控制流和视频流读取 |
| `daimon_stuff/dm_gripper_cam_py/udp_frame.py` | UDP 视频帧分片、解析和重组协议 |
| `daimon_stuff/dm_gripper_cam_py/hevc_ffmpeg_decoder.py` | HEVC ffmpeg 解码 |
| `daimon_stuff/dm_gripper_cam_py/camera_proxy*` | camera proxy 的 proto 文件和生成的 gRPC Python 文件 |

## 执行链路

```text
daimon_stuff/dual_camera_viewer.py
  -> daimon_stuff/dm_gripper_cam_py/viewer_config.py
  -> daimon_stuff/dm_gripper_cam_py/worker.py
  -> daimon_stuff/dm_gripper_cam_py/remote_camera.py
  -> daimon_stuff/dm_gripper_cam_py/camera_proxy* + udp_frame.py + hevc_ffmpeg_decoder.py
  -> daimon_stuff/dm_gripper_cam_py/dashboard.py
```

主入口只负责启动和显示循环，具体功能放在 `daimon_stuff` 中，避免重复实现。# 
#  how to grip

以下参数用于调节夹爪闭合行为，包括速度、夹持力度、接触检测和防撞策略。

| Parameter | Description | Tuning Guide |
|---|---|---|
| `--speed 60` | 闭合速度 | 当前卡顿问题与该参数无关。软物体建议 `30-50`；追求快速抓取可使用 `60` |
| `--torque 30` | 最大夹取力矩 | 夹取过紧时降低到 `20` 或 `15` |
| `--hold-torque 10` | 接触后的保持力 | 默认保持 `10` 即可，避免持续加大压力 |
| `--min-pos 300` | 最小闭合位置限制 | 防止夹爪过度闭合。目前建议保持 `300`；夹不到物体时可尝试 `250` |
| `--max-pos 900` | 初始张开位置 | 定义抓取前夹爪展开程度 |
| `--current-threshold 120` | 电流停止阈值 | 当前电流约为 `6`，主要依靠位置停滞检测判断接触，该参数保持默认即可 |
| `--poll-interval 0.05` | 状态检测周期（秒） | 越小响应越快，推荐范围 `0.03-0.08` |
| `--contact-grace 0.4` | 接触检测延迟时间 | 忽略启动阶段无运动状态，避免误判。速度较慢时可增加到 `0.6` |
| `--progress-epsilon 2` | 最小运动距离阈值 | 判断夹爪是否仍在运动。误停止时降低到 `1`；停止过晚时增加到 `3-5` |
| `--stall-samples 5` | 停滞检测次数 | 连续多次无明显运动后认为夹取完成。响应慢改为 `3`；误停止改为 `6-8` |
