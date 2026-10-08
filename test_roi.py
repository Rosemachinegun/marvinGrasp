#!/usr/bin/env python3
"""
RealSense ROI calibration / testing tool.

功能：
1. 实时显示 RealSense RGB 图像
2. 鼠标拖动框选 ROI
3. 显示 ROI 像素范围和中心点
4. 读取 ROI 中心深度
5. 将 ROI 中心反投影到相机坐标系
6. 按 s 保存 ROI 配置
7. 按 r 清除 ROI
8. 按 q 退出
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


OUTPUT_FILE = "realsense_roi.json"

WIDTH = 640
HEIGHT = 480
FPS = 30


class ROISelector:
    def __init__(self):
        self.dragging = False

        self.start_x = 0
        self.start_y = 0

        self.end_x = 0
        self.end_y = 0

        self.roi = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True

            self.start_x = x
            self.start_y = y

            self.end_x = x
            self.end_y = y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.end_x = x
                self.end_y = y

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

            self.end_x = x
            self.end_y = y

            x1 = min(self.start_x, self.end_x)
            y1 = min(self.start_y, self.end_y)

            x2 = max(self.start_x, self.end_x)
            y2 = max(self.start_y, self.end_y)

            self.roi = (x1, y1, x2, y2)

            print("\nROI selected:")
            print(f"  x_min = {x1}")
            print(f"  y_min = {y1}")
            print(f"  x_max = {x2}")
            print(f"  y_max = {y2}")
            print(f"  width = {x2 - x1}")
            print(f"  height = {y2 - y1}")


def get_roi_depth(depth_frame, roi):
    """
    计算 ROI 内有效深度的中位数。

    中位数比单独使用中心像素更加稳定，
    可以减少深度孔洞和异常点的影响。
    """

    x1, y1, x2, y2 = roi

    depth_image = np.asanyarray(depth_frame.get_data())

    roi_depth = depth_image[y1:y2, x1:x2]

    if roi_depth.size == 0:
        return None

    valid_depth = roi_depth[roi_depth > 0]

    if len(valid_depth) == 0:
        return None

    depth_scale = depth_frame.get_units()

    median_depth_raw = np.median(valid_depth)

    return float(median_depth_raw * depth_scale)


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        WIDTH,
        HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    config.enable_stream(
        rs.stream.depth,
        WIDTH,
        HEIGHT,
        rs.format.z16,
        FPS,
    )

    print("Starting RealSense...")

    profile = pipeline.start(config)

    # --------------------------------------------------
    # Depth 对齐到 RGB
    # --------------------------------------------------

    align = rs.align(rs.stream.color)

    # --------------------------------------------------
    # 获取 RGB 相机内参
    # --------------------------------------------------

    color_profile = (
        profile
        .get_stream(rs.stream.color)
        .as_video_stream_profile()
    )

    intr = color_profile.get_intrinsics()

    print("\nCamera intrinsics")
    print("-----------------")
    print(f"width  : {intr.width}")
    print(f"height : {intr.height}")
    print(f"fx     : {intr.fx}")
    print(f"fy     : {intr.fy}")
    print(f"cx     : {intr.ppx}")
    print(f"cy     : {intr.ppy}")
    print(f"model  : {intr.model}")
    print(f"coeffs : {intr.coeffs}")

    selector = ROISelector()

    window_name = "RealSense ROI Calibration"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cv2.setMouseCallback(
        window_name,
        selector.mouse_callback,
    )

    print("\nControls")
    print("--------")
    print("Mouse drag : select ROI")
    print("s          : save ROI")
    print("r          : reset ROI")
    print("q / ESC    : quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()

            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(
                color_frame.get_data()
            )

            display = color_image.copy()

            # --------------------------------------------------
            # 正在拖动
            # --------------------------------------------------

            if selector.dragging:
                cv2.rectangle(
                    display,
                    (selector.start_x, selector.start_y),
                    (selector.end_x, selector.end_y),
                    (0, 255, 255),
                    2,
                )

            # --------------------------------------------------
            # 已确定 ROI
            # --------------------------------------------------

            if selector.roi is not None:

                x1, y1, x2, y2 = selector.roi

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.circle(
                    display,
                    (cx, cy),
                    5,
                    (0, 0, 255),
                    -1,
                )

                # ----------------------------------------------
                # ROI 深度中位数
                # ----------------------------------------------

                depth_m = get_roi_depth(
                    depth_frame,
                    selector.roi,
                )

                text1 = (
                    f"ROI: "
                    f"({x1},{y1}) "
                    f"-> "
                    f"({x2},{y2})"
                )

                cv2.putText(
                    display,
                    text1,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                if depth_m is not None:

                    # ------------------------------------------
                    # 2D -> 3D
                    # ------------------------------------------

                    point_3d = rs.rs2_deproject_pixel_to_point(
                        intr,
                        [cx, cy],
                        depth_m,
                    )

                    X, Y, Z = point_3d

                    text2 = (
                        f"Center: ({cx},{cy}) "
                        f"Depth: {depth_m:.3f} m"
                    )

                    text3 = (
                        f"Camera XYZ: "
                        f"{X:.3f}, "
                        f"{Y:.3f}, "
                        f"{Z:.3f} m"
                    )

                    cv2.putText(
                        display,
                        text2,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                    )

                    cv2.putText(
                        display,
                        text3,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 0),
                        2,
                    )

            cv2.imshow(
                window_name,
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            # --------------------------------------------------
            # 保存
            # --------------------------------------------------

            if key == ord("s"):

                if selector.roi is None:
                    print("ROI has not been selected.")
                    continue

                x1, y1, x2, y2 = selector.roi

                roi_config = {
                    "image": {
                        "width": WIDTH,
                        "height": HEIGHT,
                    },

                    "roi": {
                        "x_min": x1,
                        "y_min": y1,
                        "x_max": x2,
                        "y_max": y2,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "center_x": (x1 + x2) // 2,
                        "center_y": (y1 + y2) // 2,
                    },

                    "camera_intrinsics": {
                        "fx": intr.fx,
                        "fy": intr.fy,
                        "cx": intr.ppx,
                        "cy": intr.ppy,
                        "coeffs": list(intr.coeffs),
                    },
                }

                output_path = Path(OUTPUT_FILE)

                with output_path.open(
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        roi_config,
                        f,
                        indent=4,
                        ensure_ascii=False,
                    )

                print(
                    f"\nROI saved to: "
                    f"{output_path.resolve()}"
                )

            # --------------------------------------------------
            # Reset
            # --------------------------------------------------

            elif key == ord("r"):

                selector.roi = None

                print("\nROI reset.")

            # --------------------------------------------------
            # Quit
            # --------------------------------------------------

            elif key == ord("q") or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()