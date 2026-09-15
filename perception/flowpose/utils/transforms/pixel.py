from numpy import ndarray
from utils.transforms.metadata import CameraIntrinsicsBase
import numpy as np

def pixel2xyz(h: int, w: int, pixel: ndarray, intrinsics: CameraIntrinsicsBase):
    """
    Transform `(pixel[0], pixel[1])` to normalized 3D vector under cv space, using camera intrinsics.

    :param h: height of the actual image
    :param w: width of the actual image
    """

    # scale camera parameters
    scale_x = w / intrinsics.width
    scale_y = h / intrinsics.height
    fx = intrinsics.fx * scale_x
    fy = intrinsics.fy * scale_y
    x_offset = intrinsics.cx * scale_x
    y_offset = intrinsics.cy * scale_y

    x = (pixel[1] - x_offset) / fx
    y = (pixel[0] - y_offset) / fy
    vec = np.array([x, y, 1])
    return vec / np.linalg.norm(vec)