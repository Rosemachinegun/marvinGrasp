import numpy as np

def get_bbox(bbox, img_height = 480, img_length = 640):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, img_height - 40, img_length - 40)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_height:
        delt = rmax - img_height
        rmax = img_height
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def aug_bbox_eval(bbox_xyxy, im_H, im_W):
    """Used for DZI, the augmented box is a square (maybe enlarged)
    Args:
        bbox_xyxy (np.ndarray):
    Returns:
        center, scale
    """
    x1, y1, x2, y2 = bbox_xyxy.copy()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1
    bbox_center = np.array([cx, cy])  # (w/2, h/2)
    scale = max(y2 - y1, x2 - x1)
    scale = min(scale, max(im_H, im_W)) * 1.0
    return bbox_center, scale