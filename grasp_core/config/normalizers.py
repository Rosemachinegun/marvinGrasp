"""Public configuration normalization entry points."""

def normalize_gripper_args(args):
    from grasp_core.config.request_ik_config import normalize_gripper_args as _normalize
    return _normalize(args)

def normalize_roi_xyxy(values):
    from grasp_core.config.request_ik_config import normalize_roi_xyxy as _normalize
    return _normalize(values)
