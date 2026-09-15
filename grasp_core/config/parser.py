"""Public command-line parser entry point."""

def parse_args():
    from grasp_core.config.request_ik_config import parse_args as _parse_args
    return _parse_args()
