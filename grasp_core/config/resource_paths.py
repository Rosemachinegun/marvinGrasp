"""Paths to external grasp_core resources."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_ROOT = PROJECT_ROOT / "grasp_core" / "resources"
DEFAULT_TOOL_TEMPLATE_PATH = RESOURCES_ROOT / "tool.yaml"
DEFAULT_ROBOT_XACRO_PATH = RESOURCES_ROOT / "stand_v3.urf.xacro"
DEFAULT_SAM3_ROI_XYXY = (144,220,560,430)
DEFAULT_TRAJECTORY_PLOT_DIR = PROJECT_ROOT / "captures" / "request_ik_trajectories"
DEFAULT_JOINT_TRAJECTORY_CSV_DIR = PROJECT_ROOT / "captures" / "request_ik_joint_trajectories"
