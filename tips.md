/home/kewei/922/marvinGrasp/grasp_core/core/math/pose.py line.451

def select_ik_hand(position_xyz: np.ndarray, hand_mode: str) -> str:
    if hand_mode in {"left", "right"}:
        return hand_mode
    return "left" if float(position_xyz[1]) > 0.1 else "right"


/home/kewei/922/marvinGrasp/grasp_core/execution/config.py line.44

target_trajectory_speed_mps: float = 0.15
