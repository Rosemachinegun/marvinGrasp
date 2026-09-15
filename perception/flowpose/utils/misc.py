import torch
import torch.nn.functional as F
import numpy as np

def average_quaternion_batch(Q, weights=None):
    """calculate the average quaternion of the multiple quaternions
    Args:
        Q (tensor): [B, num_quaternions, 4]
        weights (tensor, optional): [B, num_quaternions]. Defaults to None.

    Returns:
        oriented_q_avg: average quaternion, [B, 4]
    """
    
    if weights is None:
        weights = torch.ones((Q.shape[0], Q.shape[1]), device=Q.device) / Q.shape[1]
    A = torch.zeros((Q.shape[0], 4, 4), device=Q.device)
    weight_sum = torch.sum(weights, axis=-1)

    oriented_Q = ((Q[:, :, 0:1] > 0).float() - 0.5) * 2 * Q
    A = torch.einsum("abi,abk->abik", (oriented_Q, oriented_Q))
    A = torch.sum(torch.einsum("abij,ab->abij", (A, weights)), 1)
    A /= weight_sum.reshape(A.shape[0], -1).unsqueeze(-1).repeat(1, 4, 4)

    q_avg = torch.linalg.eigh(A)[1][:, :, -1]
    oriented_q_avg = ((q_avg[:, 0:1] > 0).float() - 0.5) * 2 * q_avg
    return oriented_q_avg

def get_pose_dim(rot_mode):
    assert rot_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix'], \
        f"the rotation mode {rot_mode} is not supported!"
        
    if rot_mode == 'quat_wxyz' or rot_mode == 'quat_xyzw':
        pose_dim = 7
    elif rot_mode == 'euler_xyz':
        pose_dim = 6
    elif rot_mode == 'euler_xyz_sx_cx' or rot_mode == 'rot_matrix':
        pose_dim = 9
    else:
        raise NotImplementedError
    return pose_dim

def encode_axes(axes: torch.Tensor, dim: int) -> torch.Tensor:
    ''' axes: Bx... '''
    bs = axes.shape[0]
    axes = axes.reshape(bs, -1, 1)
    embedding = []
    exponent = (2 ** torch.arange(dim, device=axes.device, dtype=torch.float32)).reshape(1, 1, -1)
    for fn in [torch.sin, torch.cos]:
        embedding.append(fn(exponent * axes).reshape(bs, -1))
    return torch.concat(embedding, dim=-1)

def rot6d_to_mat_batch(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix.
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks. CVPR 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # poses
    x_raw = d6[..., 0:3]  # bx3
    y_raw = d6[..., 3:6]  # bx3

    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)  # b*3
    z = np.cross(x, y_raw) # b*3
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)          # b*3
    y = np.cross(z, x)     # b*3                      

    return np.stack((x, y, z), axis=-1)  # (b,3,3)

def merge_results(results_ori, results_new):
    if len(results_ori.keys()) == 0:
        return results_new
    else:
        results = {
            'pred_pose': torch.cat([results_ori['pred_pose'], results_new['pred_pose']], dim=0),
            'gt_pose': torch.cat([results_ori['gt_pose'], results_new['gt_pose']], dim=0),
            'cls_id': torch.cat([results_ori['cls_id'], results_new['cls_id']], dim=0),
            'handle_visibility': torch.cat([results_ori['handle_visibility'], results_new['handle_visibility']], dim=0),
            # 'path': results_ori['path'] + results_new['path'],
        }
        return results