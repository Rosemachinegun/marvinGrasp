import torch
import math
import torch.nn.functional as F
from typing import Literal
from dataclasses import dataclass
import copy

def _normalize(q):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1, keepdim=True)
    return q.div(torch.clamp(norm, min=1e-9))

def _generate_random_quaternion(quaternion_shape):
    assert quaternion_shape[-1] == 4
    rand_norm = torch.randn(quaternion_shape)
    rand_q = _normalize(rand_norm)
    return rand_q

def _jitter_quaternion(q, theta):  #[Bs, 4], [Bs, 1]
    new_q = _generate_random_quaternion(q.shape).to(q.device)
    dot_product = torch.sum(q*new_q, dim=-1, keepdim=True)  #
    shape = (tuple(1 for _ in range(len(dot_product.shape) - 1)) + (4, ))
    q_orthogonal = _normalize(new_q - q * dot_product.repeat(*shape))
    # theta = 2arccos(|p.dot(q)|)
    # |p.dot(q)| = cos(theta/2)
    tile_theta = theta.repeat(shape)
    jittered_q = q*torch.cos(tile_theta/2) + q_orthogonal*torch.sin(tile_theta/2)

    return jittered_q

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def _noisy_rot_matrix(matrix, rad, type='normal'):
    if type == 'normal':
        theta = torch.abs(torch.randn_like(matrix[..., 0, 0])) * rad
    elif type == 'uniform':
        theta = torch.rand_like(matrix[..., 0, 0]) * rad
    quater = matrix_to_quaternion(matrix)
    new_quater = _jitter_quaternion(quater, theta.unsqueeze(-1))
    new_mat = quaternion_to_matrix(new_quater)
    return new_mat

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def eulerZYX_to_matrix(euler_zyx: torch.Tensor) -> torch.Tensor:
    """
    Convert ZYX Euler angles to rotation matrices explicitly.

    Builds R = Rz @ Ry @ Rx, where angles are given in [Z, Y, X] order.

    Args:
        euler_zyx: tensor of shape (..., 3) where
                   [..., 0] = yaw   (rotation around Z)
                   [..., 1] = pitch (rotation around Y)
                   [..., 2] = roll  (rotation around X)
                   Units: radians.

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    z = euler_zyx[..., 0]
    y = euler_zyx[..., 1]
    x = euler_zyx[..., 2]

    cz, sz = torch.cos(z), torch.sin(z)
    cy, sy = torch.cos(y), torch.sin(y)
    cx, sx = torch.cos(x), torch.sin(x)

    # Rz
    O = torch.zeros_like(cz)
    I = torch.ones_like(cz)
    Rz = torch.stack([
        torch.stack([ cz, -sz,  O], dim=-1),
        torch.stack([ sz,  cz,  O], dim=-1),
        torch.stack([  O,   O,  I], dim=-1),
    ], dim=-2)  # (..., 3, 3)

    # Ry
    Ry = torch.stack([
        torch.stack([ cy,  O, sy], dim=-1),
        torch.stack([  O,  I,  O], dim=-1),
        torch.stack([-sy,  O, cy], dim=-1),
    ], dim=-2)

    # Rx
    Rx = torch.stack([
        torch.stack([ I,  O,  O], dim=-1),
        torch.stack([ O, cx, -sx], dim=-1),
        torch.stack([ O, sx,  cx], dim=-1),
    ], dim=-2)

    return torch.matmul(torch.matmul(Rz, Ry), Rx)


def get_rot_matrix(batch_pose, pose_mode):
    """
    pose_mode: 
        'quat_wxyz'  -> batch_pose [B, 4]
        'quat_xyzw'  -> batch_pose [B, 4] 
        'euler_xyz'  -> batch_pose [B, 3] 
        'rot_matrix' -> batch_pose [B, 6]
        
    Return: rot_matrix [B, 3, 3]
    """
    assert pose_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix'],\
        f"the rotation mode {pose_mode} is not supported!"
        
    if pose_mode in ['quat_wxyz', 'quat_xyzw']:
        if pose_mode == 'quat_wxyz':
            quat_wxyz = batch_pose
        else:
            index = [3, 0, 1, 2]
            quat_wxyz = batch_pose[:, index]
        rot_mat = quaternion_to_matrix(quat_wxyz)
            
    elif pose_mode == 'rot_matrix':
        rot_mat= rotation_6d_to_matrix(batch_pose).permute(0, 2, 1)
        
    elif pose_mode == 'euler_xyz_sx_cx':
        rot_sin_theta = batch_pose[:, :3]
        rot_cos_theta = batch_pose[:, 3:6]
        theta = torch.atan2(rot_sin_theta, rot_cos_theta)
        rot_mat = euler_angles_to_matrix(theta, 'ZYX')
    elif pose_mode == 'euler_xyz':
        rot_mat = euler_angles_to_matrix(batch_pose, 'ZYX')
    else:
        raise NotImplementedError
    
    return rot_mat

def normalize_rotation(rotation, rotation_mode):
    if rotation_mode == 'quat_wxyz' or rotation_mode == 'quat_xyzw':
        rotation /= torch.norm(rotation, dim=-1, keepdim=True)
    elif rotation_mode == 'rot_matrix':
        rot_matrix = get_rot_matrix(rotation, rotation_mode)
        rotation[:, :3] = rot_matrix[:, :, 0]
        rotation[:, 3:6] = rot_matrix[:, :, 1]
    elif rotation_mode == 'euler_xyz_sx_cx':
        rot_sin_theta = rotation[:, :3]
        rot_cos_theta = rotation[:, 3:6]
        theta = torch.atan2(rot_sin_theta, rot_cos_theta)
        rotation[:, :3] = torch.sin(theta)
        rotation[:, 3:6] = torch.cos(theta)
    elif rotation_mode == 'euler_xyz':
        pass
    else:
        raise NotImplementedError
    return rotation

def add_noise_to_R(RT, type='normal', r=5.0, t=0.03):
    rand_type = type  # 'uniform' or 'normal' --> we use 'normal'
    # new_RT: torch.Tensor = copy.deepcopy(RT)
    new_RT = RT.clone()
    new_RT[:, :3, :3] = _noisy_rot_matrix(RT[:, :3, :3], r/180*math.pi, type=rand_type).reshape(RT[:, :3, :3].shape)
    assert not torch.any(torch.isnan(new_RT)) and not torch.any(torch.isinf(new_RT))

    return new_RT

def get_pose_representation(batch_rot_mat, pose_mode):
    """
    pose_mode: 
        'quat_wxyz'  -> [B, 4]
        'quat_xyzw'  -> [B, 4] 
        'euler_xyz'  -> [B, 3] 
        'rot_matrix' -> [B, 6]
    """
    assert pose_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix'],\
        f"the rotation mode {pose_mode} is not supported!"
    
    if pose_mode == 'quat_xyzw':
        rot = matrix_to_quaternion(batch_rot_mat)
    elif pose_mode == 'quat_wxyz':
        rot = matrix_to_quaternion(batch_rot_mat)[:, [3, 0, 1, 2]]
    elif pose_mode == 'euler_xyz':
        rot = matrix_to_euler_angles(batch_rot_mat, 'ZYX')
    elif pose_mode == 'euler_xyz_sx_cx':
        rot = matrix_to_euler_angles(batch_rot_mat, 'ZYX')
        rot_sin_theta = torch.sin(rot)
        rot_cos_theta = torch.cos(rot)
        rot = torch.cat((rot_sin_theta, rot_cos_theta), dim=-1)
    elif pose_mode == 'rot_matrix':
        rot = matrix_to_rotation_6d(batch_rot_mat.permute(0, 2, 1)).reshape(batch_rot_mat.shape[0], -1)
    else:
        raise NotImplementedError
    
    return rot

@dataclass
class SymLabel:
    """
    Symmetry labels for real-world objects.

    Axis rotation details:

    - `any`: arbitrary rotation around this axis is ok
    - `half`: rotate 180 degrees along this axis (central symmetry)
    - `quarter`: rotate 90 degrees along this axis (like square)

    .. doctest::

        >>> from cutoop.rotation import SymLabel
        >>> sym = SymLabel(any=False, x='any', y='none', z='none')
        >>> str(sym)
        'x-cone'
        >>> sym = SymLabel(any=False, x='any', y='any', z='none') # two any produces 'any'
        >>> str(sym)
        'any'
        >>> sym = SymLabel(any=False, x='half', y='half', z='half')
        >>> str(sym)
        'box'

    """

    any: bool  # sphere
    """Whether arbitrary rotation is allowed"""
    x: "Literal['none', 'any', 'half', 'quarter']"
    """axis rotation for x"""
    y: "Literal['none', 'any', 'half', 'quarter']"
    """axis rotation for y"""
    z: "Literal['none', 'any', 'half', 'quarter']"
    """axis rotation for z"""

    def get_only(self, tag: 'Literal["any", "half", "quarter"]'):
        """Get the only axis marked with the tag. If multiple or none is find, return ``None``."""
        ret: 'list[Literal["x", "y", "z"]]' = []
        if self.x == tag:
            ret.append("x")
        if self.y == tag:
            ret.append("y")
        if self.z == tag:
            ret.append("z")
        if len(ret) != 1:
            return None
        return ret[0]

    @staticmethod
    def from_str(s: str) -> "SymLabel":
        """Construct symmetry from string.

        .. note:: See also :obj:`STANDARD_SYMMETRY`.
        """
        if s in STANDARD_SYMMETRY:
            return copy.deepcopy(STANDARD_SYMMETRY[s])
        else:
            raise ValueError(f"invalid symmetry: {s}")

    def __str__(self) -> str:
        """
        For human readability, rotations are divided into the following types (names):

        - ``any``: arbitrary rotation is ok;
        - ``cube``: the same symmetry as a cube;
        - ``box``: the same symmetry as a box (flipping along x, y, and z axis);
        - ``none``: no symmetry is provided;
        - ``{x,y,z}-flip``: flip along a single axis;
        - ``{x,y,z}-square-tube``: the same symmetry as a square tube alone the axis;
        - ``{x,y,z}-square-pyramid``: the same symmetry as a pyramid alone the axis;
        - ``{x,y,z}-cylinder``: the same symmetry as a cylinder the axis;
        - ``{x,y,z}-cone``: the same symmetry as a cone the axis.

        """
        c_any = (self.x == "any") + (self.y == "any") + (self.z == "any")
        c_quarter = (
            (self.x == "quarter") + (self.y == "quarter") + (self.z == "quarter")
        )
        c_half = (self.x == "half") + (self.y == "half") + (self.z == "half")

        if self.any or c_any > 1 or (c_any > 0 and c_quarter > 0):  # any rotation is ok
            return "any"

        if c_any == 0:
            if c_quarter > 1:
                return "cube"  # cube group
            elif c_quarter == 0:
                if c_half > 1:
                    return "box"  # cube_flip_group
                else:  # one half or none
                    axis = self.get_only("half")
                    return f"{axis}-flip" if axis is not None else "none"
            else:  # c_quarter == 1
                axis = self.get_only("quarter")
                if c_half > 0:
                    return f"{axis}-square-tube"
                else:
                    return f"{axis}-square-pyramid"
        else:
            assert c_any == 1 and c_quarter == 0
            axis = self.get_only("any")
            if c_half > 0:
                return f"{axis}-cylinder"
            else:
                return f"{axis}-cone"


STANDARD_SYMMETRY = {
    "any": SymLabel(any=True, x="any", y="any", z="any"),
    "cube": SymLabel(any=False, x="quarter", y="quarter", z="quarter"),
    "box": SymLabel(any=False, x="half", y="half", z="half"),
    "none": SymLabel(any=False, x="none", y="none", z="none"),
    "x-flip": SymLabel(any=False, x="half", y="none", z="none"),
    "y-flip": SymLabel(any=False, x="none", y="half", z="none"),
    "z-flip": SymLabel(any=False, x="none", y="none", z="half"),
    "x-square-tube": SymLabel(any=False, x="quarter", y="half", z="half"),
    "y-square-tube": SymLabel(any=False, x="half", y="quarter", z="half"),
    "z-square-tube": SymLabel(any=False, x="half", y="half", z="quarter"),
    "x-square-pyramid": SymLabel(any=False, x="quarter", y="none", z="none"),
    "y-square-pyramid": SymLabel(any=False, x="none", y="quarter", z="none"),
    "z-square-pyramid": SymLabel(any=False, x="none", y="none", z="quarter"),
    "x-cylinder": SymLabel(any=False, x="any", y="half", z="half"),
    "y-cylinder": SymLabel(any=False, x="half", y="any", z="half"),
    "z-cylinder": SymLabel(any=False, x="half", y="half", z="any"),
    "x-cone": SymLabel(any=False, x="any", y="none", z="none"),
    "y-cone": SymLabel(any=False, x="none", y="any", z="none"),
    "z-cone": SymLabel(any=False, x="none", y="none", z="any"),
}