import kornia
import torch
from torch import Tensor

from ..utils import misc
from . import transforms as gtr
from .reconstruction import Pose


def _mean_isotropic_scale_normalize(
    points: torch.Tensor,
    eps: float = 1e-3,
    weights: torch.Tensor | None = None,
    return_pose: bool = False,
    f_eps: float = 0.0,
) -> tuple[torch.Tensor, Pose | torch.Tensor]:
    r"""Normalize points. Avoid inplace operations.

    Args:
       points : Tensor containing the points to be normalized with shape :math:`(B, N, D)`.
       eps : Small value to avoid division by zero error.
       weights : Optional tensor containing weights for each point with shape :math:`(B, N)`.

    Returns:
       Tuple containing the normalized points in the shape :math:`(B, N, D)` and the transformation matrix
       in the shape :math:`(B, D+1, D+1)`.

    """
    if weights is not None:
        x_mean = misc.wmean(points, weights[..., None], dim=1, keepdim=True)  # Bx1xD
        # clamp norm to prevent NaN gradient from norm(0)
        scale = misc.wmean(
            (points - x_mean).norm(dim=-1, p=2).clamp(min=f_eps), weights, dim=-1
        )  # B
    else:
        x_mean = torch.mean(points, dim=1, keepdim=True)  # Bx1xD
        scale = (points - x_mean).norm(dim=-1, p=2).clamp(min=f_eps).mean(dim=-1)  # B
    D_int = points.shape[-1]
    D_float = torch.tensor(points.shape[-1], dtype=torch.float64, device=points.device)
    scale = torch.sqrt(D_float) / (scale + eps)  # B
    scale = scale[:, None]  # B x 1

    norm_t_w = (
        torch.cat(
            [kornia.utils.eye_like(D_int, points), -x_mean.transpose(-1, -2)], dim=-1
        )
        * scale[..., None]
    )

    last_row = torch.cat(
        [torch.zeros_like(x_mean), torch.ones_like(x_mean[..., :1])], dim=-1
    )
    norm_T_w = torch.cat(
        [norm_t_w, last_row],
        dim=-2,
    )  # Bx(D+1)x(D+1)

    points_norm = kornia.geometry.linalg.transform_points(norm_T_w, points)  # BxNxD
    if return_pose:
        return (points_norm, Pose.from_projection_matrix(norm_T_w[:, :-1]))
    else:
        return (points_norm, norm_T_w)


@misc.AMP_CUSTOM_FWD_F32
def pnp_dlt(p3d_w: Tensor, p2d_c: Tensor, weights: Tensor | None = None) -> Pose:
    # p3d_w: (B, N, 3) - 3D points in world coordinates
    # p2d_c: (B, N, 2) - 2D points in camera coordinates (normalized)
    # weights: (B, N) - weights for each point correspondence
    B, N = p3d_w.shape[:2]

    p3d_w_norm, world_transform_norm = _mean_isotropic_scale_normalize(p3d_w)
    p3d_w_norm_h = gtr.to_homogeneous(p3d_w_norm)
    img_points_norm, img_transform_norm = _mean_isotropic_scale_normalize(p2d_c)
    inv_img_transform_norm = torch.inverse(img_transform_norm)

    # Setting up the system (the matrix A in Ax=0)
    _system = torch.zeros((B, 2 * N, 4), dtype=p3d_w.dtype, device=p3d_w.device)
    system = torch.cat(
        [
            _system.slice_scatter(p3d_w_norm_h, dim=1, start=0, step=2),
            _system.slice_scatter(p3d_w_norm_h, dim=1, start=1, step=2),
            _system.slice_scatter(
                p3d_w_norm_h * (-1) * img_points_norm[..., 0:1], dim=1, start=0, step=2
            ).slice_scatter(
                p3d_w_norm_h * (-1) * img_points_norm[..., 1:2], dim=1, start=1, step=2
            ),
        ],
        dim=-1,
    )

    if weights is not None:
        weights = weights.repeat_interleave(2, dim=1).sqrt()  # (B, 2N)
        system = system * weights[..., None]  # Apply weights to the system

    # Getting the solution vectors.
    _, _, v = torch.svd(system)
    solution = v[..., -1]

    # Reshaping the solution vectors to the correct shape.
    solution = solution.reshape(B, 3, 4)
    last_row = torch.cat(
        [
            torch.zeros_like(solution[:, :1, :-1]),
            torch.ones_like(solution[:, :1, -1:]),
        ],
        dim=-1,
    )
    # Creating solution_4x4
    solution_4x4 = torch.cat([solution, last_row], dim=-2)

    # De-normalizing the solution
    intermediate = torch.bmm(solution_4x4, world_transform_norm)
    solution = torch.bmm(inv_img_transform_norm, intermediate[:, :3, :])

    # We obtained one solution for each element of the batch. We may
    # need to multiply each solution with a scalar. This is because
    # if x is a solution to Ax=0, then cx is also a solution. We can
    # find the required scalars by using the properties of
    # rotation matrices. We do this in two parts:

    # First, we fix the sign by making sure that the determinant of
    # the all the rotation matrices are non negative (since determinant
    # of a rotation matrix should be 1).
    det = torch.det(solution[:, :3, :3])
    ones = torch.ones_like(det)
    sign_fix = torch.where(det < 0, ones * -1, ones)
    solution = solution * sign_fix[:, None, None]

    # Then, we make sure that norm of the 0th columns of the rotation
    # matrices are 1. Do note that the norm of any column of a rotation
    # matrix should be 1. Here we use the 0th column to calculate norm_col.
    # We then multiply solution with mul_factor.
    norm_col = torch.norm(input=solution[:, :3, 0], p=2, dim=1)
    mul_factor = (1 / norm_col)[:, None, None]
    temp = solution * mul_factor

    # To make sure that the rotation matrix would be orthogonal, we apply
    # QR decomposition.
    ortho, right = torch.linalg.qr(temp[:, :3, :3])

    # We may need to fix the signs of the columns of the ortho matrix.
    # If right[i, j, j] is negative, then we need to flip the signs of
    # the column ortho[i, :, j]. The below code performs the necessary
    # operations in an better way.
    mask = kornia.utils.eye_like(3, ortho)
    col_sign_fix = torch.sign(mask * right)
    rot_mat = torch.bmm(ortho, col_sign_fix)

    return Pose.from_Rt(rot_mat, temp[:, :3, 3])
