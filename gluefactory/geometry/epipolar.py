import torch

from .utils import skew_symmetric, to_homogeneous
from .wrappers import Camera, Pose


def T_to_E(T: Pose):
    """Convert batched poses (..., 4, 4) to batched essential matrices."""
    return skew_symmetric(T.t) @ T.R


def T_to_F(cam0: Camera, cam1: Camera, T_0to1: Pose):
    return E_to_F(cam0, cam1, T_to_E(T_0to1))


def E_to_F(cam0: Camera, cam1: Camera, E: torch.Tensor):
    assert cam0._data.shape[-1] == 6, "only pinhole cameras supported"
    assert cam1._data.shape[-1] == 6, "only pinhole cameras supported"
    K0 = cam0.calibration_matrix()
    K1 = cam1.calibration_matrix()
    return K1.inverse().transpose(-1, -2) @ E @ K0.inverse()


def F_to_E(cam0: Camera, cam1: Camera, F: torch.Tensor):
    assert cam0._data.shape[-1] == 6, "only pinhole cameras supported"
    assert cam1._data.shape[-1] == 6, "only pinhole cameras supported"
    K0 = cam0.calibration_matrix()
    K1 = cam1.calibration_matrix()
    return K1.transpose(-1, -2) @ F @ K0


def sym_epipolar_distance(p0, p1, E, squared=True):
    """Compute batched symmetric epipolar distances.
    Args:
        p0, p1: batched tensors of N 2D points of size (..., N, 2).
        E: essential matrices from camera 0 to camera 1, size (..., 3, 3).
    Returns:
        The symmetric epipolar distance of each point-pair: (..., N).
    """
    assert p0.shape[-2] == p1.shape[-2]
    if p0.shape[-2] == 0:
        return torch.zeros(p0.shape[:-1]).to(p0)
    if p0.shape[-1] != 3:
        p0 = to_homogeneous(p0)
    if p1.shape[-1] != 3:
        p1 = to_homogeneous(p1)
    p1_E_p0 = torch.einsum("...ni,...ij,...nj->...n", p1, E, p0)
    E_p0 = torch.einsum("...ij,...nj->...ni", E, p0)
    Et_p1 = torch.einsum("...ij,...ni->...nj", E, p1)
    d0 = (E_p0[..., 0] ** 2 + E_p0[..., 1] ** 2).clamp(min=1e-6)
    d1 = (Et_p1[..., 0] ** 2 + Et_p1[..., 1] ** 2).clamp(min=1e-6)
    if squared:
        d = p1_E_p0**2 * (1 / d0 + 1 / d1)
    else:
        d = p1_E_p0.abs() * (1 / d0.sqrt() + 1 / d1.sqrt()) / 2
    return d


def sym_epipolar_distance_all(p0, p1, E, eps=1e-15):
    if p0.shape[-1] != 3:
        p0 = to_homogeneous(p0)
    if p1.shape[-1] != 3:
        p1 = to_homogeneous(p1)
    p1_E_p0 = torch.einsum("...mi,...ij,...nj->...nm", p1, E, p0).abs()
    E_p0 = torch.einsum("...ij,...nj->...ni", E, p0)
    Et_p1 = torch.einsum("...ij,...mi->...mj", E, p1)
    d0 = p1_E_p0 / (E_p0[..., None, 0] ** 2 + E_p0[..., None, 1] ** 2 + eps).sqrt()
    d1 = (
        p1_E_p0
        / (Et_p1[..., None, :, 0] ** 2 + Et_p1[..., None, :, 1] ** 2 + eps).sqrt()
    )
    return (d0 + d1) / 2


def generalized_epi_dist(
    kpts0, kpts1, cam0: Camera, cam1: Camera, T_0to1: Pose, all=True, essential=True
):
    if essential:
        E = T_to_E(T_0to1)
        p0 = cam0.image2cam(kpts0)
        p1 = cam1.image2cam(kpts1)
        if all:
            return sym_epipolar_distance_all(p0, p1, E, agg="max")
        else:
            return sym_epipolar_distance(p0, p1, E, squared=False)
    else:
        assert cam0._data.shape[-1] == 6
        assert cam1._data.shape[-1] == 6
        K0, K1 = cam0.calibration_matrix(), cam1.calibration_matrix()
        F = K1.inverse().transpose(-1, -2) @ T_to_E(T_0to1) @ K0.inverse()
        if all:
            return sym_epipolar_distance_all(kpts0, kpts1, F)
        else:
            return sym_epipolar_distance(kpts0, kpts1, F, squared=False)


def decompose_essential_matrix(E):
    # decompose matrix by its singular values
    U, _, V = torch.svd(E)
    Vt = V.transpose(-2, -1)

    mask = torch.ones_like(E)
    mask[..., -1:] *= -1.0  # fill last column with negative values

    maskt = mask.transpose(-2, -1)

    # avoid singularities
    U = torch.where((torch.det(U) < 0.0)[..., None, None], U * mask, U)
    Vt = torch.where((torch.det(Vt) < 0.0)[..., None, None], Vt * maskt, Vt)

    W = skew_symmetric(E.new_tensor([[0, 0, 1]]))
    W[..., 2, 2] += 1.0

    # reconstruct rotations and retrieve translation vector
    U_W_Vt = U @ W @ Vt
    U_Wt_Vt = U @ W.transpose(-2, -1) @ Vt

    # return values
    R1 = U_W_Vt
    R2 = U_Wt_Vt
    T = U[..., -1]
    return R1, R2, T


# pose errors
# TODO: test for batched data
def angle_error_mat(R1, R2):
    cos = (torch.trace(torch.einsum("...ij, ...jk -> ...ik", R1.T, R2)) - 1) / 2
    cos = torch.clip(cos, -1.0, 1.0)  # numerical errors can make it out of bounds
    return torch.rad2deg(torch.abs(torch.arccos(cos)))


def angle_error_vec(v1, v2, eps=1e-10):
    n = torch.clip(v1.norm(dim=-1) * v2.norm(dim=-1), min=eps)
    v1v2 = (v1 * v2).sum(dim=-1)  # dot product in the last dimension
    return torch.rad2deg(torch.arccos(torch.clip(v1v2 / n, -1.0, 1.0)))


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0, eps=1e-10):
    if isinstance(T_0to1, torch.Tensor):
        R_gt, t_gt = T_0to1[:3, :3], T_0to1[:3, 3]
    else:
        R_gt, t_gt = T_0to1.R, T_0to1.t
    R_gt, t_gt = torch.squeeze(R_gt), torch.squeeze(t_gt)

    # angle error between 2 vectors
    t_err = angle_error_vec(t, t_gt, eps)
    t_err = torch.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if t_gt.norm() < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    r_err = angle_error_mat(R, R_gt)

    return t_err, r_err
