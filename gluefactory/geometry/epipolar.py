import torch

from . import reconstruction
from . import transforms as tr


def T_to_E(T: reconstruction.Pose):
    """Convert batched poses (..., 4, 4) to batched essential matrices."""
    return tr.skew_symmetric(T.t) @ T.R


def T_to_F(
    cam0: reconstruction.Camera,
    cam1: reconstruction.Camera,
    T_0to1: reconstruction.Pose,
):
    return E_to_F(cam0, cam1, T_to_E(T_0to1))


def E_to_F(cam0: reconstruction.Camera, cam1: reconstruction.Camera, E: torch.Tensor):
    assert cam0.dist.shape[-1] == 0, "only pinhole cameras supported"
    assert cam1.dist.shape[-1] == 0, "only pinhole cameras supported"
    K0 = cam0.calibration_matrix()
    K1 = cam1.calibration_matrix()
    return K1.inverse().transpose(-1, -2) @ E @ K0.inverse()


def F_to_E(cam0: reconstruction.Camera, cam1: reconstruction.Camera, F: torch.Tensor):
    assert cam0.dist.shape[-1] == 0, "only pinhole cameras supported"
    assert cam1.dist.shape[-1] == 0, "only pinhole cameras supported"
    K0 = cam0.calibration_matrix()
    K1 = cam1.calibration_matrix()
    return K1.transpose(-1, -2) @ F @ K0


def sym_epipolar_distance(p0, p1, E, squared=True, symmetric=True):
    """Compute batched epipolar distances.
    Args:
        p0, p1: batched tensors of N 2D points of size (..., N, 2).
        E: essential matrices from camera 0 to camera 1, size (..., 3, 3).
        squared: if True, return squared distances.
        symmetric: if True, average distance from both sides. If False,
            compute only the distance of p1 to the epipolar line of p0.
    Returns:
        The epipolar distance of each point-pair: (..., N).
    """
    assert p0.shape[-2] == p1.shape[-2]
    if p0.shape[-2] == 0:
        return torch.zeros(p0.shape[:-1]).to(p0)
    if p0.shape[-1] != 3:
        p0 = tr.to_homogeneous(p0)
    if p1.shape[-1] != 3:
        p1 = tr.to_homogeneous(p1)
    p1_E_p0 = torch.einsum("...ni,...ij,...nj->...n", p1, E, p0)
    E_p0 = torch.einsum("...ij,...nj->...ni", E, p0)
    d0 = (E_p0[..., 0] ** 2 + E_p0[..., 1] ** 2).clamp(min=1e-6)
    if symmetric:
        Et_p1 = torch.einsum("...ij,...ni->...nj", E, p1)
        d1 = (Et_p1[..., 0] ** 2 + Et_p1[..., 1] ** 2).clamp(min=1e-6)
        if squared:
            d = p1_E_p0**2 * (1 / d0 + 1 / d1)
        else:
            d = p1_E_p0.abs() * (1 / d0.sqrt() + 1 / d1.sqrt()) / 2
    else:
        if squared:
            d = p1_E_p0**2 / d0
        else:
            d = p1_E_p0.abs() / d0.sqrt()
    return d


def sym_epipolar_distance_all(p0, p1, E, eps=1e-15, symmetric=True):
    if p0.shape[-1] != 3:
        p0 = tr.to_homogeneous(p0)
    if p1.shape[-1] != 3:
        p1 = tr.to_homogeneous(p1)
    p1_E_p0 = torch.einsum("...mi,...ij,...nj->...nm", p1, E, p0).abs()
    E_p0 = torch.einsum("...ij,...nj->...ni", E, p0)
    d0 = p1_E_p0 / (E_p0[..., None, 0] ** 2 + E_p0[..., None, 1] ** 2 + eps).sqrt()
    if symmetric:
        Et_p1 = torch.einsum("...ij,...mi->...mj", E, p1)
        d1 = (
            p1_E_p0
            / (Et_p1[..., None, :, 0] ** 2 + Et_p1[..., None, :, 1] ** 2 + eps).sqrt()
        )
        return (d0 + d1) / 2
    return d0


def bearing_epipolar_distance(
    bearings0: torch.Tensor,
    bearings1: torch.Tensor,
    T_0to1: reconstruction.Pose,
    squared: bool = False,
    symmetric: bool = True,
) -> torch.Tensor:
    """Epipolar distance between bearing directions (depth-independent).

    Unlike generalized_epi_dist which operates on 2D keypoints projected
    through the full camera model (including translation), this function
    works directly with bearing directions in each camera frame. The bearings
    should be computed using rotation only (no translation), making the
    distance independent of point depth.

    Operates in normalized camera coordinates with the essential matrix.

    Args:
        bearings0: bearing directions in camera 0 frame, (..., N, 3).
        bearings1: bearing directions in camera 1 frame, (..., N, 3).
            Typically computed as R_wahba @ pred_xyz (rotation only,
            no translation), then normalized to z=1.
        T_0to1: GT relative pose for the essential matrix.
        squared: if True, return squared distances.
        symmetric: if True, average distance from both sides. If False,
            compute distance of bearings1 to the epipolar line of bearings0.

    Returns:
        Epipolar distance per point pair: (..., N).
    """
    E = T_to_E(T_0to1)
    # Normalize to z=1 so the distance is independent of point depth.
    b0 = bearings0 / bearings0[..., 2:].clamp(min=1e-6)
    b1 = bearings1 / bearings1[..., 2:].clamp(min=1e-6)
    return sym_epipolar_distance(b0, b1, E, squared=squared, symmetric=symmetric)


def generalized_epi_dist(
    kpts0,
    kpts1,
    cam0: reconstruction.Camera,
    cam1: reconstruction.Camera,
    T_0to1: reconstruction.Pose,
    all=True,
    essential=True,
    symmetric=True,
):
    if essential:
        E = T_to_E(T_0to1)
        p0 = cam0.image2cam(kpts0)
        p1 = cam1.image2cam(kpts1)
        if all:
            return sym_epipolar_distance_all(p0, p1, E, symmetric=symmetric)
        else:
            return sym_epipolar_distance(p0, p1, E, squared=False, symmetric=symmetric)
    else:
        # assert cam0.data_.shape[-1] == 6
        # assert cam1.data_.shape[-1] == 6
        F = T_to_F(cam0, cam1, T_0to1)
        if all:
            return sym_epipolar_distance_all(kpts0, kpts1, F, symmetric=symmetric)
        else:
            return sym_epipolar_distance(
                kpts0, kpts1, F, squared=False, symmetric=symmetric
            )


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

    W = tr.skew_symmetric(E.new_tensor([[0, 0, 1]]))
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


def check_epipolar_intersection(
    x_i0: torch.Tensor,
    i1_F_i0: torch.Tensor,
    width1: int | torch.Tensor,
    height1: int | torch.Tensor,
) -> torch.BoolTensor:
    x_i0 = tr.to_homogeneous(x_i0)  # (..., 3)
    L_B = x_i0 @ i1_F_i0.T

    l1, l2, l3 = L_B.split(1, dim=1)

    eps = 1e-6

    # Vertical boundary checks (x=0, x=W_B)
    mask_v = torch.abs(l2) > eps
    l2_inv = torch.where(mask_v, 1.0 / l2, torch.tensor(0.0, device=x_i0.device))

    y0 = -l3 * l2_inv
    yW = -(l1 * width1 + l3) * l2_inv

    check_v = mask_v & (((y0 >= 0) & (y0 <= height1)) | ((yW >= 0) & (yW <= height1)))

    # Horizontal boundary checks (y=0, y=H_B)
    mask_h = torch.abs(l1) > eps
    l1_inv = torch.where(mask_h, 1.0 / l1, torch.tensor(0.0, device=x_i0.device))

    x0 = -l3 * l1_inv
    xH = -(l2 * height1 + l3) * l1_inv

    check_h = mask_h & (((x0 >= 0) & (x0 <= width1)) | ((xH >= 0) & (xH <= width1)))

    return (check_v | check_h).squeeze(-1)


def rays_to_plucker(c_t_w: reconstruction.Pose, rays_cam: torch.Tensor) -> torch.Tensor:
    """Convert camera rays to Plücker coordinates in world frame.

    Args:
        c_t_w: Pose in camera-from-world convention, shape (...).
        rays_cam: Ray directions in camera coordinates, shape (..., N, 3).

    Returns:
        Plücker coordinates (direction, moment) in world frame, shape (..., N, 6).
    """
    # Get world-from-camera transform
    w_t_c = c_t_w.inv()

    # Camera center in world coordinates
    cam_center = w_t_c.t  # (..., 3)

    # Transform ray directions to world frame
    rays_world = (w_t_c.R @ rays_cam.unsqueeze(-1)).squeeze(-1)  # (..., N, 3)

    # Compute moment: m = origin × direction
    # Broadcast camera center to match rays shape
    moment = torch.cross(
        cam_center.unsqueeze(-2).expand_as(rays_world),
        rays_world,
        dim=-1,
    )  # (..., N, 3)

    # Stack into Plücker coordinates: (direction, moment)
    plucker = torch.cat([rays_world, moment], dim=-1)  # (..., N, 6)
    return plucker


def plucker_epipolar_distance_all(
    plucker0: torch.Tensor, plucker1: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """All-pairs symmetric epipolar distance from Plücker rays.

    Uses the reciprocal product of two Plücker lines, which equals the
    scalar triple product [(o0-o1), d0, d1] — the coplanarity residual.
    Zero iff the two rays intersect (epipolar constraint satisfied).

    Generalizes sym_epipolar_distance_all to work directly from 6D rays
    without cameras, poses, or the essential matrix. Handles non-central
    cameras (per-ray origins) for free.

    Args:
        plucker0: Plücker coords (direction, moment) for view 0, (B, N0, 6).
        plucker1: Plücker coords (direction, moment) for view 1, (B, N1, 6).
        eps: numerical stability constant.

    Returns:
        All-pairs distance, shape (B, N0, N1).
    """
    d0, m0 = plucker0[..., :3], plucker0[..., 3:]
    d1, m1 = plucker1[..., :3], plucker1[..., 3:]

    # Reciprocal product: d0·m1 + m0·d1 = [(o0-o1), d0, d1]
    rp = torch.einsum("bni,bmi->bnm", d0, m1) + torch.einsum(
        "bni,bmi->bnm", m0, d1
    )

    # Normalize by direction norms (absorbed by learned scale, but keeps
    # the quantity well-conditioned for unnormalized rays)
    norm0 = d0.norm(dim=-1).clamp(min=eps)  # (B, N0)
    norm1 = d1.norm(dim=-1).clamp(min=eps)  # (B, N1)
    return rp.abs() / (norm0[:, :, None] * norm1[:, None, :])


def xyz_epipolar_distance_all(
    origin0: torch.Tensor,
    origin1: torch.Tensor,
    xyz0: torch.Tensor,
    xyz1: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """All-pairs epipolar distance from camera origins and 3D coordinates.

    Efficient specialization of plucker_epipolar_distance_all for pinhole
    cameras where all rays per view share a single origin. Computes the
    scalar triple product [(o1-o0), d0, d1] via one cross and one einsum.

    Args:
        origin0: Camera center for view 0, (B, 3).
        origin1: Camera center for view 1, (B, 3).
        xyz0: 3D coordinates of keypoints in view 0, (B, N0, 3).
        xyz1: 3D coordinates of keypoints in view 1, (B, N1, 3).
        eps: numerical stability constant.

    Returns:
        All-pairs distance, shape (B, N0, N1).
    """
    d0 = xyz0 - origin0[:, None, :]  # (B, N0, 3)
    d1 = xyz1 - origin1[:, None, :]  # (B, N1, 3)
    b = origin1 - origin0  # (B, 3)

    # [b, d0, d1] = d1 · (b × d0)  for all pairs
    b_cross_d0 = torch.linalg.cross(
        b[:, None].expand_as(d0), d0
    )  # (B, N0, 3)
    triple = torch.einsum("bni,bmi->bnm", b_cross_d0, d1)  # (B, N0, N1)

    norm0 = d0.norm(dim=-1).clamp(min=eps)  # (B, N0)
    norm1 = d1.norm(dim=-1).clamp(min=eps)  # (B, N1)
    return triple.abs() / (norm0[:, :, None] * norm1[:, None, :])


def triangulate_from_plucker(
    plucker1: torch.Tensor, plucker2: torch.Tensor
) -> torch.Tensor:
    """Triangulate 3D point from two Plücker rays.

    Finds the midpoint of the closest points on each ray.

    Args:
        plucker1: Plücker coordinates (direction, moment), shape (..., 6).
        plucker2: Plücker coordinates (direction, moment), shape (..., 6).

    Returns:
        3D point (midpoint of closest points), shape (..., 3).
    """
    d1, m1 = plucker1[..., :3], plucker1[..., 3:]
    d2, m2 = plucker2[..., :3], plucker2[..., 3:]

    # Recover point on each line: p = (d × m) / ||d||²
    d1_sq = (d1 * d1).sum(-1, keepdim=True)
    d2_sq = (d2 * d2).sum(-1, keepdim=True)
    o1 = torch.cross(d1, m1, dim=-1) / d1_sq
    o2 = torch.cross(d2, m2, dim=-1) / d2_sq

    # Find closest points between two lines
    w = o1 - o2
    a = (d1 * d1).sum(-1)
    b = (d1 * d2).sum(-1)
    c = (d2 * d2).sum(-1)
    d = (d1 * w).sum(-1)
    e = (d2 * w).sum(-1)

    denom = a * c - b * b + 1e-8  # avoid division by zero for parallel rays
    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom

    # Closest points on each ray
    p1 = o1 + t1[..., None] * d1
    p2 = o2 + t2[..., None] * d2

    # Return midpoint
    # You can also return t1, t2 if you need depth values,
    # or (p1 - p2).norm(dim=-1) as a confidence measure (smaller distance = more reliable intersection).
    return (p1 + p2) / 2
