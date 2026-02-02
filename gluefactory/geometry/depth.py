import functools

import kornia
import torch

from ..utils import misc
from . import absolute_pose, epipolar, reconstruction


def shape_normalize(kpts, w, h):
    """Normalize points to [-1, 1] range."""
    kpts = kpts.clone()
    kpts[..., 0] = kpts[..., 0] * 2 / w - 1
    kpts[..., 1] = kpts[..., 1] * 2 / h - 1

    kpts = kpts[:, None]
    return kpts


def sample_fmap(pts, fmap):
    h, w = fmap.shape[-2:]
    grid_sample = torch.nn.functional.grid_sample
    pts = shape_normalize(pts, w, h)
    # @TODO: This might still be a source of noise --> bilinear interpolation dangerous
    interp_lin = grid_sample(fmap, pts, align_corners=False, mode="bilinear")
    interp_nn = grid_sample(fmap, pts, align_corners=False, mode="nearest")
    return torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)[:, :, 0].permute(
        0, 2, 1
    )


def sample_depth(pts, depth_):
    depth = torch.where(depth_ > 0, depth_, torch.nan)
    if depth_.dim() == 2:
        depth = depth[None]
    depth = depth[:, None]
    interp = sample_fmap(pts, depth).squeeze(-1)
    valid = (~torch.isnan(interp)) & (interp > 0)
    if depth_.dim() == 2:
        interp = interp[0]
        valid = valid[0]
    interp[~valid] = 0.0
    return interp, valid


def sample_normals_from_depth(pts, depth, K):
    depth = depth[:, None]
    normals = kornia.geometry.depth.depth_to_normals(depth, K)
    normals = torch.where(depth > 0, normals, 0.0)
    interp = sample_fmap(pts, normals)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


@misc.AMP_CUSTOM_FWD_F32
def project(
    kpi,
    di,
    depthj,
    camera_i,
    camera_j,
    T_itoj,
    ccth=None,
    sample_depth_fun=sample_depth,
    sample_depth_kwargs=None,
    max_rel_depth_error=None,
    add_epi_outliers=False,
):
    if sample_depth_kwargs is None:
        sample_depth_kwargs = {}

    kpi_3d_i = camera_i.image2cam(kpi)
    kpi_3d_i = kpi_3d_i * di[..., None]
    kpi_3d_j = T_itoj.transform(kpi_3d_i)
    kpi_j, valid = camera_j.cam2image(kpi_3d_j)
    invalid = ~valid & (di > 1.0e-3)
    if add_epi_outliers:
        i1_F_i0 = epipolar.T_to_F(camera_i, camera_j, T_itoj)

        if i1_F_i0.ndim == 2 and kpi.ndim == 3 and kpi.shape[0] == 1:
            evalid0 = epipolar.check_epipolar_intersection(
                kpi[0], i1_F_i0, camera_j.size[..., 0], camera_j.size[..., 1]
            )[None]
        else:
            evalid0 = torch.vmap(
                epipolar.check_epipolar_intersection,
            )(kpi, i1_F_i0, camera_j.size[..., 0], camera_j.size[..., 1])

        invalid = invalid | (~evalid0)
    # di_j = kpi_3d_j[..., -1]
    if depthj is None or ccth is None:
        return kpi_j, valid, invalid
    else:
        # circle consistency
        dj, validj = sample_depth_fun(kpi_j, depthj, **sample_depth_kwargs)
        validj = validj & (dj > 1.0e-3)
        kpi_j_3d_j = camera_j.image2cam(kpi_j) * dj[..., None]
        kpi_j_3d_i = T_itoj.inv().transform(kpi_j_3d_j)
        dji = kpi_j_3d_i[..., -1]
        if max_rel_depth_error is not None:
            max_rel_depth = 1.0 + max_rel_depth_error
            dij = kpi_3d_j[..., -1]
            invalid = invalid | (
                (di > 0)
                & validj
                & ~((dji < di * max_rel_depth) & (dji > di / max_rel_depth))
                & ~((dij < dj * max_rel_depth) & (dij > dj / max_rel_depth))
            )
        kpi_j_i, validj_i = camera_i.cam2image(kpi_j_3d_i)
        reproj_error = ((kpi - kpi_j_i) ** 2).sum(-1)
        consistent = reproj_error < ccth**2
        inconsistent = reproj_error > ccth**2
        visible = valid & consistent & validj_i & validj & ~invalid
        invalid = invalid | (
            (validj & ((~validj_i) | (inconsistent))) & valid & (di > 1.0e-3)
        )
        # visible = validi
        return kpi_j, visible, invalid


def dense_warp_consistency(
    depthi: torch.Tensor,
    depthj: torch.Tensor,
    T_itoj: torch.Tensor,
    camerai: reconstruction.Camera,
    cameraj: reconstruction.Camera,
    **kwargs,
):
    kpi = misc.get_image_coords(depthi).flatten(-3, -2)
    di = depthi.flatten(
        -2,
    )
    validi = di > 0
    kpir, validir, invalid = project(
        kpi, di, depthj, camerai, cameraj, T_itoj, **kwargs
    )
    validir = validir & validi

    return (
        kpir.unflatten(-2, depthi.shape[-2:]),
        validir.unflatten(-1, (depthi.shape[-2:])),
        invalid.unflatten(-1, (depthi.shape[-2:])),
    )


def symmetric_reprojection_error(
    pts0: torch.Tensor,  # B x N x 2
    pts1: torch.Tensor,  # B x N x 2
    camera0: reconstruction.Camera,
    camera1: reconstruction.Camera,
    T_0to1: reconstruction.Pose,
    depth0: torch.Tensor,
    depth1: torch.Tensor,
    ccth: float = 10,
    agg: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    T_1to0 = T_0to1.inv()
    d0, valid0 = sample_depth(pts0, depth0)
    d1, valid1 = sample_depth(pts1, depth1)

    pts0_1, visible0, _ = project(pts0, d0, depth1, camera0, camera1, T_0to1, ccth=ccth)
    visible0 = visible0 & valid0
    pts1_0, visible1, _ = project(pts1, d1, depth0, camera1, camera0, T_1to0, ccth=ccth)
    visible1 = visible1 & valid1

    if agg == "mean":
        reprojection_errors_px = 0.5 * (
            (pts0_1 - pts1).norm(dim=-1) + (pts1_0 - pts0).norm(dim=-1)
        )
    elif agg == "max":
        reprojection_errors_px = torch.max(
            (pts0_1 - pts1).norm(dim=-1), (pts1_0 - pts0).norm(dim=-1)
        )
    elif agg == "min":
        reprojection_errors_px = torch.min(
            (pts0_1 - pts1).norm(dim=-1), (pts1_0 - pts0).norm(dim=-1)
        )
    else:
        raise ValueError(f"Unknown agg method: {agg}")

    valid = valid0 & valid1
    return reprojection_errors_px, valid


def symmetric_bias(
    pts0: torch.Tensor,  # B x N x 2
    pts1: torch.Tensor,  # B x N x 2
    camera0: reconstruction.Camera,
    camera1: reconstruction.Camera,
    T_0to1: reconstruction.Pose,
    depth0: torch.Tensor,
    depth1: torch.Tensor,
    ccth: float = 10,
    agg: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    T_1to0 = T_0to1.inv()
    d0, valid0 = sample_depth(pts0, depth0)
    d1, valid1 = sample_depth(pts1, depth1)

    pts0_1, visible0, _ = project(pts0, d0, depth1, camera0, camera1, T_0to1, ccth=ccth)
    visible0 = visible0 & valid0
    pts1_0, visible1, _ = project(pts1, d1, depth0, camera1, camera0, T_1to0, ccth=ccth)
    visible1 = visible1 & valid1

    bias = 0.5 * ((pts0_1 - pts1) + (pts1_0 - pts0))

    valid = valid0 & valid1
    return bias, valid


def align_pointclouds(
    pts_v0: torch.Tensor,
    pts_v1: torch.Tensor,
    weights: torch.Tensor = None,
    return_Rt: bool = False,
    scale_only: bool = False,
) -> tuple[
    reconstruction.Pose | None | tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Estimate a similarity transformation (sim3) between two point clouds."""
    assert pts_v0.shape == pts_v1.shape, f"{pts_v0.shape} != {pts_v1.shape}"
    assert pts_v0.shape[-1] == 3 and len(pts_v0.shape) == 2, f"{pts_v0.shape}"
    pts_v0, pts_v1 = pts_v0.float(), pts_v1.float()
    if weights is not None:
        weights = weights.float()

    pts_v1_in = pts_v1.clone()
    # estimate a sim3 transformation to align two point clouds
    # find M = argmin ||P1 - M @ P2||
    if weights is None:
        weights = torch.ones_like(pts_v0[..., 0])
    weights = weights[:, None]

    t0 = misc.wmean(pts_v0, weights, dim=0)
    t1 = misc.wmean(pts_v1, weights, dim=0)

    if scale_only:
        t0 = torch.zeros_like(t0)
        t1 = torch.zeros_like(t1)
    pts_v0 = pts_v0 - t0[None, :]
    pts_v1 = pts_v1 - t1[None, :]

    s0 = misc.wmean(pts_v0.square().sum(dim=-1), weights[:, 0]).sqrt()
    s1 = misc.wmean(pts_v1.square().sum(dim=-1), weights[:, 0]).sqrt()

    # Set scale 1 if no weights (i.e. all invalid)
    s0 = torch.where(weights.sum() > 0, s0, torch.tensor(1.0, device=s0.device))
    s1 = torch.where(weights.sum() > 0, s1, torch.tensor(1.0, device=s1.device))

    pts_v0 = pts_v0 / s0
    pts_v1 = pts_v1 / s1

    pts_v0 = pts_v0 * weights
    # Do not mult here as this is used in the output
    # pts_v1 = pts_v1 * weights
    if scale_only:
        R = torch.eye(3, dtype=t0.dtype, device=t0.device)
    else:
        try:
            A = pts_v0.T @ pts_v1
            U, _, V = A.double().svd()
            U: torch.Tensor = U
            V: torch.Tensor = V
        except:
            print("Procustes failed: SVD did not converge!")
            s = s0 / s1
            return (
                reconstruction.Pose.identity(device=pts_v1.device).to_Rt(),
                s,
                pts_v1,
            )
        # build rotation matrix
        R = (U @ V.T).float()
        R = torch.stack(
            [R[:, 0], R[:, 1], R[:, 2] * R.det().sign()], dim=-1
        )  # ensure a right-handed coordinate system
    s = s0 / s1
    t = t0 - s * (t1 @ R.T)
    c0_t_c1 = reconstruction.Pose.from_Rt(R, t)
    pts1_v0 = c0_t_c1.transform(pts_v1_in * s)
    if return_Rt:
        return (R, t), s, pts1_v0
    else:
        return c0_t_c1, s, pts1_v0


def align_pointclouds_robust(
    pts_v0: torch.Tensor,
    pts_v1: torch.Tensor,
    weights: torch.Tensor = None,
    return_Rt: bool = False,
    scale_only: bool = False,
    num_iters: int = 5,
    robust_fn: str = "huber",
    robust_scale: float | None = None,  # None is automatic estimation
) -> tuple[
    reconstruction.Pose | None | tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    if weights is None:
        weights = torch.ones_like(pts_v0[..., 0])

    rweights = 1.0

    estimate_robust_scale = robust_scale is None
    if not estimate_robust_scale:
        robust_scale = torch.tensor(
            robust_scale, device=pts_v0.device, dtype=pts_v0.dtype
        )
        robust_scale = robust_scale[None, None]
    valid = weights > 1.0e-6

    for _ in range(num_iters):
        _, _, pts1_aligned = align_pointclouds(
            pts_v0, pts_v1, rweights * weights, return_Rt=False, scale_only=scale_only
        )

        # Compute residuals
        residuals = (pts_v0 - pts1_aligned).norm(dim=-1)

        if estimate_robust_scale:
            # Estimate robust scale using median absolute deviation
            med_residual = misc.masked_median(residuals, valid, dim=-1, keepdim=True)
            robust_scale = 1.4826 * misc.masked_median(
                torch.abs(residuals - med_residual), valid, dim=-1, keepdim=True
            )

        # Update weights using robust kernel derivative
        if robust_fn == "huber":
            rweights = torch.where(
                residuals < robust_scale,
                torch.ones_like(residuals),
                robust_scale / residuals.clamp(min=1e-8),
            )
        elif robust_fn == "cauchy":
            rweights = 1 / (1 + (residuals / robust_scale) ** 2)
        elif robust_fn == "geman_mcclure":
            rweights = 1 / (1 + (residuals / robust_scale) ** 2) ** 2
        else:
            raise ValueError(f"Unknown robust function: {robust_fn}")

    # Final pass with converged weights
    return align_pointclouds(pts_v0, pts_v1, rweights * weights, return_Rt, scale_only)


def batch_align_pointclouds(
    pts_v0: torch.Tensor,
    pts_v1: torch.Tensor,
    weights: torch.Tensor = None,
    scale_only: bool = False,
    num_iters: int = 0,
    align_normalized: bool = False,
    **kwargs,
) -> tuple[reconstruction.Pose | None, torch.Tensor, torch.Tensor]:

    in_dims = (0, 0, 0) if weights is not None else (0, 0)

    if align_normalized:
        pts_v0n, c0n_t_c0 = absolute_pose._mean_isotropic_scale_normalize(
            pts_v0, return_pose=True
        )
        c0n_t_c0, norm_scale = c0n_t_c0.normalize_rotation()
    else:
        pts_v0n = pts_v0
        c0n_t_c0 = reconstruction.Pose.identity(device=pts_v0.device)[None]
        norm_scale = torch.ones(pts_v0.shape[0], device=pts_v0.device)

    c0n_Rt_c1, scales, pts1_v0n = torch.vmap(
        functools.partial(
            align_pointclouds_robust,
            return_Rt=True,
            scale_only=scale_only,
            num_iters=num_iters,
            **kwargs,
        ),
        in_dims=in_dims,
        out_dims=0,
    )(pts_v0n, pts_v1, weights)

    # Scale c0n_t_c1's translation to match normalized frame convention
    R_n, t_n = c0n_Rt_c1
    t_n_scaled = t_n / norm_scale[..., None]
    c0n_t_c1 = reconstruction.Pose.from_Rt(R_n, t_n_scaled)
    c0_t_c1 = c0n_t_c0.inv() @ c0n_t_c1

    # Correct scale: undo normalization scaling
    scales = scales / norm_scale

    # Correct points: scale back to original frame
    pts1_v0 = c0n_t_c0.inv().transform(pts1_v0n) / norm_scale[..., None, None]

    return c0_t_c1, scales, pts1_v0


def conormalize_pointclouds(
    xyz_gt: torch.Tensor,  # To estimate scale and translation
    *xyz_preds: torch.Tensor,  # To be transformed
    valid: torch.Tensor | None = None,  # On which to compute the normalization
) -> tuple[torch.Tensor, ...] | torch.Tensor:
    """Normalize point clouds to have zero mean and isotropic unit scale."""
    if valid is None:
        valid = torch.ones_like(xyz_gt[..., 0])
    xyz_gt_n, n_t_gt = absolute_pose._mean_isotropic_scale_normalize(
        xyz_gt, weights=valid, return_pose=True
    )
    xyz_preds_n = [n_t_gt.transform(xyz_pred) for xyz_pred in xyz_preds]
    if len(xyz_preds_n) == 0:
        return xyz_gt_n
    return xyz_gt_n, *xyz_preds_n
