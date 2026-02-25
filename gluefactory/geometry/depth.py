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


def covisible_bbox(
    depth_i: torch.Tensor,
    camera_i: reconstruction.Camera,
    camera_j: reconstruction.Camera,
    T_itoj: reconstruction.Pose,
    depth_j: torch.Tensor | None = None,
    pad: int = 0,
    min_fraction: float = 0.1,
    max_rel_depth_error: float = 0.05,
    th_consistency: float = 10,
    stride: int = 4,
) -> torch.Tensor:
    """Bounding box of pixels in view i that project into view j.

    Args:
        depth_i: (B, H, W) depth map of view i
        camera_i, camera_j: Camera objects for views i and j
        T_itoj: Pose from view i to view j
        depth_j: (B, H, W) depth map of view j (optional, for circle consistency check)
        pad: padding in pixels around the bbox
        min_fraction: minimum bbox side as fraction of image size (default 0.1)
        max_rel_depth_error: maximum relative depth error for circle consistency (default 0.05)
        th_consistency: reprojection error threshold for circle consistency (default 10 pixels)
        stride: use every n-th pixel in each dimension (default 1, no subsampling)
    Returns:
        (B, 4) tensor of [wmin, hmin, wmax, hmax] in pixels, clamped to image
    """
    h, w = depth_i.shape[-2:]
    depth_s = depth_i[..., ::stride, ::stride]
    kpi = misc.get_image_coords(depth_s, expand=True) * stride
    hs, ws = depth_s.shape[-2:]
    kpi = kpi.flatten(-3, -2)
    di = depth_s.flatten(-2)

    _, visible, _ = project(
        kpi,
        di,
        depth_j,
        camera_i,
        camera_j,
        T_itoj,
        ccth=th_consistency,
        max_rel_depth_error=max_rel_depth_error,
    )
    covis = (visible & (di > 0)).unflatten(-1, (hs, ws))  # (B, hs, ws)

    row_any = covis.any(-1).int()  # (B, hs)
    col_any = covis.any(-2).int()  # (B, ws)
    hmin = (row_any.argmax(-1) * stride).float()
    hmax = (h - row_any.flip(-1).argmax(-1) * stride).float()
    wmin = (col_any.argmax(-1) * stride).float()
    wmax = (w - col_any.flip(-1).argmax(-1) * stride).float()

    # Enforce minimum bbox size
    min_h = min_fraction * h
    min_w = min_fraction * w
    h_center = 0.5 * (hmin + hmax)
    w_center = 0.5 * (wmin + wmax)
    hmin = torch.minimum(hmin, h_center - min_h / 2)
    hmax = torch.maximum(hmax, h_center + min_h / 2)
    wmin = torch.minimum(wmin, w_center - min_w / 2)
    wmax = torch.maximum(wmax, w_center + min_w / 2)

    wmin = (wmin - pad).clamp(min=0)
    hmin = (hmin - pad).clamp(min=0)
    wmax = (wmax + pad).clamp(max=w)
    hmax = (hmax + pad).clamp(max=h)
    return torch.stack([wmin, hmin, wmax, hmax], dim=-1)


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
    use_eigh: bool = False,
    eps: float = 0.0,
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

    # clamp before sqrt to block NaN gradients from sqrt(0)
    s0 = misc.wmean(pts_v0.square().sum(dim=-1), weights[:, 0]).clamp(min=eps).sqrt()
    s1 = misc.wmean(pts_v1.square().sum(dim=-1), weights[:, 0]).clamp(min=eps).sqrt()
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
    elif use_eigh:
        A = pts_v0.T @ pts_v1
        A = A + eps * torch.eye(3, device=A.device, dtype=A.dtype)
        ATA = (A.T @ A).double()
        eigenvalues, V = torch.linalg.eigh(ATA)
        S = eigenvalues.clamp(min=eps).sqrt()
        U = A.double() @ V / S[None, :]
        R = (U @ V.mT).float()
        R = torch.stack([R[:, 0], R[:, 1], R[:, 2] * R.det().sign()], dim=-1)
    else:
        try:
            A = pts_v0.T @ pts_v1
            # Regularize to prevent degenerate SVD (e.g. all-zero weights)
            A = A + eps * torch.eye(3, device=A.device, dtype=A.dtype)
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
    use_eigh: bool = False,
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
            pts_v0,
            pts_v1,
            rweights * weights,
            return_Rt=False,
            scale_only=scale_only,
            use_eigh=use_eigh,
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
    return align_pointclouds(
        pts_v0,
        pts_v1,
        rweights * weights,
        return_Rt,
        scale_only,
        use_eigh=use_eigh,
    )


@misc.force_f32
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
        c0n_t_c0 = reconstruction.Pose.identity(device=pts_v0.device)[None].expand(
            pts_v0.shape[0]
        )
        norm_scale = torch.ones(pts_v0.shape[0], device=pts_v0.device)

    if weights is not None:
        # This forces identity alignment for invalid weights, which is a reasonable default and prevents NaN gradients from degenerate SVD in align_pointclouds_robust
        w_valid = weights.sum(dim=-1, keepdim=True) > 1.0e-6
        weights = torch.where(w_valid, weights, torch.ones_like(weights))
        pts_v0n = torch.where(w_valid[..., None], pts_v0n, pts_v0.detach())
        pts_v1 = torch.where(w_valid[..., None], pts_v1, pts_v0n.detach())
        c0n_t_c0 = torch.where(w_valid[:, 0], c0n_t_c0, c0n_t_c0.detach())
        norm_scale = torch.where(w_valid[:, 0], norm_scale, norm_scale.detach())

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


def relative_pose_reprojection_residual(
    p3d_w: torch.Tensor,
    p2d_i0: torch.Tensor,
    p2d_i1: torch.Tensor,
    camera0: reconstruction.Camera,
    camera1: reconstruction.Camera,
    c1_tgt_c0: reconstruction.Pose,
    weights: torch.Tensor | None = None,
    num_align_iters: int = 0,
    **align_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, reconstruction.Pose]:
    """Per-point reprojection residual w.r.t. the GT relative pose.

    Finds the Sim(3) alignment of p3d_w that minimizes reprojection error
    in both views, where the views are related by the GT relative pose
    c1_tgt_c0 (cam0 -> cam1, with unknown translation scale).

    Internally solves for per-point depths in cam0 from the GT relative pose
    constraint on bearing vectors, then aligns predicted points via Procrustes.

    Args:
        p3d_w: (B, N, 3) predicted 3D points in arbitrary frame.
        p2d_i0: (B, N, 2) pixel coordinates in image 0.
        p2d_i1: (B, N, 2) pixel coordinates in image 1.
        camera0, camera1: Camera intrinsics.
        c1_tgt_c0: GT relative pose cam0 -> cam1 (translation scale arbitrary).
        weights: (B, N) optional point weights for alignment.
        num_align_iters: robust alignment iterations (0 = L2 Procrustes).
        **align_kwargs: passed to batch_align_pointclouds.

    Returns:
        reproj_i0: (B, N) reprojection error in image 0 (pixels).
        reproj_i1: (B, N) reprojection error in image 1 (pixels).
        p3d_c0: (B, N, 3) predicted points aligned to cam0 frame.
        c1_tgt_c0_w: GT relative pose with translation scaled to world frame.
    """
    # Bearing vectors (homogeneous camera coordinates)
    p2d_c0 = camera0.image2cam(p2d_i0)  # (B, N, 3), z=1
    p2d_c1 = camera1.image2cam(p2d_i1)

    c1_R_c0 = c1_tgt_c0.R  # (..., 3, 3)
    c1_tt_c0 = c1_tgt_c0.t  # (..., 3)

    # Relative pose constraint: d1*p2d_c1 = d0*c1_R_c0@p2d_c0 + s*c1_tt_c0
    # Cross with p2d_c1 to eliminate d1:
    #   d0 * (p2d_c1 x c1_R_c0@p2d_c0) + s * (p2d_c1 x c1_tt_c0) = 0
    # Set s=1 (Procrustes absorbs scale), solve for d0 per point:
    #   d0_i = -(a_i . b_i) / ||a_i||^2
    p2d0_c1 = (c1_R_c0[..., None, :, :] @ p2d_c0[..., None]).squeeze(-1)
    c1_tt_c0_exp = c1_tt_c0[..., None, :].expand_as(p2d_c1)
    a = torch.cross(p2d_c1, p2d0_c1, dim=-1)  # (B, N, 3)
    b = torch.cross(p2d_c1, c1_tt_c0_exp, dim=-1)  # (B, N, 3)
    d0 = -(a * b).sum(-1) / (a * a).sum(-1).clamp(min=1e-8)  # (B, N)

    # Reference points in cam0 frame (reprojection-consistent with GT pose)
    p3d_ref_c0 = d0[..., None] * p2d_c0  # (B, N, 3)

    # Sim(3) align predicted points to reference
    _, scale, p3d_c0 = batch_align_pointclouds(
        p3d_ref_c0,
        p3d_w,
        weights=weights,
        num_iters=num_align_iters,
        **align_kwargs,
    )

    # Reprojection in image 0
    p2d_i0_proj, _ = camera0.cam2image(p3d_c0)
    reproj_i0 = (p2d_i0_proj - p2d_i0).norm(dim=-1)

    # Reprojection in image 1 (using GT relative pose, s=1 matches reference)
    p3d_c1 = c1_tgt_c0.transform(p3d_c0)
    p2d_i1_proj, _ = camera1.cam2image(p3d_c1)
    reproj_i1 = (p2d_i1_proj - p2d_i1).norm(dim=-1)

    # GT relative pose scaled to world frame units
    c1_tgt_c0_w = reconstruction.Pose.from_Rt(c1_R_c0, c1_tt_c0 / scale[..., None])

    return reproj_i0, reproj_i1, p3d_c0, c1_tgt_c0_w


def _essential_matrix_8pt(
    b0: torch.Tensor,
    b1: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalised 8-point algorithm for the essential matrix.

    Args:
        b0: (B, N, 3) bearing vectors in cam 0 (from image2cam, z=1).
        b1: (B, N, 3) bearing vectors in cam 1.
        weights: (B, N) optional per-point weights.

    Returns:
        E: (B, 3, 3) essential matrix satisfying b1^T E b0 = 0.
    """
    from . import transforms as gtr

    B, N, _ = b0.shape

    # Hartley normalisation on the 2-D parts
    p0_norm, T0 = absolute_pose._mean_isotropic_scale_normalize(b0[..., :2])
    p1_norm, T1 = absolute_pose._mean_isotropic_scale_normalize(b1[..., :2])
    h0 = gtr.to_homogeneous(p0_norm)  # (B, N, 3)
    h1 = gtr.to_homogeneous(p1_norm)

    # Constraint matrix: (b1 ⊗ b0).vec(E) = 0  →  (B, N, 9)
    A = torch.einsum("bni,bnj->bnij", h1, h0).reshape(B, N, 9)
    if weights is not None:
        A = A * weights[..., None]

    # Solve via SVD
    _, _, Vh = torch.linalg.svd(A)
    E = Vh[..., -1, :].reshape(B, 3, 3)

    # Enforce rank-2 + equal singular values
    U, S, Vh = torch.linalg.svd(E)
    s_mean = (S[..., 0] + S[..., 1]) / 2
    S_new = torch.stack([s_mean, s_mean, torch.zeros_like(s_mean)], dim=-1)
    E = U @ torch.diag_embed(S_new) @ Vh

    # Denormalise: E_orig = T1^T @ E_norm @ T0
    E = T1.transpose(-1, -2) @ E @ T0
    return E


@misc.AMP_CUSTOM_FWD_F32
def relative_pnp(
    p3d_w: torch.Tensor,
    p2d_i0: torch.Tensor,
    p2d_i1: torch.Tensor,
    camera0: reconstruction.Camera,
    camera1: reconstruction.Camera,
    weights: torch.Tensor | None = None,
) -> reconstruction.Pose:
    """Relative pose from 3D points and their projections in two views.

    Uses the 8-point algorithm on the 2D-2D correspondences to recover
    (R, t_hat), the cross-product trick to recover per-point depths in
    camera 0, and the known 3D structure to recover translation scale.

    No absolute poses are computed.

    Args:
        p3d_w:   (B, N, 3) 3D points in an arbitrary coordinate frame.
        p2d_i0:  (B, N, 2) pixel coordinates in image 0.
        p2d_i1:  (B, N, 2) pixel coordinates in image 1.
        camera0: intrinsics for image 0.
        camera1: intrinsics for image 1.
        weights: (B, N) optional per-point weights.

    Returns:
        c1_T_c0: Pose transforming camera 0 -> camera 1.
    """
    # --- bearing vectors ---
    b0 = camera0.image2cam(p2d_i0)  # (B, N, 3)
    b1 = camera1.image2cam(p2d_i1)

    # --- essential matrix (8-point) ---
    E = _essential_matrix_8pt(b0, b1, weights)

    # --- decompose E → two (R, t) candidates ---
    R1, R2, t_hat = epipolar.decompose_essential_matrix(E)

    # --- chirality: pick (R, t_sign) that gives most positive depths ---
    best_R = R1
    best_t = t_hat
    best_count = t_hat.new_zeros(t_hat.shape[:-1])

    for R_cand, t_sign in [(R1, 1), (R1, -1), (R2, 1), (R2, -1)]:
        t_cand = t_sign * t_hat
        # cross-product depth recovery (same as relative_pose_reprojection_residual)
        Rb0 = (R_cand[..., None, :, :] @ b0[..., None]).squeeze(-1)
        a = torch.cross(b1, Rb0, dim=-1)
        c = torch.cross(b1, t_cand[..., None, :].expand_as(b1), dim=-1)
        d0 = -(a * c).sum(-1) / (a * a).sum(-1).clamp(min=1e-8)
        # depth in cam 1
        d1 = (R_cand[..., None, :, :] @ (d0[..., None] * b0)[..., None]).squeeze(-1)
        d1 = d1[..., 2] + t_cand[..., None, 2]
        count = ((d0 > 0) & (d1 > 0)).sum(-1)
        better = count > best_count
        best_R = torch.where(better[..., None, None], R_cand, best_R)
        best_t = torch.where(better[..., None], t_cand, best_t)
        best_count = torch.where(better, count, best_count)

    R, t_hat = best_R, best_t

    # --- depth ratios in cam 0 (with chosen R, t_hat) ---
    Rb0 = (R[..., None, :, :] @ b0[..., None]).squeeze(-1)
    a = torch.cross(b1, Rb0, dim=-1)
    c = torch.cross(b1, t_hat[..., None, :].expand_as(b1), dim=-1)
    f = -(a * c).sum(-1) / (a * a).sum(-1).clamp(min=1e-8)  # (B, N)

    # --- recover translation scale from 3D structure ---
    # Points in cam-0 frame (up to scale s):  Y_k = f_k * b0_k
    # True cam-0 points:  s * Y_k = R_abs @ X_k + t_abs
    # Procrustes recovers the scale s between Y and X_w.
    Y = f[..., None] * b0  # (B, N, 3)
    _, scale, _ = batch_align_pointclouds(Y, p3d_w, weights=weights)

    return reconstruction.Pose.from_Rt(R, t_hat / scale[..., None])


def world_rays(
    ci_t_w: reconstruction.Pose,
    camera: reconstruction.Camera,
    p2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute world-space rays through image points.

    Args:
        ci_t_w:  Pose world -> camera.
        camera:  Camera intrinsics.
        p2d:     (..., N, 2) pixel coordinates.

    Returns:
        origin:    (..., N, 3) camera center in world coordinates.
        direction: (..., N, 3) unit ray directions in world coordinates.
    """
    bearing = camera.image2cam(p2d)  # (..., N, 3)
    direction = bearing @ ci_t_w.R  # rotate to world: R^T @ bearing
    origin = (
        ci_t_w.inv().t[:, None].expand_as(direction)
    )  # world position of camera center
    return origin, direction


@misc.force_f32
def recover_pose_scale(
    pts3d_c0: torch.Tensor,
    p2d_i1: torch.Tensor,
    camera1: reconstruction.Camera,
    c1_T_c0: reconstruction.Pose,
    weights: torch.Tensor | None = None,
    robust: bool = False,
) -> torch.Tensor:
    """Recover translation scale for an up-to-scale relative pose.

    Args:
        pts3d_c0: (..., N, 3) 3D points in camera-0 coordinates.
        p2d_i1:   (..., N, 2) pixel projections in image 1.
        camera1:  Camera intrinsics for image 1.
        c1_T_c0:  Relative pose cam0 -> cam1 (unit translation).
        weights:  (..., N) optional per-point weights.
        robust:   If True, use median of per-point estimates.

    Returns:
        s: (...,) scale such that true_t = s * t.
    """
    p = camera1.image2cam(p2d_i1)  # (..., N, 3), z=1
    A = pts3d_c0 @ c1_T_c0.R.mT
    t = c1_T_c0.t
    a = torch.stack(
        [
            t[..., None, 0] - p[..., 0] * t[..., None, 2],
            t[..., None, 1] - p[..., 1] * t[..., None, 2],
        ],
        -1,
    ).flatten(-2)
    b = torch.stack(
        [p[..., 0] * A[..., 2] - A[..., 0], p[..., 1] * A[..., 2] - A[..., 1]], -1
    ).flatten(-2)
    if weights is None:
        weights = a.new_ones(pts3d_c0.shape[:-1])
    w = weights.repeat_interleave(2, dim=-1)  # (..., 2N)
    valid = a.abs() > 1e-8
    if robust:
        s_per = b / a.clamp(min=1e-8)
        return misc.masked_median(s_per, valid)
    return (w * a * b).sum(-1) / (w * a * a).sum(-1).clamp(min=1e-8)


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
