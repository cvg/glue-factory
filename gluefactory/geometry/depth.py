import kornia
import torch

from .utils import get_image_coords
from .wrappers import Camera, Pose


def sample_fmap(pts, fmap):
    h, w = fmap.shape[-2:]
    grid_sample = torch.nn.functional.grid_sample
    pts = (pts / pts.new_tensor([[w, h]]) * 2 - 1)[:, None]
    # @TODO: This might still be a source of noise --> bilinear interpolation dangerous
    interp_lin = grid_sample(fmap, pts, align_corners=False, mode="bilinear")
    interp_nn = grid_sample(fmap, pts, align_corners=False, mode="nearest")
    return torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)[:, :, 0].permute(
        0, 2, 1
    )


def sample_depth(pts, depth_):
    depth = torch.where(depth_ > 0, depth_, depth_.new_tensor(float("nan")))
    depth = depth[:, None]
    interp = sample_fmap(pts, depth).squeeze(-1)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


def sample_normals_from_depth(pts, depth, K):
    depth = depth[:, None]
    normals = kornia.geometry.depth.depth_to_normals(depth, K)
    normals = torch.where(depth > 0, normals, 0.0)
    interp = sample_fmap(pts, normals)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


def project(
    kpi,
    di,
    depthj,
    camera_i,
    camera_j,
    T_itoj,
    validi,
    ccth=None,
    sample_depth_fun=sample_depth,
    sample_depth_kwargs=None,
):
    if sample_depth_kwargs is None:
        sample_depth_kwargs = {}

    kpi_3d_i = camera_i.image2cam(kpi)
    kpi_3d_i = kpi_3d_i * di[..., None]
    kpi_3d_j = T_itoj.transform(kpi_3d_i)
    kpi_j, validj = camera_j.cam2image(kpi_3d_j)
    # di_j = kpi_3d_j[..., -1]
    validi = validi & validj
    if depthj is None or ccth is None:
        return kpi_j, validi & validj
    else:
        # circle consistency
        dj, validj = sample_depth_fun(kpi_j, depthj, **sample_depth_kwargs)
        kpi_j_3d_j = camera_j.image2cam(kpi_j) * dj[..., None]
        kpi_j_i, validj_i = camera_i.cam2image(T_itoj.inv().transform(kpi_j_3d_j))
        consistent = ((kpi - kpi_j_i) ** 2).sum(-1) < ccth
        visible = validi & consistent & validj_i & validj
        # visible = validi
        return kpi_j, visible


def dense_warp_consistency(
    depthi: torch.Tensor,
    depthj: torch.Tensor,
    T_itoj: torch.Tensor,
    camerai: Camera,
    cameraj: Camera,
    **kwargs,
):
    kpi = get_image_coords(depthi).flatten(-3, -2)
    di = depthi.flatten(
        -2,
    )
    validi = di > 0
    kpir, validir = project(kpi, di, depthj, camerai, cameraj, T_itoj, validi, **kwargs)

    return kpir.unflatten(-2, depthi.shape[-2:]), validir.unflatten(
        -1, (depthj.shape[-2:])
    )


def symmetric_reprojection_error(
    pts0: torch.Tensor,  # B x N x 2
    pts1: torch.Tensor,  # B x N x 2
    camera0: Camera,
    camera1: Camera,
    T_0to1: Pose,
    depth0: torch.Tensor,
    depth1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    T_1to0 = T_0to1.inv()
    d0, valid0 = sample_depth(pts0, depth0)
    d1, valid1 = sample_depth(pts1, depth1)

    pts0_1, visible0 = project(
        pts0, d0, depth1, camera0, camera1, T_0to1, valid0, ccth=None
    )
    pts1_0, visible1 = project(
        pts1, d1, depth0, camera1, camera0, T_1to0, valid1, ccth=None
    )

    reprojection_errors_px = 0.5 * (
        (pts0_1 - pts1).norm(dim=-1) + (pts1_0 - pts0).norm(dim=-1)
    )

    valid = valid0 & valid1
    return reprojection_errors_px, valid
