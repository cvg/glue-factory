import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .depth import project, sample_depth
from .epipolar import T_to_E, sym_epipolar_distance_all
from .homography import warp_points_torch

IGNORE_FEATURE = -2
UNMATCHED_FEATURE = -1


@torch.no_grad()
def gt_matches_from_pose_depth(
    kp0, kp1, data, pos_th=3, neg_th=5, epi_th=None, cc_th=None, **kw
):
    if kp0.shape[1] == 0 or kp1.shape[1] == 0:
        b_size, n_kp0 = kp0.shape[:2]
        n_kp1 = kp1.shape[1]
        assignment = torch.zeros(
            b_size, n_kp0, n_kp1, dtype=torch.bool, device=kp0.device
        )
        m0 = -torch.ones_like(kp0[:, :, 0]).long()
        m1 = -torch.ones_like(kp1[:, :, 0]).long()
        return assignment, m0, m1
    camera0, camera1 = data["view0"]["camera"], data["view1"]["camera"]
    T_0to1, T_1to0 = data["T_0to1"], data["T_1to0"]

    depth0 = data["view0"].get("depth")
    depth1 = data["view1"].get("depth")
    if "depth_keypoints0" in kw and "depth_keypoints1" in kw:
        d0, valid0 = kw["depth_keypoints0"], kw["valid_depth_keypoints0"]
        d1, valid1 = kw["depth_keypoints1"], kw["valid_depth_keypoints1"]
    else:
        assert depth0 is not None
        assert depth1 is not None
        d0, valid0 = sample_depth(kp0, depth0)
        d1, valid1 = sample_depth(kp1, depth1)

    kp0_1, visible0 = project(
        kp0, d0, depth1, camera0, camera1, T_0to1, valid0, ccth=cc_th
    )
    kp1_0, visible1 = project(
        kp1, d1, depth0, camera1, camera0, T_1to0, valid1, ccth=cc_th
    )
    mask_visible = visible0.unsqueeze(-1) & visible1.unsqueeze(-2)

    # build a distance matrix of size [... x M x N]
    dist0 = torch.sum((kp0_1.unsqueeze(-2) - kp1.unsqueeze(-3)) ** 2, -1)
    dist1 = torch.sum((kp0.unsqueeze(-2) - kp1_0.unsqueeze(-3)) ** 2, -1)
    dist = torch.max(dist0, dist1)
    inf = dist.new_tensor(float("inf"))
    dist = torch.where(mask_visible, dist, inf)

    min0 = dist.min(-1).indices
    min1 = dist.min(-2).indices

    ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
    ismin1 = ismin0.clone()
    ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
    ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
    positive = ismin0 & ismin1 & (dist < pos_th**2)

    negative0 = (dist0.min(-1).values > neg_th**2) & valid0
    negative1 = (dist1.min(-2).values > neg_th**2) & valid1

    # pack the indices of positive matches
    # if -1: unmatched point
    # if -2: ignore point
    unmatched = min0.new_tensor(UNMATCHED_FEATURE)
    ignore = min0.new_tensor(IGNORE_FEATURE)
    m0 = torch.where(positive.any(-1), min0, ignore)
    m1 = torch.where(positive.any(-2), min1, ignore)
    m0 = torch.where(negative0, unmatched, m0)
    m1 = torch.where(negative1, unmatched, m1)

    F = (
        camera1.calibration_matrix().inverse().transpose(-1, -2)
        @ T_to_E(T_0to1)
        @ camera0.calibration_matrix().inverse()
    )
    epi_dist = sym_epipolar_distance_all(kp0, kp1, F)

    # Add some more unmatched points using epipolar geometry
    if epi_th is not None:
        mask_ignore = (m0.unsqueeze(-1) == ignore) & (m1.unsqueeze(-2) == ignore)
        epi_dist = torch.where(mask_ignore, epi_dist, inf)
        exclude0 = epi_dist.min(-1).values > neg_th
        exclude1 = epi_dist.min(-2).values > neg_th
        m0 = torch.where((~valid0) & exclude0, ignore.new_tensor(-1), m0)
        m1 = torch.where((~valid1) & exclude1, ignore.new_tensor(-1), m1)

    return {
        "assignment": positive,
        "reward": (dist < pos_th**2).float() - (epi_dist > neg_th).float(),
        "matches0": m0,
        "matches1": m1,
        "matching_scores0": (m0 > -1).float(),
        "matching_scores1": (m1 > -1).float(),
        "depth_keypoints0": d0,
        "depth_keypoints1": d1,
        "proj_0to1": kp0_1,
        "proj_1to0": kp1_0,
        "visible0": visible0,
        "visible1": visible1,
    }


@torch.no_grad()
def gt_matches_from_homography(kp0, kp1, H, pos_th=3, neg_th=6, **kw):
    if kp0.shape[1] == 0 or kp1.shape[1] == 0:
        b_size, n_kp0 = kp0.shape[:2]
        n_kp1 = kp1.shape[1]
        assignment = torch.zeros(
            b_size, n_kp0, n_kp1, dtype=torch.bool, device=kp0.device
        )
        m0 = -torch.ones_like(kp0[:, :, 0]).long()
        m1 = -torch.ones_like(kp1[:, :, 0]).long()
        return assignment, m0, m1
    kp0_1 = warp_points_torch(kp0, H, inverse=False)
    kp1_0 = warp_points_torch(kp1, H, inverse=True)

    # build a distance matrix of size [... x M x N]
    dist0 = torch.sum((kp0_1.unsqueeze(-2) - kp1.unsqueeze(-3)) ** 2, -1)
    dist1 = torch.sum((kp0.unsqueeze(-2) - kp1_0.unsqueeze(-3)) ** 2, -1)
    dist = torch.max(dist0, dist1)

    reward = (dist < pos_th**2).float() - (dist > neg_th**2).float()

    min0 = dist.min(-1).indices
    min1 = dist.min(-2).indices

    ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
    ismin1 = ismin0.clone()
    ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
    ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
    positive = ismin0 & ismin1 & (dist < pos_th**2)

    negative0 = dist0.min(-1).values > neg_th**2
    negative1 = dist1.min(-2).values > neg_th**2

    # pack the indices of positive matches
    # if -1: unmatched point
    # if -2: ignore point
    unmatched = min0.new_tensor(UNMATCHED_FEATURE)
    ignore = min0.new_tensor(IGNORE_FEATURE)
    m0 = torch.where(positive.any(-1), min0, ignore)
    m1 = torch.where(positive.any(-2), min1, ignore)
    m0 = torch.where(negative0, unmatched, m0)
    m1 = torch.where(negative1, unmatched, m1)

    return {
        "assignment": positive,
        "reward": reward,
        "matches0": m0,
        "matches1": m1,
        "matching_scores0": (m0 > -1).float(),
        "matching_scores1": (m1 > -1).float(),
        "proj_0to1": kp0_1,
        "proj_1to0": kp1_0,
    }


def sample_pts(lines, npts):
    dir_vec = (lines[..., 2:4] - lines[..., :2]) / (npts - 1)
    pts = lines[..., :2, np.newaxis] + dir_vec[..., np.newaxis].expand(
        dir_vec.shape + (npts,)
    ) * torch.arange(npts).to(lines)
    pts = torch.transpose(pts, -1, -2)
    return pts


def torch_perp_dist(segs2d, points_2d):
    # Check batch size and segments format
    assert segs2d.shape[0] == points_2d.shape[0]
    assert segs2d.shape[-1] == 4
    dir = segs2d[..., 2:] - segs2d[..., :2]
    sizes = torch.norm(dir, dim=-1).half()
    norm_dir = dir / torch.unsqueeze(sizes, dim=-1)
    # middle_ptn = 0.5 * (segs2d[..., 2:] + segs2d[..., :2])
    # centered [batch, nsegs0, nsegs1, n_sampled_pts, 2]
    centered = points_2d[:, None] - segs2d[..., None, None, 2:]

    R = torch.cat(
        [
            norm_dir[..., 0, None],
            norm_dir[..., 1, None],
            -norm_dir[..., 1, None],
            norm_dir[..., 0, None],
        ],
        dim=2,
    ).reshape((len(segs2d), -1, 2, 2))
    # Try to reduce the memory consumption by using float16 type
    if centered.is_cuda:
        centered, R = centered.half(), R.half()
    # R: [batch, nsegs0, 2, 2] , centered: [batch, nsegs1, n_sampled_pts, 2]
    #    -> [batch, nsegs0, nsegs1, n_sampled_pts, 2]
    rotated = torch.einsum("bdji,bdepi->bdepj", R, centered)

    overlaping = (rotated[..., 0] <= 0) & (
        torch.abs(rotated[..., 0]) <= sizes[..., None, None]
    )

    return torch.abs(rotated[..., 1]), overlaping


@torch.no_grad()
def gt_line_matches_from_pose_depth(
    pred_lines0,
    pred_lines1,
    valid_lines0,
    valid_lines1,
    data,
    npts=50,
    dist_th=5,
    overlap_th=0.2,
    min_visibility_th=0.5,
):
    """Compute ground truth line matches and label the remaining the lines as:
    - UNMATCHED: if reprojection is outside the image
                 or far away from any other line.
    - IGNORE: if a line has not enough valid depth pixels along itself
              or it is labeled as invalid."""
    lines0 = pred_lines0.clone()
    lines1 = pred_lines1.clone()

    if pred_lines0.shape[1] == 0 or pred_lines1.shape[1] == 0:
        bsize, nlines0, nlines1 = (
            pred_lines0.shape[0],
            pred_lines0.shape[1],
            pred_lines1.shape[1],
        )
        positive = torch.zeros(
            (bsize, nlines0, nlines1), dtype=torch.bool, device=pred_lines0.device
        )
        m0 = torch.full((bsize, nlines0), -1, device=pred_lines0.device)
        m1 = torch.full((bsize, nlines1), -1, device=pred_lines0.device)
        return positive, m0, m1

    if lines0.shape[-2:] == (2, 2):
        lines0 = torch.flatten(lines0, -2)
    elif lines0.dim() == 4:
        lines0 = torch.cat([lines0[:, :, 0], lines0[:, :, -1]], dim=2)
    if lines1.shape[-2:] == (2, 2):
        lines1 = torch.flatten(lines1, -2)
    elif lines1.dim() == 4:
        lines1 = torch.cat([lines1[:, :, 0], lines1[:, :, -1]], dim=2)
    b_size, n_lines0, _ = lines0.shape
    b_size, n_lines1, _ = lines1.shape
    h0, w0 = data["view0"]["depth"][0].shape
    h1, w1 = data["view1"]["depth"][0].shape

    lines0 = torch.min(
        torch.max(lines0, torch.zeros_like(lines0)),
        lines0.new_tensor([w0 - 1, h0 - 1, w0 - 1, h0 - 1], dtype=torch.float),
    )
    lines1 = torch.min(
        torch.max(lines1, torch.zeros_like(lines1)),
        lines1.new_tensor([w1 - 1, h1 - 1, w1 - 1, h1 - 1], dtype=torch.float),
    )

    # Sample points along each line
    pts0 = sample_pts(lines0, npts).reshape(b_size, n_lines0 * npts, 2)
    pts1 = sample_pts(lines1, npts).reshape(b_size, n_lines1 * npts, 2)

    # Sample depth and valid points
    d0, valid0_pts0 = sample_depth(pts0, data["view0"]["depth"])
    d1, valid1_pts1 = sample_depth(pts1, data["view1"]["depth"])

    # Reproject to the other view
    pts0_1, visible0 = project(
        pts0,
        d0,
        data["view1"]["depth"],
        data["view0"]["camera"],
        data["view1"]["camera"],
        data["T_0to1"],
        valid0_pts0,
    )
    pts1_0, visible1 = project(
        pts1,
        d1,
        data["view0"]["depth"],
        data["view1"]["camera"],
        data["view0"]["camera"],
        data["T_1to0"],
        valid1_pts1,
    )

    h0, w0 = data["view0"]["image"].shape[-2:]
    h1, w1 = data["view1"]["image"].shape[-2:]
    # If a line has less than min_visibility_th inside the image is considered OUTSIDE
    pts_out_of0 = (pts1_0 < 0).any(-1) | (
        pts1_0 >= torch.tensor([w0, h0]).to(pts1_0)
    ).any(-1)
    pts_out_of0 = pts_out_of0.reshape(b_size, n_lines1, npts).float()
    out_of0 = pts_out_of0.mean(dim=-1) >= (1 - min_visibility_th)
    pts_out_of1 = (pts0_1 < 0).any(-1) | (
        pts0_1 >= torch.tensor([w1, h1]).to(pts0_1)
    ).any(-1)
    pts_out_of1 = pts_out_of1.reshape(b_size, n_lines0, npts).float()
    out_of1 = pts_out_of1.mean(dim=-1) >= (1 - min_visibility_th)

    # visible0 is [bs, nl0 * npts]
    pts0_1 = pts0_1.reshape(b_size, n_lines0, npts, 2)
    pts1_0 = pts1_0.reshape(b_size, n_lines1, npts, 2)

    perp_dists0, overlaping0 = torch_perp_dist(lines0, pts1_0)
    close_points0 = (perp_dists0 < dist_th) & overlaping0  # [bs, nl0, nl1, npts]
    del perp_dists0, overlaping0
    close_points0 = close_points0 * visible1.reshape(b_size, 1, n_lines1, npts)

    perp_dists1, overlaping1 = torch_perp_dist(lines1, pts0_1)
    close_points1 = (perp_dists1 < dist_th) & overlaping1  # [bs, nl1, nl0, npts]
    del perp_dists1, overlaping1
    close_points1 = close_points1 * visible0.reshape(b_size, 1, n_lines0, npts)
    torch.cuda.empty_cache()

    # For each segment detected in 0, how many sampled points from
    # reprojected segments 1 are close
    num_close_pts0 = close_points0.sum(dim=-1)  # [bs, nl0, nl1]

    # num_close_pts0_t = num_close_pts0.transpose(-1, -2)
    # For each segment detected in 1, how many sampled points from
    # reprojected segments 0 are close
    num_close_pts1 = close_points1.sum(dim=-1)
    num_close_pts1_t = num_close_pts1.transpose(-1, -2)  # [bs, nl1, nl0]
    num_close_pts = num_close_pts0 * num_close_pts1_t
    mask_close = (
        num_close_pts1_t
        > visible0.reshape(b_size, n_lines0, npts).float().sum(-1)[:, :, None]
        * overlap_th
    ) & (
        num_close_pts0
        > visible1.reshape(b_size, n_lines1, npts).float().sum(-1)[:, None] * overlap_th
    )
    # mask_close = (num_close_pts1_t > npts * overlap_th) & (
    # num_close_pts0 > npts * overlap_th)

    # Define the unmatched lines
    unmatched0 = torch.all(~mask_close, dim=2) | out_of1
    unmatched1 = torch.all(~mask_close, dim=1) | out_of0

    # Define the lines to ignore
    ignore0 = (
        valid0_pts0.reshape(b_size, n_lines0, npts).float().mean(dim=-1)
        < min_visibility_th
    ) | ~valid_lines0
    ignore1 = (
        valid1_pts1.reshape(b_size, n_lines1, npts).float().mean(dim=-1)
        < min_visibility_th
    ) | ~valid_lines1

    cost = -num_close_pts.clone()
    # High score for unmatched and non-valid lines
    cost[unmatched0] = 1e6
    cost[ignore0] = 1e6
    # TODO: Is it reasonable to forbid the matching with a segment because it
    #  has not GT depth?
    cost = cost.transpose(1, 2)
    cost[unmatched1] = 1e6
    cost[ignore1] = 1e6
    cost = cost.transpose(1, 2)

    # For each row, returns the col of max number of points
    assignation = np.array(
        [linear_sum_assignment(C) for C in cost.detach().cpu().numpy()]
    )
    assignation = torch.tensor(assignation).to(num_close_pts)
    # Set ignore and unmatched labels
    unmatched = assignation.new_tensor(UNMATCHED_FEATURE)
    ignore = assignation.new_tensor(IGNORE_FEATURE)

    positive = num_close_pts.new_zeros(num_close_pts.shape, dtype=torch.bool)
    all_in_batch = (
        torch.arange(b_size)[:, None].repeat(1, assignation.shape[-1]).flatten()
    )
    positive[all_in_batch, assignation[:, 0].flatten(), assignation[:, 1].flatten()] = (
        True
    )

    m0 = assignation.new_full((b_size, n_lines0), unmatched, dtype=torch.long)
    m0.scatter_(-1, assignation[:, 0], assignation[:, 1])
    m1 = assignation.new_full((b_size, n_lines1), unmatched, dtype=torch.long)
    m1.scatter_(-1, assignation[:, 1], assignation[:, 0])

    positive = positive & mask_close
    # Remove values to be ignored or unmatched
    positive[unmatched0] = False
    positive[ignore0] = False
    positive = positive.transpose(1, 2)
    positive[unmatched1] = False
    positive[ignore1] = False
    positive = positive.transpose(1, 2)
    m0[~positive.any(-1)] = unmatched
    m0[unmatched0] = unmatched
    m0[ignore0] = ignore
    m1[~positive.any(-2)] = unmatched
    m1[unmatched1] = unmatched
    m1[ignore1] = ignore

    if num_close_pts.numel() == 0:
        no_matches = torch.zeros(positive.shape[0], 0).to(positive)
        return positive, no_matches, no_matches

    return positive, m0, m1


@torch.no_grad()
def gt_line_matches_from_homography(
    pred_lines0,
    pred_lines1,
    valid_lines0,
    valid_lines1,
    shape0,
    shape1,
    H,
    npts=50,
    dist_th=5,
    overlap_th=0.2,
    min_visibility_th=0.2,
):
    """Compute ground truth line matches and label the remaining the lines as:
    - UNMATCHED: if reprojection is outside the image or far away from any other line.
    - IGNORE: if a line is labeled as invalid."""
    h0, w0 = shape0[-2:]
    h1, w1 = shape1[-2:]
    lines0 = pred_lines0.clone()
    lines1 = pred_lines1.clone()
    if lines0.shape[-2:] == (2, 2):
        lines0 = torch.flatten(lines0, -2)
    elif lines0.dim() == 4:
        lines0 = torch.cat([lines0[:, :, 0], lines0[:, :, -1]], dim=2)
    if lines1.shape[-2:] == (2, 2):
        lines1 = torch.flatten(lines1, -2)
    elif lines1.dim() == 4:
        lines1 = torch.cat([lines1[:, :, 0], lines1[:, :, -1]], dim=2)
    b_size, n_lines0, _ = lines0.shape
    b_size, n_lines1, _ = lines1.shape

    lines0 = torch.min(
        torch.max(lines0, torch.zeros_like(lines0)),
        lines0.new_tensor([w0 - 1, h0 - 1, w0 - 1, h0 - 1], dtype=torch.float),
    )
    lines1 = torch.min(
        torch.max(lines1, torch.zeros_like(lines1)),
        lines1.new_tensor([w1 - 1, h1 - 1, w1 - 1, h1 - 1], dtype=torch.float),
    )

    # Sample points along each line
    pts0 = sample_pts(lines0, npts).reshape(b_size, n_lines0 * npts, 2)
    pts1 = sample_pts(lines1, npts).reshape(b_size, n_lines1 * npts, 2)

    # Project the points to the other image
    pts0_1 = warp_points_torch(pts0, H, inverse=False)
    pts1_0 = warp_points_torch(pts1, H, inverse=True)
    pts0_1 = pts0_1.reshape(b_size, n_lines0, npts, 2)
    pts1_0 = pts1_0.reshape(b_size, n_lines1, npts, 2)

    # If a line has less than min_visibility_th inside the image is considered OUTSIDE
    pts_out_of0 = (pts1_0 < 0).any(-1) | (
        pts1_0 >= torch.tensor([w0, h0]).to(pts1_0)
    ).any(-1)
    pts_out_of0 = pts_out_of0.reshape(b_size, n_lines1, npts).float()
    out_of0 = pts_out_of0.mean(dim=-1) >= (1 - min_visibility_th)
    pts_out_of1 = (pts0_1 < 0).any(-1) | (
        pts0_1 >= torch.tensor([w1, h1]).to(pts0_1)
    ).any(-1)
    pts_out_of1 = pts_out_of1.reshape(b_size, n_lines0, npts).float()
    out_of1 = pts_out_of1.mean(dim=-1) >= (1 - min_visibility_th)

    perp_dists0, overlaping0 = torch_perp_dist(lines0, pts1_0)
    close_points0 = (perp_dists0 < dist_th) & overlaping0  # [bs, nl0, nl1, npts]
    del perp_dists0, overlaping0

    perp_dists1, overlaping1 = torch_perp_dist(lines1, pts0_1)
    close_points1 = (perp_dists1 < dist_th) & overlaping1  # [bs, nl1, nl0, npts]
    del perp_dists1, overlaping1
    torch.cuda.empty_cache()

    # For each segment detected in 0,
    # how many sampled points from reprojected segments 1 are close
    num_close_pts0 = close_points0.sum(dim=-1)  # [bs, nl0, nl1]
    # num_close_pts0_t = num_close_pts0.transpose(-1, -2)
    # For each segment detected in 1,
    # how many sampled points from reprojected segments 0 are close
    num_close_pts1 = close_points1.sum(dim=-1)
    num_close_pts1_t = num_close_pts1.transpose(-1, -2)  # [bs, nl1, nl0]

    num_close_pts = num_close_pts0 * num_close_pts1_t
    mask_close = (
        (num_close_pts1_t > npts * overlap_th)
        & (num_close_pts0 > npts * overlap_th)
        & ~out_of0.unsqueeze(1)
        & ~out_of1.unsqueeze(-1)
    )

    # Define the unmatched lines
    unmatched0 = torch.all(~mask_close, dim=2) | out_of1
    unmatched1 = torch.all(~mask_close, dim=1) | out_of0

    # Define the lines to ignore
    ignore0 = ~valid_lines0
    ignore1 = ~valid_lines1

    cost = -num_close_pts.clone()
    # High score for unmatched and non-valid lines
    cost[unmatched0] = 1e6
    cost[ignore0] = 1e6
    cost = cost.transpose(1, 2)
    cost[unmatched1] = 1e6
    cost[ignore1] = 1e6
    cost = cost.transpose(1, 2)
    # For each row, returns the col of max number of points
    assignation = np.array(
        [linear_sum_assignment(C) for C in cost.detach().cpu().numpy()]
    )
    assignation = torch.tensor(assignation).to(num_close_pts)

    # Set unmatched labels
    unmatched = assignation.new_tensor(UNMATCHED_FEATURE)
    ignore = assignation.new_tensor(IGNORE_FEATURE)

    positive = num_close_pts.new_zeros(num_close_pts.shape, dtype=torch.bool)
    # TODO Do with a single and beautiful call
    # for b in range(b_size):
    #     positive[b][assignation[b, 0], assignation[b, 1]] = True
    positive[
        torch.arange(b_size)[:, None].repeat(1, assignation.shape[-1]).flatten(),
        assignation[:, 0].flatten(),
        assignation[:, 1].flatten(),
    ] = True

    m0 = assignation.new_full((b_size, n_lines0), unmatched, dtype=torch.long)
    m0.scatter_(-1, assignation[:, 0], assignation[:, 1])
    m1 = assignation.new_full((b_size, n_lines1), unmatched, dtype=torch.long)
    m1.scatter_(-1, assignation[:, 1], assignation[:, 0])

    positive = positive & mask_close
    # Remove values to be ignored or unmatched
    positive[unmatched0] = False
    positive[ignore0] = False
    positive = positive.transpose(1, 2)
    positive[unmatched1] = False
    positive[ignore1] = False
    positive = positive.transpose(1, 2)
    m0[~positive.any(-1)] = unmatched
    m0[unmatched0] = unmatched
    m0[ignore0] = ignore
    m1[~positive.any(-2)] = unmatched
    m1[unmatched1] = unmatched
    m1[ignore1] = ignore

    if num_close_pts.numel() == 0:
        no_matches = torch.zeros(positive.shape[0], 0).to(positive)
        return positive, no_matches, no_matches

    return positive, m0, m1
