import torch

from .desc_losses import get_sparse_desc
from .homography_adaptation import warp_points


def get_metric_one_way(kp0, kp1, valid_mask0, valid_mask1, H, thresh=3):
    """Given a list of keypoints [B, N, 2] in 2 images (xy convention)
    related by a homography, compute the repeatability score and
    localization error from image 0 to 1."""
    # Warp the kp0 to image 1
    w_kp0 = warp_points(kp0[:, :, [1, 0]], H)[:, :, [1, 0]]

    # Build the [B, N, N] distance matrix
    dist = torch.norm(w_kp0[:, :, None] - kp1[:, None], dim=3)
    dist[~valid_mask0] = thresh
    dist = dist.permute(0, 2, 1)
    dist[~valid_mask1] = thresh
    dist = dist.permute(0, 2, 1)

    # Find the closest keypoints and whether it is close enough
    min_dist = dist.min(dim=1)[0]
    valid = (min_dist < thresh).float()
    rep = valid.mean(dim=1)
    num_valid = valid.sum(dim=1)
    num_valid[num_valid == 0] = 1
    loc_err = (min_dist * valid).sum(dim=1) / num_valid
    return rep, loc_err


def get_repeatability_and_loc_error(kp0, kp1, scores0, scores1, H, thresh=3):
    """Given a list of keypoints [B, N, 2] in 2 images (xy convention)
    related by a homography, compute the repeatability score
    and localization error."""
    # Ignore points with score 0
    valid_mask0 = scores0 > 0
    valid_mask1 = scores1 > 0

    # One way 0 -> 1
    rep, loc_err = get_metric_one_way(kp0, kp1, valid_mask0, valid_mask1, H, thresh)

    # One way 1 -> 0
    rep2, loc_err2 = get_metric_one_way(
        kp1, kp0, valid_mask1, valid_mask0, torch.inverse(H), thresh
    )

    rep = (rep + rep2) / 2
    loc_err = (loc_err + loc_err2) / 2
    return rep, loc_err


def matching_score(kp0, valid_kp0, H, dense_desc0, dense_desc1):
    # Compute the descriptor distances
    kp1, valid_mask, valid_desc0, valid_desc1 = get_sparse_desc(
        kp0, valid_kp0, H, dense_desc0, dense_desc1
    )
    valid_desc0 = valid_desc0[valid_mask]
    valid_desc1 = valid_desc1[valid_mask]
    desc_sim = valid_desc0 @ valid_desc1.t()

    # Compute the percentage of correct matches
    matches0 = torch.max(desc_sim, dim=1)[1]
    matches1 = torch.max(desc_sim, dim=0)[1]
    m_score0 = matches0 == torch.arange(len(matches0), device=kp0.device)
    m_score1 = matches1 == torch.arange(len(matches1), device=kp0.device)
    m_score = (m_score0.float().mean() + m_score1.float().mean()) / 2
    return m_score.repeat(len(kp0))
