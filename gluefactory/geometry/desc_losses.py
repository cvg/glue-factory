from math import prod
import torch
import torch.nn.functional as F

from .utils import keypoints_to_grid
from .homography_adaptation import warp_points


def get_pos_dist(desc_dists):
    return torch.diag(desc_dists)


def get_neg_dist(desc_dists, dist_mask):
    device = desc_dists.device
    n_correct_points = len(desc_dists)
    desc_dists[
        torch.arange(n_correct_points, dtype=torch.long, device=device),
        torch.arange(n_correct_points, dtype=torch.long, device=device)] = 4
    desc_dists[dist_mask] = 4
    neg_dist = torch.min(torch.min(desc_dists, dim=1)[0],
                         torch.min(desc_dists, dim=0)[0])
    return neg_dist


def get_dist_mask(kp0, kp1, valid_mask, dist_thresh):
    """ Return a 2D matrix indicating the local neighborhood of each point
        for a given threshold and two lists of corresponding keypoints. """
    b_size, n_points, _ = kp0.size()
    dist_mask0 = torch.norm(kp0.unsqueeze(2) - kp0.unsqueeze(1), dim=-1)
    dist_mask1 = torch.norm(kp1.unsqueeze(2) - kp1.unsqueeze(1), dim=-1)
    dist_mask = torch.min(dist_mask0, dist_mask1)
    dist_mask = dist_mask <= dist_thresh
    dist_mask = dist_mask.repeat(1, 1, b_size).reshape(b_size * n_points,
                                                       b_size * n_points)
    dist_mask = dist_mask[valid_mask, :][:, valid_mask]
    return dist_mask


def get_sparse_desc(kp0, valid_kp0, H, desc0, desc1):
    """ Reproject keypoints with the homography H, extract sparse descriptors
        from both images, and keep track of the valid matches. """
    b_size, _, Hc, Wc = desc0.size()
    h, w = (Hc * 8, Wc * 8)

    # Warp the keypoints and keep valid ones
    kp1 = warp_points(kp0.float(), H)
    valid = ((kp1[:, :, 0] >= 0) & (kp1[:, :, 0] <= h - 1)
             & (kp1[:, :, 1] >= 0) & (kp1[:, :, 1] <= w - 1))
    kp1[~valid] = 0
    valid_mask = valid.float() * valid_kp0  # shape [B, n_kp]

    n_points = valid_mask.shape[1]
    valid_mask = valid_mask.bool()

    # Convert the keypoints to a grid suitable for interpolation
    grid0 = keypoints_to_grid(kp0, (h, w))
    grid1 = keypoints_to_grid(kp1, (h, w))

    # Extract the descriptors
    valid_desc0 = F.grid_sample(desc0, grid0, align_corners=False).permute(
        0, 2, 3, 1).reshape(b_size, n_points, -1)
    valid_desc0 = F.normalize(valid_desc0, dim=2)
    valid_desc1 = F.grid_sample(desc1, grid1, align_corners=False).permute(
        0, 2, 3, 1).reshape(b_size, n_points, -1)
    valid_desc1 = F.normalize(valid_desc1, dim=2)

    return kp1, valid_mask, valid_desc0, valid_desc1


def triplet_loss(kp0, valid_kp0, H, dense_desc0, dense_desc1,
                 margin=1., dist_threshold=8):
    """ Hardest in batch triplet loss. """
    # Compute the descriptor distances
    kp1, valid_mask, valid_desc0, valid_desc1 = get_sparse_desc(
        kp0, valid_kp0, H, dense_desc0, dense_desc1)
    valid_desc0 = valid_desc0[valid_mask]
    valid_desc1 = valid_desc1[valid_mask]
    desc_dist = 2 - 2 * (valid_desc0 @ valid_desc1.t())

    # Compute the distances between the keypoints of image1
    dist_mask = get_dist_mask(kp0, kp1, valid_mask.flatten(), dist_threshold)

    # Positive loss
    pos_dist = get_pos_dist(desc_dist)

    # Negative loss
    neg_dist = get_neg_dist(desc_dist, dist_mask)

    return torch.mean(F.relu(margin + pos_dist - neg_dist)).repeat(len(kp0))


def nll_loss(kp0, valid_kp0, H, dense_desc0, dense_desc1, temperature=1.):
    """ Negative log likelihood loss for keypoint matching,
        with dual softmax. """
    # Compute the descriptor cosine similarities
    _, valid_mask, desc0, desc1 = get_sparse_desc(
        kp0, valid_kp0, H, dense_desc0, dense_desc1)
    prob_desc = torch.einsum("bnd,bmd->bnm", desc0, desc1)

    # Mask out invalid keypoints
    prob_desc[~valid_mask] = -1
    prob_desc = prob_desc.permute(0, 2, 1)
    prob_desc[~valid_mask] = -1
    prob_desc = prob_desc.permute(0, 2, 1)

    # Dual softmax
    prob_desc = (F.log_softmax(prob_desc / temperature, dim=2)
                 + F.log_softmax(prob_desc / temperature, dim=1)) / 2

    # NLL
    loss = -torch.diagonal(prob_desc, dim1=1, dim2=2)
    valid_mask = valid_mask.float()
    loss = (loss * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
    return loss


def caps_loss(kp, kp_scores, descriptors, H, dense_desc_w, s=8):
    """ CAPS loss with direct supervision, as described in 
        https://arxiv.org/pdf/2004.13324.pdf.
    Args:
        kp: [B, n_kp, 2] tensor of KP in image 0.
        kp_scores: [B, n_kp] tensor (0 = invalid) of KP scores in image 0.
        descriptors: [B, n_kp, D] tensor of descriptors in image 0.
        H: [B, 3, 3] homography from image 0 to 1.
        dense_desc_w: [B, D, Hc, Wc] tensor of dense descriptors in image 1.
    """
    b_size, dim, Hc, Wc = dense_desc_w.size()
    h, w = (Hc * s, Wc * s)
    device = kp.device
    eps = 1e-8

    # Reproject keypoints to the other image and keep track of valid ones
    kp_w = warp_points(kp.float(), H)
    valid = ((kp_w[:, :, 0] >= 0) & (kp_w[:, :, 0] <= h - 1)
             & (kp_w[:, :, 1] >= 0) & (kp_w[:, :, 1] <= w - 1))
    kp_w[~valid] = 0
    valid_mask = valid.float() * (kp_scores > 0)  # shape [B, n_kp]

    # Correlate the kp desc with a grid of descriptors in the other image
    desc_grid = dense_desc_w[:, :, ::2, ::2].reshape(b_size, dim, -1)
    # desc_grid has 1/(s*2) = 1/16 of the img resolution
    corr = torch.einsum('bnd,bdm->bnm', descriptors, desc_grid)
    corr = torch.softmax(corr, dim=2)  # corr is [B, n_kp, Hc*Wc/4]

    # Compute the grid locations
    grid = torch.stack(torch.meshgrid(
        torch.arange(Hc // 2, dtype=torch.float, device=device),
        torch.arange(Wc // 2, dtype=torch.float, device=device),
        indexing='ij'), dim=-1)
    grid = s * 2 * grid + s / 2  # grid is [Hc/2, Wc/2, 2]
    assert desc_grid.shape[2] == prod(grid.shape[:2]), f"{desc_grid.shape[2]} and {prod(grid.shape[:2])} should be equal"
    grid = grid.reshape(1, 1, -1, 2)

    # Compute the expected location
    expect_coord = (grid * corr[..., None]).sum(dim=2)  # [B, n_kp, 2]

    # Compute the std of each keypoint matching
    var = (grid ** 2 * corr[..., None]).sum(dim=2) - expect_coord ** 2
    std = torch.sqrt(torch.clamp(var, min=eps)).sum(dim=-1)  # [B, n_kp]
    inv_std = 1. / torch.clamp(std, min=eps)
    inv_std = inv_std / inv_std.sum(dim=1, keepdim=True)  # it was a mean here in the original CAPS
    inv_std = inv_std.detach()

    # Return the loss, weighted by the std and ignoring invalid points
    loss = torch.norm(expect_coord - kp_w, dim=2)
    loss = ((loss * inv_std * valid_mask).sum(dim=1)
            / (valid_mask.sum(dim=1) + eps))
    return loss


def caps_window_loss(kp, kp_scores, descriptors, H, dense_desc_w,
                     temperature=50., std=True, ns=9, sd=8, s=8):
    """ CAPS loss with direct supervision, as described in 
        https://arxiv.org/pdf/2004.13324.pdf.
    Args:
        kp: [B, n_kp, 2] tensor of KP in image 0.
        kp_scores: [B, n_kp] tensor (0 = invalid) of KP scores in image 0.
        descriptors: [B, n_kp, D] tensor of descriptors in image 0.
        H: [B, 3, 3] homography from image 0 to 1.
        dense_desc_w: [B, D, Hc, Wc] tensor of dense descriptors in image 1.
        temperature: softmax temperature (higher means sharper expectations).
        std: whether to weight descriptors by their std or not.
        ns: the dense_desc will be interpolated at ns x ns locations.
        sd: each sample is separated by sd pixels.
            The total window size is thus sd * (ns - 1) = 64.
    """
    b_size, dim, Hc, Wc = dense_desc_w.size()
    h, w = (Hc * s, Wc * s)
    device = kp.device
    eps = 1e-8

    # Reproject keypoints to the other image and keep track of valid ones
    kp_w = warp_points(kp.float(), H)
    valid = ((kp_w[:, :, 0] >= 0) & (kp_w[:, :, 0] < h)
             & (kp_w[:, :, 1] >= 0) & (kp_w[:, :, 1] < w))
    kp_w[~valid] = 0
    valid_mask = valid.float() * (kp_scores > 0)  # shape [B, n_kp]

    # Extract a window around each kp_w
    grid = torch.stack(torch.meshgrid(
        torch.arange(ns, dtype=torch.float, device=device),
        torch.arange(ns, dtype=torch.float, device=device),
        indexing='ij'), dim=-1)  # grid is [ns, ns, 2]
    grid = (grid - ns // 2) * sd
    grid = grid[None, None] + kp_w[:, :, None, None]  # [B, n_kp, ns, ns, 2]
    # Randomize slightly the position of the grid
    grid = grid + torch.rand_like(kp_w)[:, :, None, None] * sd * 4 - 2 * sd
    grid = grid.reshape(b_size, -1, 2)
    desc_grid = F.grid_sample(
        dense_desc_w, keypoints_to_grid(grid, (h, w)))
    desc_grid = desc_grid.reshape(b_size, dim, -1, ns * ns)
    desc_grid = F.normalize(desc_grid, dim=1)
    # desc_grid is [B, D, n_kp, ns * ns]

    # Correlate the kp desc with the window of descriptors in the other image
    corr = torch.einsum('bnd,bdnm->bnm', descriptors, desc_grid)
    corr = torch.softmax(temperature * corr, dim=2)  # corr is [B, n_kp, ns * ns]

    # Compute the expected location
    grid = grid.reshape(b_size, -1, ns * ns, 2)
    expect_coord = (grid * corr[..., None]).sum(dim=2)  # [B, n_kp, 2]

    # Compute the std of each keypoint matching
    if std:
        var = (grid ** 2 * corr[..., None]).sum(dim=2) - expect_coord ** 2
        inv_std = 1. / torch.clamp(torch.sqrt(torch.clamp(
            var, min=eps)).sum(dim=-1), min=eps)  # [B, n_kp]
        inv_std = inv_std / inv_std.sum(dim=1, keepdim=True)  # it was a mean here in the original CAPS
        inv_std = inv_std.detach()
    else:
        inv_std = torch.ones_like(expect_coord[:, :, 0])

    # Return the loss, weighted by the std and ignoring invalid points
    loss = torch.norm(expect_coord - kp_w, dim=2)
    loss = ((loss * inv_std * valid_mask).sum(dim=1)
            / (valid_mask.sum(dim=1) + eps))
    return loss
