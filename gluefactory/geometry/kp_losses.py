import torch
import torch.nn.functional as F

from .homography_adaptation import warp_points
from ..models.extractors.superpoint import soft_argmax_refinement


def get_kp_cell(keypoints, k=8):
    """ Given a list of keypoints [N, 2] or [B, N, 2], retrieve their original
        k x k cell location and return the [(B, )N, 2] cell indices. """
    return torch.div(keypoints, k, rounding_mode='floor')[..., [1, 0]].long()


def get_pos_focal_loc_loss(scores, scores_w, loc, loc_w, keypoints,
                           valid_kp, H, gamma, max_dist=8, k=8):
    """ Loss enforcing that cells where a keypoint is detected should
        have a high score and the predicting kp location should match.
        Keypoints are here expected in xy convention with shape [B, N, 2]. """
    bs, h, w = scores.shape
    h, w = k * h, k * w
    n_kp = keypoints.shape[1]
    eps = 1e-8

    # Enforce a high score on cells in the first image
    pos_cells = get_kp_cell(keypoints, k)
    bs_grid = torch.arange(bs, device=scores.device)[:, None].expand(-1, n_kp)
    pos_scores = scores[bs_grid, pos_cells[:, :, 0], pos_cells[:, :, 1]]
    pos_loss = -(1 - pos_scores) ** gamma * torch.log(pos_scores + 1e-8)
    pos_loss = (pos_loss * valid_kp).sum(dim=1) / (valid_kp.sum(dim=1) + eps)

    # Reproject points to the other image and keep only shared points
    kp_w = warp_points(keypoints[:, :, [1, 0]].float(), H)[:, :, [1, 0]]
    valid = ((kp_w[:, :, 1] >= 0) & (kp_w[:, :, 1] < h)
             & (kp_w[:, :, 0] >= 0) & (kp_w[:, :, 0] < w))

    # Enforce a high score on the valid warped keypoints
    pos_cells_w = get_kp_cell(kp_w, k)
    pos_cells_w[~valid] = 0
    pos_scores_w = scores_w[bs_grid, pos_cells_w[:, :, 0],
                            pos_cells_w[:, :, 1]]
    pos_loss_w = -(1 - pos_scores_w) ** gamma * torch.log(pos_scores_w + 1e-8)
    valid = (valid & valid_kp).float()  # shape [B, n_kp]
    pos_loss_w = (pos_loss_w * valid).sum(dim=1) / (valid.sum(dim=1) + eps)
    pos_loss = (pos_loss + pos_loss_w) / 2

    # Location loss
    pred_kp = loc[bs_grid, pos_cells[:, :, 0], pos_cells[:, :, 1]]
    warped_pred_kp = warp_points(pred_kp[:, :, [1, 0]], H)[:, :, [1, 0]]
    pred_kp_w = loc_w[bs_grid, pos_cells_w[:, :, 0], pos_cells_w[:, :, 1]]
    dist = torch.norm(warped_pred_kp - pred_kp_w, dim=2)
    valid = valid * (dist < max_dist).float()  # Ignore faraway kp 
    loc_loss = (dist * valid).sum(dim=1) / (valid.sum(dim=1) + eps)
    return pos_loss, loc_loss


def soft_argmax_loss(scores, scores_w, keypoints, valid_kp,
                     H, gamma, radius, max_dist=8):
    bs, h, w = scores.shape
    n_kp = keypoints.shape[1]
    width = 2 * radius + 1
    eps = 1e-8
    keypoints_rd = torch.round(keypoints)[:, :, [1, 0]].long()

    # Enforce a high score on positive cells in the first image
    pos_scores = F.max_pool2d(scores.unsqueeze(1), width, 1, radius)[:, 0]
    pos_scores = pos_scores[
        torch.arange(bs, device=scores.device)[:, None].expand(-1, n_kp),
        keypoints_rd[:, :, 0], keypoints_rd[:, :, 1]]  # shape [bs, n_kp]
    pos_loss = -(1 - pos_scores) ** gamma * torch.log(pos_scores + 1e-8)
    pos_loss = (pos_loss * valid_kp).sum(dim=1) / (valid_kp.sum(dim=1) + eps)

    # Reproject points to the other image and keep only shared points
    kp_w = warp_points(keypoints[:, :, [1, 0]].float(), H)
    valid = ((kp_w[:, :, 0] >= 0) & (kp_w[:, :, 0] <= h - 1)
             & (kp_w[:, :, 1] >= 0) & (kp_w[:, :, 1] <= w - 1))
    keypoints_rd_w = torch.round(kp_w).long()
    keypoints_rd_w[~valid] = 0

    # Enforce a high score on the valid warped keypoints
    pos_scores_w = F.max_pool2d(scores_w.unsqueeze(1), width, 1, radius)[:, 0]
    pos_scores_w = pos_scores_w[
        torch.arange(bs, device=scores.device)[:, None].expand(-1, n_kp),
        keypoints_rd_w[:, :, 0], keypoints_rd_w[:, :, 1]]  # shape [bs, n_kp]
    pos_loss_w = -(1 - pos_scores_w) ** gamma * torch.log(pos_scores_w + 1e-8)
    valid = valid.float() * valid_kp  # shape [B, n_kp]
    pos_loss_w = (pos_loss_w * valid).sum(dim=1) / (valid.sum(dim=1) + eps)
    pos_loss = (pos_loss + pos_loss_w) / 2

    # Location loss
    pred_kp = torch.stack(soft_argmax_refinement(keypoints_rd, scores,
                                                 radius), dim=0)
    warped_pred_kp = warp_points(pred_kp, H)
    pred_kp_w = torch.stack(soft_argmax_refinement(keypoints_rd_w, scores_w,
                                                   radius), dim=0)
    dist = torch.norm(warped_pred_kp - pred_kp_w, dim=2)
    valid = valid * (dist < max_dist).float()  # Ignore faraway kp 
    loc_loss = (dist * valid).sum(dim=1) / (valid.sum(dim=1) + eps)
    return pos_loss, loc_loss


def soft_argmax_peaky_loss(scores, scores_w, keypoints, valid_kp,
                           H, radius, max_dist=8):
    def get_peaky_loss(scores, kp_rd, kp_ref, valid_kp, radius):
        eps = 1e-6
        width = 2 * radius + 1
        bs, h, w = scores.shape
        device = scores.device

        # Unfold the scores and sample them at the keypoints
        unfolded_scores = F.unfold(scores[:, None], (width, width),
                                   padding=radius)
        # unfolded_scores is [B, width*width, H*W]
        unfolded_scores = torch.softmax(unfolded_scores, dim=1)
        unfolded_scores = unfolded_scores.permute(0, 2, 1).reshape(
            bs, h, w, width * width)
        unfolded_scores = unfolded_scores[
            torch.arange(bs, device=device)[:, None].expand(-1, n_kp),
            kp_rd[:, :, 0], kp_rd[:, :, 1]].reshape(bs, n_kp, width * width)

        # Unfold the pixel coordinates
        grid = torch.stack(torch.meshgrid(
            [torch.arange(h, dtype=torch.float, device=device),
             torch.arange(w, dtype=torch.float, device=device)],
            indexing='ij'), dim=0)[None]  # [1, 2, H, W]
        unfolded_grid = F.unfold(grid, (width, width), padding=radius)[0]
        # unfolded_grid is [2*width*width, H*W]
        unfolded_grid = unfolded_grid.T.reshape(h, w, 2 * width * width)

        # Compute the distance to the refined keypoints
        pixel_coords = unfolded_grid[
            kp_rd[:, :, 0], kp_rd[:, :, 1]].reshape(bs, n_kp, 2, width*width)
        distances = torch.norm(pixel_coords - kp_ref.unsqueeze(-1), dim=2)

        # Compute the peaky loss
        peaky_loss = (distances * unfolded_scores).sum(dim=2)
        peaky_loss = ((peaky_loss * valid_kp).sum(dim=1)
                      / (valid_kp.sum(dim=1) + eps))
        return peaky_loss

    bs, h, w = scores.shape
    n_kp = keypoints.shape[1]
    width = 2 * radius + 1
    eps = 1e-6
    keypoints_rd = torch.round(keypoints)[:, :, [1, 0]].long()

    # Reproject points to the other image and keep only shared points
    kp_w = warp_points(keypoints[:, :, [1, 0]].float(), H)
    valid = ((kp_w[:, :, 0] >= 0) & (kp_w[:, :, 0] <= h - 1)
             & (kp_w[:, :, 1] >= 0) & (kp_w[:, :, 1] <= w - 1))
    keypoints_rd_w = torch.round(kp_w).long()
    keypoints_rd_w[~valid] = 0
    
    # Peaky loss in the first image
    pred_kp = torch.stack(soft_argmax_refinement(keypoints_rd, scores,
                                                 radius), dim=0)
    peaky_loss = get_peaky_loss(scores, keypoints_rd, pred_kp,
                                valid_kp, radius)
    
    # Peaky loss in the second image
    pred_kp_w = torch.stack(soft_argmax_refinement(keypoints_rd_w, scores_w,
                                                   radius), dim=0)
    peaky_loss_w = get_peaky_loss(scores_w, keypoints_rd_w, pred_kp_w,
                                  valid_kp * valid.float(), radius)
    peaky_loss = (peaky_loss + peaky_loss_w) / 2

    # Location loss
    warped_pred_kp = warp_points(pred_kp, H)
    dist = torch.norm(warped_pred_kp - pred_kp_w, dim=2)
    valid = valid_kp * valid.float() * (dist < max_dist).float()  # Ignore faraway kp
    loc_loss = (dist * valid).sum(dim=1) / (valid.sum(dim=1) + eps)

    return peaky_loss, loc_loss


def alike_loss(scores, scores_w, pred_kp, pred_kp_w, sp_kp,
               valid_kp, H, gamma, radius, max_dist=8):
    bs, h, w = scores.shape
    n_kp = pred_kp.shape[1]
    width = 2 * radius + 1
    eps = 1e-8
    keypoints_rd = torch.round(sp_kp)[:, :, [1, 0]].long()

    # Enforce a high score on positive cells in the first image
    pos_scores = F.max_pool2d(scores.unsqueeze(1), width, 1, radius)[:, 0]
    pos_scores = pos_scores[
        torch.arange(bs, device=scores.device)[:, None].expand(-1, n_kp),
        keypoints_rd[:, :, 0], keypoints_rd[:, :, 1]]  # shape [bs, n_kp]
    pos_loss = -(1 - pos_scores) ** gamma * torch.log(pos_scores + 1e-8)
    pos_loss = (pos_loss * valid_kp).sum(dim=1) / (valid_kp.sum(dim=1) + eps)

    # Reproject points to the other image and keep only shared points
    kp_w = warp_points(sp_kp[:, :, [1, 0]].float(), H)
    valid = ((kp_w[:, :, 0] >= 0) & (kp_w[:, :, 0] <= h - 1)
             & (kp_w[:, :, 1] >= 0) & (kp_w[:, :, 1] <= w - 1))
    keypoints_rd_w = torch.round(kp_w).long()
    keypoints_rd_w[~valid] = 0

    # Enforce a high score on the valid warped keypoints
    pos_scores_w = F.max_pool2d(scores_w.unsqueeze(1), width, 1, radius)[:, 0]
    pos_scores_w = pos_scores_w[
        torch.arange(bs, device=scores.device)[:, None].expand(-1, n_kp),
        keypoints_rd_w[:, :, 0], keypoints_rd_w[:, :, 1]]  # shape [bs, n_kp]
    pos_loss_w = -(1 - pos_scores_w) ** gamma * torch.log(pos_scores_w + 1e-8)
    valid = valid.float() * valid_kp  # shape [B, n_kp]
    pos_loss_w = (pos_loss_w * valid).sum(dim=1) / (valid.sum(dim=1) + eps)
    pos_loss = (pos_loss + pos_loss_w) / 2

    # Location loss
    warped_pred_kp = warp_points(pred_kp[:, :, [1, 0]], H)[:, :, [1, 0]]
    warped_pred_kp_w = warp_points(pred_kp_w[:, :, [1, 0]],
                                   torch.inverse(H))[:, :, [1, 0]]
    dist_w = torch.norm(warped_pred_kp[:, :, None] - pred_kp_w[:, None],
                        dim=3).min(dim=1)[0]
    dist = torch.norm(warped_pred_kp_w[:, None] - pred_kp[:, :, None],
                      dim=3).min(dim=2)[0]
    valid = (dist < max_dist).float()  # Ignore faraway kp
    valid_w = (dist_w < max_dist).float()  # Ignore faraway kp
    loc_loss = (dist * valid).sum(dim=1) / (valid.sum(dim=1) + eps)
    loc_loss_w = (dist_w * valid_w).sum(dim=1) / (valid_w.sum(dim=1) + eps)
    loc_loss = (loc_loss + loc_loss_w) / 2

    return pos_loss, loc_loss


def sargmax_ms_loss(heatmap, heatmap_w, scores, scores_w, keypoints, valid_kp,
                    H, gamma, radius, max_dist=8, k=8, with_loc=True):
    bs, h, w = heatmap.shape
    n_kp = keypoints.shape[1]
    eps = 1e-8

    # Enforce a high score on cells in the first image
    pos_cells = get_kp_cell(keypoints, k)
    bs_grid = torch.arange(bs, device=scores.device)[:, None].expand(-1, n_kp)
    pos_scores = scores[bs_grid, pos_cells[:, :, 0], pos_cells[:, :, 1]]
    pos_loss = -(1 - pos_scores) ** gamma * torch.log(pos_scores + 1e-8)
    pos_loss = (pos_loss * valid_kp).sum(dim=1) / (valid_kp.sum(dim=1) + eps)

    # Reproject points to the other image and keep only shared points
    kp_w = warp_points(keypoints[:, :, [1, 0]].float(), H)[:, :, [1, 0]]
    valid = ((kp_w[:, :, 1] >= 0) & (kp_w[:, :, 1] <= h - 1)
             & (kp_w[:, :, 0] >= 0) & (kp_w[:, :, 0] <= w - 1))

    # Enforce a high score on the valid warped keypoints
    pos_cells_w = get_kp_cell(kp_w, k)
    pos_cells_w[~valid] = 0
    pos_scores_w = scores_w[bs_grid, pos_cells_w[:, :, 0],
                            pos_cells_w[:, :, 1]]
    pos_loss_w = -(1 - pos_scores_w) ** gamma * torch.log(pos_scores_w + 1e-8)
    valid = (valid & valid_kp).float()  # shape [B, n_kp]
    pos_loss_w = (pos_loss_w * valid).sum(dim=1) / (valid.sum(dim=1) + eps)
    pos_loss = (pos_loss + pos_loss_w) / 2

    # Location loss
    if with_loc:
        keypoints_rd = torch.round(keypoints)[:, :, [1, 0]].long()
        keypoints_rd_w = torch.round(kp_w)[:, :, [1, 0]].long()
        keypoints_rd_w[~valid.bool()] = 0
        pred_kp = torch.stack(soft_argmax_refinement(keypoints_rd, heatmap,
                                                     radius), dim=0)
        warped_pred_kp = warp_points(pred_kp, H)
        pred_kp_w = torch.stack(soft_argmax_refinement(
            keypoints_rd_w, heatmap_w, radius), dim=0)
        dist = torch.norm(warped_pred_kp - pred_kp_w, dim=2)
        valid = valid * (dist < max_dist).float()  # Ignore faraway kp 
        loc_loss = (dist * valid).sum(dim=1) / (valid.sum(dim=1) + eps)
    else:
        loc_loss = 0
    return pos_loss, loc_loss


def kp_ce_loss(logits, gt_kp, valid_kp, valid_mask=None):
    """ Cross entropy loss for keypoints, as in SuperPoint. """
    bs, _, h, w = logits.shape
    h, w = h * 8, w * 8
    device = logits.device

    # Convert the GT keypoints to a heatmap
    gt_kp_int = torch.clamp(torch.round(gt_kp).long(),
                            torch.tensor([0, 0], device=device),
                            torch.tensor([w - 1, h - 1], device=device))
    gt_scores = torch.zeros(bs, h, w, dtype=torch.float, device=device)
    for b in range(bs):
        valid_kp_int = gt_kp_int[b, valid_kp[b]]
        gt_scores[b, valid_kp_int[:, 1], valid_kp_int[:, 0]] = 1

    # Extract the GT class from the GT heatmap
    gt_scores = F.pixel_unshuffle(gt_scores[:, None], 8)
    gt_scores = torch.cat([2 * gt_scores, torch.ones_like(gt_scores[:, :1])],
                          dim=1)  # gt_scores is now [B, 65, H, W]
    gt_labels = torch.argmax(gt_scores + torch.rand_like(gt_scores) * 0.1,
                             dim=1)  # breaks ties for kp in the same cell
    # gt_labels is [B, H, W]

    # Perform cross entropy
    loss = F.cross_entropy(logits, gt_labels, reduction='none')
    
    if valid_mask is None:
        valid_mask = torch.ones_like(loss)
    else:
        valid_mask = -F.max_pool2d(-valid_mask, 8)
    return (loss * valid_mask).sum(dim=[1, 2]) / valid_mask.sum(dim=[1, 2])


def kp_ce_loss_indep(logits, gt_kp, valid_kp, valid_mask=None):
    """ Cross entropy loss for keypoints, with independent coordinates. """
    bs, _, h, w = logits.shape
    h, w = h * 8, w * 8
    device = logits.device

    # Split the logits into row and col coordinates
    logits_h = torch.cat([logits[:, :8], logits[:, 16:]], dim=1)
    logits_w = torch.cat([logits[:, 8:16], logits[:, 16:]], dim=1)

    # Convert the GT keypoints to a heatmap
    gt_kp_int = torch.clamp(torch.round(gt_kp).long(),
                            torch.tensor([0, 0], device=device),
                            torch.tensor([w - 1, h - 1], device=device))
    gt_scores = torch.zeros(bs, h, w, dtype=torch.float, device=device)
    for b in range(bs):
        valid_kp_int = gt_kp_int[b, valid_kp[b]]
        gt_scores[b, valid_kp_int[:, 1], valid_kp_int[:, 0]] = 1

    # Extract the GT class from the GT heatmap
    gt_scores = F.pixel_unshuffle(gt_scores[:, None], 8)
    gt_scores = torch.cat([2 * gt_scores, torch.ones_like(gt_scores[:, :1])],
                          dim=1)  # gt_scores is now [B, 65, H, W]
    gt_labels = torch.argmax(gt_scores + torch.rand_like(gt_scores) * 0.1,
                             dim=1)  # breaks ties for kp in the same cell
    # gt_labels is [B, H, W]
    gt_labels_h = gt_labels // 8
    gt_labels_w = torch.remainder(gt_labels, 8)
    gt_labels_w[gt_labels == 64] = 8

    # Perform cross entropy
    loss_h = F.cross_entropy(logits_h, gt_labels_h, reduction='none')
    loss_w = F.cross_entropy(logits_w, gt_labels_w, reduction='none')
    loss = (loss_h + loss_w) / 2.
    if valid_mask is None:
        valid_mask = torch.ones_like(loss)
    else:
        valid_mask = -F.max_pool2d(-valid_mask, 8)
    return (loss * valid_mask).sum(dim=[1, 2]) / valid_mask.sum(dim=[1, 2])


def soft_argmax_only_loss(scores, scores_w, keypoints, valid_kp,
                          H, radius, max_dist=8):
    _, h, w = scores.shape
    eps = 1e-8
    keypoints_rd = torch.round(keypoints)[:, :, [1, 0]].long()

    # Reproject points to the other image and keep only shared points
    kp_w = warp_points(keypoints[:, :, [1, 0]].float(), H)
    valid = ((kp_w[:, :, 0] >= 0) & (kp_w[:, :, 0] <= h - 1)
             & (kp_w[:, :, 1] >= 0) & (kp_w[:, :, 1] <= w - 1))
    keypoints_rd_w = torch.round(kp_w).long()
    keypoints_rd_w[~valid] = 0
    valid = valid.float() * valid_kp  # shape [B, n_kp]

    # Location loss
    pred_kp = torch.stack(soft_argmax_refinement(keypoints_rd, scores,
                                                 radius), dim=0)
    warped_pred_kp = warp_points(pred_kp, H)
    pred_kp_w = torch.stack(soft_argmax_refinement(keypoints_rd_w, scores_w,
                                                   radius), dim=0)
    dist = torch.norm(warped_pred_kp - pred_kp_w, dim=2)
    valid = valid * (dist < max_dist).float()  # Ignore faraway kp 
    loc_loss = (dist * valid).sum(dim=1) / (valid.sum(dim=1) + eps)
    return loc_loss


def kp_bce_loss(logits, gt_kp, valid_kp, valid_mask=None):
    """ Binary Cross entropy loss for keypoints """
    bs, h, w = logits.shape
    device = logits.device

    # Convert the GT keypoints to a heatmap
    gt_kp_int = torch.clamp(torch.round(gt_kp).long(),
                            torch.tensor([0, 0], device=device),
                            torch.tensor([w - 1, h - 1], device=device))
    gt_scores = torch.zeros(bs, h, w, dtype=torch.float, device=device)
    for b in range(bs):
        valid_kp_int = gt_kp_int[b, valid_kp[b]]
        gt_scores[b, valid_kp_int[:, 1], valid_kp_int[:, 0]] = 1

    loss = F.binary_cross_entropy(logits, gt_scores, reduction='none')
    if valid_mask is None:
        valid_mask = torch.ones_like(loss)
    else:
        valid_mask = -F.max_pool2d(-valid_mask, 8)

    return (loss * valid_mask).sum(dim=[1, 2]) / valid_mask.sum(dim=[1, 2])
