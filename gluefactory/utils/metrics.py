import torch


@torch.no_grad()
def matcher_metrics(pred, data, prefix="", prefix_gt=None):
    def recall(m, gt_m):
        mask = (gt_m > -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def accuracy(m, gt_m):
        mask = (gt_m >= -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def precision(m, gt_m):
        mask = ((m > -1) & (gt_m >= -1)).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def ranking_ap(m, gt_m, scores):
        p_mask = ((m > -1) & (gt_m >= -1)).float()
        r_mask = (gt_m > -1).float()
        sort_ind = torch.argsort(-scores)
        sorted_p_mask = torch.gather(p_mask, -1, sort_ind)
        sorted_r_mask = torch.gather(r_mask, -1, sort_ind)
        sorted_tp = torch.gather(m == gt_m, -1, sort_ind)
        p_pts = torch.cumsum(sorted_tp * sorted_p_mask, -1) / (
            1e-8 + torch.cumsum(sorted_p_mask, -1)
        )
        r_pts = torch.cumsum(sorted_tp * sorted_r_mask, -1) / (
            1e-8 + sorted_r_mask.sum(-1)[:, None]
        )
        r_pts_diff = r_pts[..., 1:] - r_pts[..., :-1]
        return torch.sum(r_pts_diff * p_pts[:, None, -1], dim=-1)

    if prefix_gt is None:
        prefix_gt = prefix
    rec = recall(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    prec = precision(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    acc = accuracy(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    ap = ranking_ap(
        pred[f"{prefix}matches0"],
        data[f"gt_{prefix_gt}matches0"],
        pred[f"{prefix}matching_scores0"],
    )
    metrics = {
        f"{prefix}match_recall": rec,
        f"{prefix}match_precision": prec,
        f"{prefix}accuracy": acc,
        f"{prefix}average_precision": ap,
    }
    return metrics
