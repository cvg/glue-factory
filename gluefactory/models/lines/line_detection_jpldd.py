import numpy as np
import torch

from gluefactory.models.lines.line_utils import merge_lines


def get_valid_lines(
    lines,
    df,
    af,
    n_samples=10,
    df_thresh=5,
    inlier_thresh=0.7,
    r_ratio=0.1,
    a_diff_thresh=np.pi / 20,
    refinement_a_diff_thresh=np.pi / 60,
    a_std_thresh=np.pi / 30,
    a_inlier_thresh=0.5,
    r_radius=5,
    b_diff_thresh=5,
    min_len=5,
    merge=False,
    merge_thresh=4,
    check_sample=False,
):
    """
    lines: (N, 2, 2) -> each of the N elements: [[x1, x2], [y1, y2]]
    """

    sq_line_len = (lines[:, 0, 0] - lines[:, 0, 1]) ** 2 + (
        lines[:, 1, 0] - lines[:, 1, 1]
    ) ** 2
    validity = (lines[:, 0, 0] != lines[:, 0, 1]) & (lines[:, 1, 0] != lines[:, 1, 1])
    validity = validity & (sq_line_len >= (min_len**2))
    lines = lines[validity]

    offsets = torch.linspace(0, 1, n_samples).view(1, 1, -1).to(lines.device)

    xs = lines[:, 0, :1] + (lines[:, 0, 1:] - lines[:, 0, :1]) * offsets
    xs = torch.round(xs).long()
    ys = lines[:, 1, :1] + (lines[:, 1, 1:] - lines[:, 1, :1]) * offsets
    ys = torch.round(ys).long()

    interval_offsets = torch.round(
        torch.linspace(-r_radius, r_radius, 2 * r_radius + 1).view(1, 1, -1)
    ).to(lines.device)
    xint = xs.view(*xs.shape, 1) + interval_offsets
    xint = torch.clamp(xint, 0, df.shape[0] - 1)
    xint = torch.round(xint).long()
    yint = ys.view(*ys.shape, 1) + interval_offsets
    yint = torch.clamp(yint, 0, df.shape[1] - 1)
    yint = torch.round(yint).long()

    point_indices = df[xint, yint].argmin(axis=-1)
    ranges = [torch.arange(s).to(lines.device) for s in point_indices.shape]
    bs, rows, cols = torch.meshgrid(ranges, indexing="ij")

    point_indices = df[xint, yint].argmin(dim=-1)

    valid_xs = xint[bs, rows, cols, point_indices]
    valid_ys = yint[bs, rows, cols, point_indices]

    points = df[valid_xs, valid_ys]
    angles = af[valid_xs, valid_ys]

    slope = (lines[:, 0, 1] - lines[:, 0, 0]) / (
        lines[:, 1, 1] - lines[:, 1, 0] + 1e-10
    )
    direction = torch.remainder(torch.atan(slope), torch.pi)

    inlier_indices = points < df_thresh
    inlier_ratio = inlier_indices.sum(dim=-1).float() / inlier_indices.shape[-1]

    valid_angles = angles
    inlier_angle_indices = (
        torch.remainder(
            torch.abs(valid_angles[0] - direction.unsqueeze(-1)), np.pi
        ).unsqueeze(0)
        < a_diff_thresh
    )
    angle_inlier_ratio = (
        inlier_angle_indices.sum(dim=-1).float() / inlier_angle_indices.shape[-1]
    )

    crit1 = points.mean(dim=-1) < df_thresh
    crit2 = valid_angles.std(dim=-1) < a_std_thresh
    crit3 = inlier_ratio > inlier_thresh
    crit4 = (
        torch.remainder(torch.abs(valid_angles.mean(dim=-1) - direction), np.pi)
        < a_diff_thresh
    )
    crit5 = angle_inlier_ratio > a_inlier_thresh

    validity = crit1 & crit2 & crit3 & crit4 & crit5
    lines = lines.unsqueeze(0)[validity]

    if merge:
        new_lines = lines.mT
        merged_lines = merge_lines(
            new_lines, thresh=merge_thresh, overlap_thresh=0.0
        ).float()
        lines = merged_lines.mT

    return lines


def detect_jpldd_lines(
    df: np.array,
    af: np.array,
    keypoints: np.array,
    n_samples=10,
    df_thresh=2,
    inlier_thresh=0.7,
    r_ratio=0.1,
    a_diff_thresh=np.pi / 20,
    a_std_thresh=np.pi / 30,
    a_inlier_thresh=0.5,
    r_radius=5,
    b_diff_thresh=5,
    min_len=10,
    merge=False,
    merge_thresh=4,
):
    junctions = torch.zeros_like(keypoints).to(keypoints.device)
    junctions[:, 0], junctions[:, 1] = keypoints[:, 1].clone(), keypoints[:, 0].clone()
    lines = torch.hstack(
        [
            torch.cartesian_prod(junctions[:, 0], junctions[:, 0]),
            torch.cartesian_prod(junctions[:, 1], junctions[:, 1]),
        ]
    ).reshape((-1, 2, 2))

    prelim_valid_lines = get_valid_lines(
        lines,
        df,
        af,
        n_samples=10,
        df_thresh=df_thresh + 0.5,
        inlier_thresh=inlier_thresh / 2,
        a_diff_thresh=a_diff_thresh * 2,
        a_std_thresh=a_std_thresh * 3,
        a_inlier_thresh=a_inlier_thresh / 3,
        r_radius=0,
        min_len=min_len,
        merge=False,
    )
    valid_lines = get_valid_lines(
        prelim_valid_lines,
        df,
        af,
        n_samples=n_samples,
        df_thresh=df_thresh,
        inlier_thresh=inlier_thresh,
        r_ratio=r_ratio,
        a_diff_thresh=a_diff_thresh,
        a_std_thresh=a_std_thresh,
        a_inlier_thresh=a_inlier_thresh,
        r_radius=r_radius,
        b_diff_thresh=b_diff_thresh,
        min_len=min_len,
        merge=merge,
        merge_thresh=merge_thresh,
    )

    return valid_lines
