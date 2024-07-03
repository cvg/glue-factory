import torch
from line_utils import project_point_to_line, project_point_to_line_torch


def get_segment_overlap(seg_coord1d):
    """Given a list of segments parameterized by the 1D coordinate
    of the endpoints, compute the overlap with the segment [0, 1]."""
    seg_coord1d, _ = torch.sort(seg_coord1d, dim=-1)
    overlap = (
        (seg_coord1d[..., 1] > 0)
        * (seg_coord1d[..., 0] < 1)
        * (
            torch.minimum(seg_coord1d[..., 1], torch.tensor(1))
            - torch.maximum(seg_coord1d[..., 0], torch.tensor(0))
        )
    )

    return overlap


def get_segment_overlap_torch(seg_coord1d):
    """Given a list of segments as torch tensors parameterized by the 1D coordinate
    of the endpoints, compute the overlap with the segment [0, 1]."""
    seg_coord1d, _ = torch.sort(seg_coord1d, dim=-1)
    overlap = (
        (seg_coord1d[..., 1] > 0)
        * (seg_coord1d[..., 0] < 1)
        * (
            torch.minimum(seg_coord1d[..., 1], torch.tensor(1))
            - torch.maximum(seg_coord1d[..., 0], torch.tensor(0))
        )
    )

    return overlap


def get_orth_line_dist(
    line_seg1, line_seg2, min_overlap=0.5, return_overlap=False, mode="min"
):
    """Compute the symmetrical orthogonal line distance between two sets
    of lines and the average overlapping ratio of both lines.
    Enforce a high line distance for small overlaps.
    This is compatible for nD objects (e.g. both lines in 2D or 3D)."""
    n_lines1, n_lines2 = len(line_seg1), len(line_seg2)

    # Compute the average orthogonal line distance
    coords_2_on_1, line_dists2 = project_point_to_line(
        line_seg1, line_seg2.reshape(n_lines2 * 2, -1)
    )
    line_dists2 = line_dists2.reshape(n_lines1, n_lines2, 2).sum(axis=2)
    coords_1_on_2, line_dists1 = project_point_to_line(
        line_seg2, line_seg1.reshape(n_lines1 * 2, -1)
    )
    line_dists1 = line_dists1.reshape(n_lines2, n_lines1, 2).sum(axis=2)
    line_dists = (line_dists2 + line_dists1.T) / 2

    # Compute the average overlapping ratio
    coords_2_on_1 = coords_2_on_1.reshape(n_lines1, n_lines2, 2)
    overlaps1 = get_segment_overlap(coords_2_on_1)
    coords_1_on_2 = coords_1_on_2.reshape(n_lines2, n_lines1, 2)
    overlaps2 = get_segment_overlap(coords_1_on_2).T
    overlaps = (overlaps1 + overlaps2) / 2
    min_overlaps = torch.minimum(overlaps1, overlaps2)

    if return_overlap:
        return line_dists, overlaps

    # Enforce a max line distance for line segments with small overlap
    if mode == "mean":
        low_overlaps = overlaps < min_overlap
    else:
        low_overlaps = min_overlaps < min_overlap
    line_dists[low_overlaps] = torch.amax(line_dists)
    return line_dists


def get_orth_line_dist_torch(
    line_seg1, line_seg2, min_overlap=0.5, return_overlap=False, mode="min"
):
    """Compute the symmetrical orthogonal line distance between two sets
    of lines (as torch tensors) and the average overlapping ratio of both lines.
    Enforce a high line distance for small overlaps.
    This is compatible for nD objects (e.g. both lines in 2D or 3D)."""
    n_lines1, n_lines2 = len(line_seg1), len(line_seg2)

    # Compute the average orthogonal line distance
    coords_2_on_1, line_dists2 = project_point_to_line_torch(
        line_seg1, line_seg2.reshape(n_lines2 * 2, -1)
    )
    line_dists2 = line_dists2.reshape(n_lines1, n_lines2, 2).sum(axis=2)
    coords_1_on_2, line_dists1 = project_point_to_line_torch(
        line_seg2, line_seg1.reshape(n_lines1 * 2, -1)
    )
    line_dists1 = line_dists1.reshape(n_lines2, n_lines1, 2).sum(axis=2)
    line_dists = (line_dists2 + line_dists1.T) / 2

    # Compute the average overlapping ratio
    coords_2_on_1 = coords_2_on_1.reshape(n_lines1, n_lines2, 2)
    overlaps1 = get_segment_overlap_torch(coords_2_on_1)
    coords_1_on_2 = coords_1_on_2.reshape(n_lines2, n_lines1, 2)
    overlaps2 = get_segment_overlap_torch(coords_1_on_2).T
    overlaps = (overlaps1 + overlaps2) / 2
    min_overlaps = torch.minimum(overlaps1, overlaps2)

    if return_overlap:
        return line_dists, overlaps

    # Enforce a max line distance for line segments with small overlap
    if mode == "mean":
        low_overlaps = overlaps < min_overlap
    else:
        low_overlaps = min_overlaps < min_overlap
    line_dists[low_overlaps] = torch.amax(line_dists)
    return line_dists
