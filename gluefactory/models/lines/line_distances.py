import torch
from gluefactory.models.lines.line_utils import project_point_to_line, project_point_to_line_torch,intersect
import numpy as np

UPM_EPS = 1e-8

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

def get_structural_line_dist(warped_ref_line_seg, target_line_seg):
    """ Compute the distances between two sets of lines
        using the structural distance. """
    dist = (((warped_ref_line_seg[:, None, :, None]
              - target_line_seg[:, None]) ** 2).sum(-1)) ** 0.5
    dist = np.minimum(
        dist[:, :, 0, 0] + dist[:, :, 1, 1],
        dist[:, :, 0, 1] + dist[:, :, 1, 0]
    ) / 2
    return dist

def get_area_line_dist_asym(line_seg1, line_seg2, lbd=1/24):
    """ Compute an asymmetric line distance function which is not biased by
        the line length and is based on the area between segments.
        Here, line_seg2 are projected to the infinite line of line_seg1. """
    n1, n2 = len(line_seg1), len(line_seg2)

    # Determine which segments are intersecting each other
    all_line_seg1 = line_seg1[:, None].repeat(n2, axis=1).reshape(n1 * n2,
                                                                  2, 2)
    all_line_seg2 = line_seg2[None].repeat(n1, axis=0).reshape(n1 * n2, 2, 2)
    are_crossing = intersect(all_line_seg1, all_line_seg2)  # [n1 * n2]
    are_crossing = are_crossing.reshape(n1, n2)

    # Compute the orthogonal distance of the endpoints of line_seg2
    orth_dists2 = project_point_to_line(
        line_seg1, line_seg2.reshape(n2 * 2, 2))[1].reshape(n1, n2, 2)

    # Compute the angle between the line segments
    theta = angular_distance(line_seg1, line_seg2)  # [n1, n2]
    parallel = np.abs(theta) < 1e-8

    # Compute the orthogonal distance of the closest endpoint of line_seg2
    T = orth_dists2.min(axis=2)  # [n1, n2]

    # The distance for the intersecting lines is the area of two triangles,
    # divided by the length of line_seg2 squared:
    # area_dist = (d1^2+d2^2)/(2*tan(theta)*l^2)
    tan_theta = np.tan(theta)
    tan_theta[parallel] = 1
    length2 = np.linalg.norm(all_line_seg2[:, 0] - all_line_seg2[:, 1],
                             axis=1).reshape(n1, n2)
    area_dist = ((orth_dists2 ** 2).sum(axis=2)
                 / (2 * tan_theta * length2 ** 2) * (1. - parallel))

    # The distance for the non intersecting lines is lbd*T+1/4*sin(2*theta)
    non_int_area_dist = lbd * T + 1/4 * np.sin(2 * theta)
    area_dist[~are_crossing] = non_int_area_dist[~are_crossing]

    return area_dist

def get_area_line_dist(line_seg1, line_seg2, lbd=1/24):
    """ Compute a fairer line distance function which is not biased by
        the line length and is based on the area between segments. """
    area_dist_2_on_1 = get_area_line_dist_asym(line_seg1, line_seg2, lbd)
    area_dist_1_on_2 = get_area_line_dist_asym(line_seg2, line_seg1, lbd)
    area_dist = (area_dist_2_on_1 + area_dist_1_on_2.T) / 2
    return area_dist

def get_lip_line_dist_asym(line_seg1, line_seg2, default_len=30):
    """ Compute an asymmetrical length-invariant perpendicular distance. """
    n1, n2 = len(line_seg1), len(line_seg2)

    # Determine which segments are intersecting each other
    all_line_seg1 = line_seg1[:, None].repeat(n2, axis=1).reshape(n1 * n2,
                                                                  2, 2)
    all_line_seg2 = line_seg2[None].repeat(n1, axis=0).reshape(n1 * n2, 2, 2)
    are_crossing = intersect(all_line_seg1, all_line_seg2)  # [n1 * n2]
    are_crossing = are_crossing.reshape(n1, n2)

    # Compute the angle difference
    theta = angular_distance(line_seg1, line_seg2)  # [n1, n2]

    # Compute the orthogonal distance of the closest endpoint of line_seg2
    orth_dists2 = project_point_to_line(
        line_seg1, line_seg2.reshape(n2 * 2, 2))[1].reshape(n1, n2, 2)
    T = orth_dists2.min(axis=2)  # [n1, n2]
    
    # The distance is default_len * sin(theta) / 2 for intersecting lines
    # and T + default_len * sin(theta) / 2 for non intersecting ones
    # This means that a line crossing with theta=30deg is equivalent to a
    # parallel line with an offset T = default_len / 4
    lip_dist = default_len * np.sin(theta) / 2
    lip_dist[~are_crossing] += T[~are_crossing]
    return lip_dist

def get_lip_line_dist(line_seg1, line_seg2):
    """ Compute a length-invariant perpendicular distance. """
    lip_dist_2_on_1 = get_lip_line_dist_asym(line_seg1, line_seg2)
    lip_dist_1_on_2 = get_lip_line_dist_asym(line_seg2, line_seg1)
    lip_dist = (lip_dist_2_on_1 + lip_dist_1_on_2.T) / 2
    return lip_dist

def angular_distance(segs1, segs2):
    """ Compute the angular distance (via the cosine similarity)
        between two sets of line segments. """
    # Compute direction vector of segs1
    dirs1 = segs1[:, 1] - segs1[:, 0]
    dirs1 /= (np.linalg.norm(dirs1, axis=1, keepdims=True) + UPM_EPS)
    # Compute direction vector of segs2
    dirs2 = segs2[:, 1] - segs2[:, 0]
    dirs2 /= (np.linalg.norm(dirs2, axis=1, keepdims=True) + UPM_EPS)
    # https://en.wikipedia.org/wiki/Cosine_similarity
    return np.arccos(np.minimum(1, np.abs(np.einsum('ij,kj->ik', dirs1, dirs2))))