import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from gluefactory.utils.warp import warp_points


def get_structural_line_dist(warped_ref_line_seg, target_line_seg):
    """Compute the distances between two sets of lines
    using the structural distance."""
    dist = (
        ((warped_ref_line_seg[:, None, :, None] - target_line_seg[:, None]) ** 2).sum(
            -1
        )
    ) ** 0.5
    dist = (
        np.minimum(
            dist[:, :, 0, 0] + dist[:, :, 1, 1], dist[:, :, 0, 1] + dist[:, :, 1, 0]
        )
        / 2
    )
    # print(dist.shape)
    # select = np.eye(len(dist))
    dist = dist.diagonal()
    return dist.numpy()

def get_structural_dist(warped_ref_line_seg, target_line_seg):
    """Compute the distances between two sets of lines
    using the structural distance."""
    dist = (
        ((warped_ref_line_seg[:, None, :, None] - target_line_seg[:, None]) ** 2).sum(
            -1
        )
    ) ** 0.5
    dist = (
        np.minimum(
            dist[:, :, 0, 0] + dist[:, :, 1, 1], dist[:, :, 0, 1] + dist[:, :, 1, 0]
        )
        / 2
    )
    
    return dist.numpy()


def angular_distance(segs1, segs2):
    """Compute the angular distance (via the cosine similarity)
    between two sets of line segments."""
    # Compute direction vector of segs1
    dirs1 = segs1[:, 1] - segs1[:, 0]
    dirs1 /= np.linalg.norm(dirs1, axis=1, keepdims=True) + 1e-8
    # Compute direction vector of segs2
    dirs2 = segs2[:, 1] - segs2[:, 0]
    dirs2 /= np.linalg.norm(dirs2, axis=1, keepdims=True) + 1e-8
    # https://en.wikipedia.org/wiki/Cosine_similarity
    return np.arccos(np.minimum(1, np.abs(np.einsum("ij,kj->ik", dirs1, dirs2))))


def orientation(p, q, r):
    """Compute the orientation of a list of triplets of points."""
    return np.sign(
        (q[:, 1] - p[:, 1]) * (r[:, 0] - q[:, 0])
        - (q[:, 0] - p[:, 0]) * (r[:, 1] - q[:, 1])
    )


def is_on_segment(line_seg, p):
    """Check whether a point p is on a line segment, assuming the point
    to be colinear with the two endpoints."""
    return (
        (p[:, 0] >= np.min(line_seg[:, :, 0], axis=1))
        & (p[:, 0] <= np.max(line_seg[:, :, 0], axis=1))
        & (p[:, 1] >= np.min(line_seg[:, :, 1], axis=1))
        & (p[:, 1] <= np.max(line_seg[:, :, 1], axis=1))
    )


def intersect(line_seg1, line_seg2):
    """Check whether two sets of lines segments
    intersects with each other."""
    ori1 = orientation(line_seg1[:, 0], line_seg1[:, 1], line_seg2[:, 0])
    ori2 = orientation(line_seg1[:, 0], line_seg1[:, 1], line_seg2[:, 1])
    ori3 = orientation(line_seg2[:, 0], line_seg2[:, 1], line_seg1[:, 0])
    ori4 = orientation(line_seg2[:, 0], line_seg2[:, 1], line_seg1[:, 1])
    return (
        ((ori1 != ori2) & (ori3 != ori4))
        | ((ori1 == 0) & is_on_segment(line_seg1, line_seg2[:, 0]))
        | ((ori2 == 0) & is_on_segment(line_seg1, line_seg2[:, 1]))
        | ((ori3 == 0) & is_on_segment(line_seg2, line_seg1[:, 0]))
        | ((ori4 == 0) & is_on_segment(line_seg2, line_seg1[:, 1]))
    )


def project_point_to_line(line_segs, points):
    """Given a list of line segments and a list of points (2D or 3D coordinates),
    compute the orthogonal projection of all points on all lines.
    This returns the 1D coordinates of the projection on the line,
    as well as the list of orthogonal distances."""
    # Compute the 1D coordinate of the points projected on the line
    dir_vec = (line_segs[:, 1] - line_segs[:, 0])[:, None]
    coords1d = ((points[None] - line_segs[:, None, 0]) * dir_vec).sum(
        axis=2
    ) / np.linalg.norm(dir_vec, axis=2) ** 2
    # coords1d is of shape (n_lines, n_points)

    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    dist_to_line = np.linalg.norm(projection - points[None], axis=2)

    return coords1d, dist_to_line


def get_segment_overlap(seg_coord1d):
    """Given a list of segments parameterized by the 1D coordinate
    of the endpoints, compute the overlap with the segment [0, 1]."""
    seg_coord1d = np.sort(seg_coord1d, axis=-1)
    overlap = (
        (seg_coord1d[..., 1] > 0)
        * (seg_coord1d[..., 0] < 1)
        * (np.minimum(seg_coord1d[..., 1], 1) - np.maximum(seg_coord1d[..., 0], 0))
    )
    return overlap


def get_area_line_dist_asym(line_seg1, line_seg2, lbd=1 / 24):
    """Compute an asymmetric line distance function which is not biased by
    the line length and is based on the area between segments.
    Here, line_seg2 are projected to the infinite line of line_seg1."""
    n1, n2 = len(line_seg1), len(line_seg2)

    # Determine which segments are intersecting each other
    all_line_seg1 = line_seg1[:, None].repeat(n2, axis=1).reshape(n1 * n2, 2, 2)
    all_line_seg2 = line_seg2[None].repeat(n1, axis=0).reshape(n1 * n2, 2, 2)
    are_crossing = intersect(all_line_seg1, all_line_seg2)  # [n1 * n2]
    are_crossing = are_crossing.reshape(n1, n2)

    # Compute the orthogonal distance of the endpoints of line_seg2
    orth_dists2 = project_point_to_line(line_seg1, line_seg2.reshape(n2 * 2, 2))[
        1
    ].reshape(n1, n2, 2)

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
    length2 = np.linalg.norm(all_line_seg2[:, 0] - all_line_seg2[:, 1], axis=1).reshape(
        n1, n2
    )
    area_dist = (
        (orth_dists2**2).sum(axis=2) / (2 * tan_theta * length2**2) * (1.0 - parallel)
    )

    # The distance for the non intersecting lines is lbd*T+1/4*sin(2*theta)
    non_int_area_dist = lbd * T + 1 / 4 * np.sin(2 * theta)
    area_dist[~are_crossing] = non_int_area_dist[~are_crossing]

    return area_dist


def get_area_line_dist(line_seg1, line_seg2, lbd=1 / 24):
    """Compute a fairer line distance function which is not biased by
    the line length and is based on the area between segments."""
    area_dist_2_on_1 = get_area_line_dist_asym(line_seg1, line_seg2, lbd)
    area_dist_1_on_2 = get_area_line_dist_asym(line_seg2, line_seg1, lbd)
    area_dist = (area_dist_2_on_1 + area_dist_1_on_2.T) / 2
    return area_dist


def get_orth_line_dist(
    line_seg1, line_seg2, min_overlap=0.5, return_overlap=False, mode="min"
):
    """Compute the symmetrical orthogonal line distance between two sets
    of lines and the average overlapping ratio of both lines.
    Enforce a high line distance for small overlaps.
    This is compatible for nD objects (e.g. both lines in 2D or 3D)."""
    return get_orth_dist(line_seg1, line_seg2, min_overlap, return_overlap, mode).diagonal()


def get_orth_dist(
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
    min_overlaps = np.minimum(overlaps1, overlaps2)

    if return_overlap:
        return line_dists, overlaps

    # Enforce a max line distance for line segments with small overlap
    if mode == "mean":
        low_overlaps = overlaps < min_overlap
    else:
        low_overlaps = min_overlaps < min_overlap
    line_dists[low_overlaps] = np.amax(line_dists)
    return line_dists


def match_segments_to_distance(
    line_seg1, line_seg2, H_0to1, line_dist="orth", overlap_th=0.5
):

    line_seg1 = torch.from_numpy(
        warp_points(line_seg1.reshape(-1, 2)[:, [1, 0]].numpy(), H_0to1.numpy())[
            :, [1, 0]
        ].reshape(-1, 2, 2)
    )

    if line_dist == "struct":
        distances = get_structural_line_dist(line_seg1, line_seg2)
    elif line_dist == "orth":
        distances = get_orth_line_dist(line_seg1, line_seg2, overlap_th)
    else:
        raise ValueError("Unknown line distance: " + line_dist)

    # return segs1, segs2, matched_idx1, matched_idx2, distances
    # print(distances.mean())
    # print(distances.min())
    # print(distances.max())
    # print(distances.std())
    # print()
    return distances


### Metrics computation


def compute_repeatability(segs1, segs2, distances, thresholds):
    """Compute the repeatability between two sets of matched lines.
    Args:
        segs1, segs2: the original sets of lines.
        matched_idx1, matched_idx2: the indices of the matches.
        distances: the line distance of the matches.
        thresholds: correctness thresholds. Can be an int or a list.
        rep_type: 'num' will compute the ratio of repeatable lines and
                  'length' will compute the ratio of matched lengths of lines.
    Returns:
        The line repeatability, given for each threshold.
    """
    if isinstance(thresholds, int):
        thresholds = [thresholds]

    n1, n2 = len(segs1), len(segs2)
    if n1 == 0 or n2 == 0:
        return [0] * len(thresholds)

    reps = []
    for t in thresholds:
        correct = distances <= t
        rep = np.sum(correct) / min(n1, n2)
        reps.append(rep)
    return reps


def compute_loc_error(distances, thresholds):
    """Compute the line localization error between two sets of lines.
    Args:
        distances: the line distance of the matches, in increasing order.
        thresholds: int or list of number of lines to take into account.
    Returns:
        The line localization error, given for number of lines.
    """
    if isinstance(thresholds, int):
        thresholds = [thresholds]

    loc_errors = []
    for t in thresholds:
        valid_distances = distances < t
        valid_distances = distances[valid_distances]
        if len(valid_distances) == 0:
            loc_errors.append(0)
        else:
            loc_errors.append(np.mean(valid_distances))
    return loc_errors
