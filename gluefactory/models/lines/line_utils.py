"""
    A set of geometry tools to handle lines in Pytorch and Numpy.
"""

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

UPM_EPS = 1e-8


def nn_interpolate_numpy(img, x, y):
    xi = np.clip(np.round(x).astype(int), 0, img.shape[1] - 1)
    yi = np.clip(np.round(y).astype(int), 0, img.shape[0] - 1)
    return img[yi, xi]


def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T

### Line segment distances

def project_point_to_line(line_segs, points):
    """ Given a list of line segments and a list of points (2D or 3D coordinates),
        compute the orthogonal projection of all points on all lines.
        This returns the 1D coordinates of the projection on the line,
        as well as the list of orthogonal distances. """
    # Compute the 1D coordinate of the points projected on the line
    dir_vec = (line_segs[:, 1] - line_segs[:, 0])[:, None]
    coords1d = (torch.sum((points[None] - line_segs[:, None, 0]) * dir_vec,dim=2)
                / torch.norm(dir_vec.float(), dim=2) ** 2)
    # coords1d is of shape (n_lines, n_points)

    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    dist_to_line = torch.norm(projection - points[None], dim=2)

    return coords1d, dist_to_line


def get_segment_overlap(seg_coord1d):
    """ Given a list of segments parameterized by the 1D coordinate
        of the endpoints, compute the overlap with the segment [0, 1]. """
    seg_coord1d,_ = torch.sort(seg_coord1d, dim=-1)
    overlap = ((seg_coord1d[..., 1] > 0) * (seg_coord1d[..., 0] < 1)
               * (torch.minimum(seg_coord1d[..., 1], torch.tensor(1))
                  - torch.maximum(seg_coord1d[..., 0], torch.tensor(0))))

    return overlap


def get_orth_line_dist(line_seg1, line_seg2, min_overlap=0.5,
                       return_overlap=False, mode='min'):
    """ Compute the symmetrical orthogonal line distance between two sets
        of lines and the average overlapping ratio of both lines.
        Enforce a high line distance for small overlaps.
        This is compatible for nD objects (e.g. both lines in 2D or 3D). """
    n_lines1, n_lines2 = len(line_seg1), len(line_seg2)

    # Compute the average orthogonal line distance
    coords_2_on_1, line_dists2 = project_point_to_line(
        line_seg1, line_seg2.reshape(n_lines2 * 2, -1))
    line_dists2 = line_dists2.reshape(n_lines1, n_lines2, 2).sum(axis=2)
    coords_1_on_2, line_dists1 = project_point_to_line(
        line_seg2, line_seg1.reshape(n_lines1 * 2, -1))
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
    if mode == 'mean':
        low_overlaps = overlaps < min_overlap
    else:
        low_overlaps = min_overlaps < min_overlap
    line_dists[low_overlaps] = torch.amax(line_dists)
    return line_dists

def merge_line_cluster(lines):
    """ Merge a cluster of line segments.
    First compute the principal direction of the lines, compute the
    endpoints barycenter, project the endpoints onto the middle line,
    keep the two extreme projections.
    Args:
        lines: a (n, 2, 2) torch tensor containing n lines.
    Returns:
        The merged (2, 2) torch tensor line segment.
    """
    # Get the principal direction of the endpoints
    points = lines.reshape(-1, 2)
    weights = torch.norm((lines[:, 0] - lines[:, 1]).float(), dim=1)
    weights = torch.repeat_interleave(weights, 2)[:, None]
    weights /= weights.sum()  # More weights for longer lines
    avg_points = torch.sum((points * weights),dim=0)
    points_bar = points - avg_points[None]
    cov = 1 / 3 * torch.sum(torch.einsum(
        'ij,ik->ijk', points_bar, points_bar * weights),dim=0)
    a, b, c = cov[0, 0], cov[0, 1], cov[1, 1]
    # Principal component of a 2x2 symmetric matrix
    if b == 0:
        u = torch.tensor([1, 0]) if a >= c else torch.tensor([0, 1])
    else:
        m = (c - a + torch.sqrt((a - c) ** 2 + 4 * b ** 2)) / (2 * b)
        u = torch.tensor([1, m]) / torch.sqrt(1 + m ** 2)

    # Get the center of gravity of all endpoints
    cross = torch.mean(points.float(), dim=0)

    # Project the endpoints on the line defined by cross and u
    avg_line_seg = torch.stack([cross, cross + u], dim=0)
    proj = project_point_to_line(avg_line_seg[None], points)[0]

    # Take the two extremal projected endpoints
    new_line = torch.stack([cross + torch.amin(proj) * u,
                         cross + torch.amax(proj) * u], dim=0)
    return new_line


def merge_lines(lines, thresh=5., overlap_thresh=0.):
    """ Given a set of lines, merge close-by lines together.
    Two lines are merged when their orthogonal distance is smaller
    than a threshold and they have a positive overlap.
    Args:
        lines: a (N, 2, 2) torch tensor.
        thresh: maximum orthogonal distance between two lines to be merged.
        overlap_thresh: maximum distance between 2 endpoints to merge
                        two aligned lines.
    Returns:
        The new lines after merging.
    """
    if len(lines) == 0:
        return lines

    # Compute the pairwise orthogonal distances and overlap
    orth_dist, overlaps = get_orth_line_dist(lines, lines, return_overlap=True)

    # Define clusters of close-by lines to merge
    if overlap_thresh == 0:
        adjacency_mat = (overlaps > 0) * (orth_dist < thresh)
    else:
        # Filter using the distance between the two closest endpoints
        n = len(lines)
        endpoints = lines.reshape(n * 2, 2)
        close_endpoint = torch.norm(endpoints[:, None] - endpoints[None],
                                        dim=2)
        close_endpoint = close_endpoint.reshape(n, 2, n, 2).transpose(
            0, 2, 1, 3).reshape(n, n, 4)
        close_endpoint = torch.amin(close_endpoint, dim=2)
        adjacency_mat = (((overlaps > 0) | (close_endpoint < overlap_thresh))
                         * (orth_dist < thresh))
    n_comp, components = connected_components(adjacency_mat, directed=False)

    # For each cluster, merge all lines into a single one
    new_lines = []
    for i in range(n_comp):
        cluster = lines[components == i]
        new_lines.append(merge_line_cluster(cluster))

    return torch.stack(new_lines, dim=0)

def filter_outlier_lines(
    img, lines, df, angle, mode='inlier_thresh', use_grad=False,
    inlier_thresh=0.5, df_thresh=1.5, ang_thresh=np.pi / 6, n_samples=50):
    """ Filter out outlier lines either by comparing the average DF and
        line level values to a threshold or by counting the number of inliers
        across the line. It can also optionally use the image gradient.
    Args:
        img: the original image.
        lines: a (N, 2, 2) np array.
        df: np array with the distance field.
        angle: np array with the grad angle field.
        mode: 'mean' or 'inlier_thresh'.
        use_grad: True to use the image gradient instead of line_level.
        inlier_thresh: ratio of inliers to get accepted.
        df_thresh, ang_thresh: thresholds to determine a valid value.
        n_samples: number of points sampled along each line.
    Returns:
        A tuple with the filtered lines and a mask of valid lines.
    """
    # Get the right orientation of the line_level and the lines orientation
    oriented_line_level, img_grad_angle = preprocess_angle(angle, img)
    orientations = get_line_orientation(lines, oriented_line_level)

    # Get the sample positions
    t = np.linspace(0, 1, n_samples)[None, :, None]
    samples = lines[:, 0][:, None] + t * (lines[:, 1][:, None]
                                          - lines[:, 0][:, None])
    samples = samples.reshape(-1, 2)

    # Interpolate the DF and angle map
    df_samples = bilinear_interpolate_numpy(df, samples[:, 1], samples[:, 0])
    df_samples = df_samples.reshape(-1, n_samples)
    if use_grad:
        oriented_line_level = np.mod(img_grad_angle - np.pi / 2, 2 * np.pi)
    ang_samples = nn_interpolate_numpy(oriented_line_level, samples[:, 1],
                                       samples[:, 0]).reshape(-1, n_samples)

    # Check the average value or number of inliers
    if mode == 'mean':
        df_check = np.mean(df_samples, axis=1) < df_thresh
        ang_avg = np.arctan2(np.sin(ang_samples).sum(axis=1),
                             np.cos(ang_samples).sum(axis=1))
        ang_diff = np.minimum(np.abs(ang_avg - orientations),
                              2 * np.pi - np.abs(ang_avg - orientations))
        ang_check = ang_diff < ang_thresh
        valid = df_check & ang_check
    elif mode == 'inlier_thresh':
        df_check = df_samples < df_thresh
        ang_diff = np.minimum(
            np.abs(ang_samples - orientations[:, None]),
            2 * np.pi - np.abs(ang_samples - orientations[:, None]))
        ang_check = ang_diff < ang_thresh
        valid = (df_check & ang_check).mean(axis=1) > inlier_thresh
    else:
        raise ValueError("Unknown filtering mode: " + mode)

    return lines[valid], valid