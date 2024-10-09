import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

from gluefactory.models.lines.line_distances import (
    get_orth_line_dist,
    get_orth_line_dist_torch,
)
from gluefactory.models.lines.line_utils import (
    bilinear_interpolate_numpy,
    nn_interpolate_numpy,
    preprocess_angle,
    project_point_to_line,
    project_point_to_line_torch,
)


def merge_line_cluster(lines):
    """Merge a cluster of line segments.
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
    avg_points = torch.sum((points * weights), dim=0)
    points_bar = points - avg_points[None]
    cov = (
        1
        / 3
        * torch.sum(torch.einsum("ij,ik->ijk", points_bar, points_bar * weights), dim=0)
    )
    a, b, c = cov[0, 0], cov[0, 1], cov[1, 1]
    # Principal component of a 2x2 symmetric matrix
    if b == 0:
        u = torch.tensor([1, 0]) if a >= c else torch.tensor([0, 1])
    else:
        m = (c - a + torch.sqrt((a - c) ** 2 + 4 * b**2)) / (2 * b)
        u = torch.tensor([1, m]) / torch.sqrt(1 + m**2)

    # Get the center of gravity of all endpoints
    cross = torch.mean(points.float(), dim=0)

    # Project the endpoints on the line defined by cross and u
    avg_line_seg = torch.stack([cross, cross + u], dim=0)
    proj = project_point_to_line(avg_line_seg[None], points)[0]

    # Take the two extremal projected endpoints
    new_line = torch.stack(
        [cross + torch.amin(proj) * u, cross + torch.amax(proj) * u], dim=0
    )
    return new_line


def merge_line_cluster_torch(lines, return_indices=False):
    """Merge a cluster of line segments.
    First compute the principal direction of the lines, compute the
    endpoints barycenter, project the endpoints onto the middle line,
    keep the two extreme projections.
    Args:
        lines: a (N, 2, 2) torch tensor if return_indices is False, otherwise
                a (N, 2, 3) torch tensor where the first two channels are the line endpoints,
                the third channel is the index of the keypoints.
    Returns:
        The merged (2, 2) torch tensor line segment.
    """
    device = lines.device
    orig_lines = lines
    lines = orig_lines[:, :, :2]

    if return_indices:
        indices = orig_lines[:, :, 2].reshape(-1)

    # Get the principal direction of the endpoints
    points = lines.reshape(-1, 2)
    weights = torch.norm((lines[:, 0] - lines[:, 1]).float(), dim=1)
    weights = torch.repeat_interleave(weights, 2)[:, None]
    weights /= weights.sum()  # More weights for longer lines
    avg_points = torch.sum((points * weights), dim=0)
    points_bar = points - avg_points[None]
    cov = (
        1
        / 3
        * torch.sum(torch.einsum("ij,ik->ijk", points_bar, points_bar * weights), dim=0)
    )
    a, b, c = cov[0, 0], cov[0, 1], cov[1, 1]
    # Principal component of a 2x2 symmetric matrix
    if b == 0:
        u = torch.tensor([1, 0], device=device) if a >= c else torch.tensor([0, 1])
    else:
        m = (c - a + torch.sqrt((a - c) ** 2 + 4 * b**2)) / (2 * b)
        u = torch.tensor([1, m], device=device) / torch.sqrt(1 + m**2)

    # Get the center of gravity of all endpoints
    cross = torch.mean(points.float(), dim=0)

    # Project the endpoints on the line defined by cross and u
    avg_line_seg = torch.stack([cross, cross + u], dim=0)
    proj = project_point_to_line_torch(avg_line_seg[None], points)[0]

    if not return_indices:
        # Take the two extremal projected endpoints
        new_line = torch.stack(
            [cross + torch.amin(proj) * u, cross + torch.amax(proj) * u], dim=0
        )
    else:
        # Return the keypoints indices of the two extremal projected endpoints
        new_line = torch.stack(
            [indices[torch.argmin(proj)], indices[torch.argmax(proj)]], dim=0
        )

    return new_line


def merge_lines(lines, thresh=5.0, overlap_thresh=0.0):
    """Given a set of lines, merge close-by lines together.
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
        close_endpoint = torch.norm(endpoints[:, None] - endpoints[None], dim=2)
        close_endpoint = (
            close_endpoint.reshape(n, 2, n, 2).transpose(0, 2, 1, 3).reshape(n, n, 4)
        )
        close_endpoint = torch.amin(close_endpoint, dim=2)
        adjacency_mat = ((overlaps > 0) | (close_endpoint < overlap_thresh)) * (
            orth_dist < thresh
        )
    n_comp, components = connected_components(adjacency_mat, directed=False)

    # For each cluster, merge all lines into a single one
    new_lines = []
    for i in range(n_comp):
        cluster = lines[components == i]
        new_lines.append(merge_line_cluster(cluster))

    return torch.stack(new_lines, dim=0)


def merge_lines_torch(lines, thresh=5.0, overlap_thresh=0.0, return_indices=False):
    """Given a set of lines, merge close-by lines together.
    Two lines are merged when their orthogonal distance is smaller
    than a threshold and they have a positive overlap.
    Args:
        lines: a (N, 2, 2) torch tensor if return_indices is False, otherwise
                a (N, 2, 3) torch tensor. The first two channels are the line endpoints,
                the third channel is the index of the keypoints.
        thresh: maximum orthogonal distance between two lines to be merged.
        overlap_thresh: maximum distance between 2 endpoints to merge
                        two aligned lines.
    Returns:
        The new lines after merging.
    """
    if len(lines) == 0:
        return lines

    orig_lines = lines
    lines = lines[:, :, :2]

    # Compute the pairwise orthogonal distances and overlap
    orth_dist, overlaps = get_orth_line_dist_torch(lines, lines, return_overlap=True)

    # Define clusters of close-by lines to merge
    if overlap_thresh == 0:
        adjacency_mat = (overlaps > 0) * (orth_dist < thresh)
    else:
        # Filter using the distance between the two closest endpoints
        n = len(lines)
        endpoints = lines.reshape(n * 2, 2)
        close_endpoint = torch.norm(endpoints[:, None] - endpoints[None], dim=2)
        close_endpoint = (
            close_endpoint.reshape(n, 2, n, 2).transpose(0, 2, 1, 3).reshape(n, n, 4)
        )
        close_endpoint = torch.amin(close_endpoint, dim=2)
        adjacency_mat = ((overlaps > 0) | (close_endpoint < overlap_thresh)) * (
            orth_dist < thresh
        )
    n_comp, components = connected_components(adjacency_mat.cpu(), directed=False)

    # For each cluster, merge all lines into a single one
    new_lines = []
    for i in range(n_comp):
        cluster = orig_lines[components == i]
        if len(cluster) == 1:
            if return_indices:
                new_lines.append(cluster[0, :, 2])
            else:
                new_lines.append(cluster[0, :, :2])
        else:
            new_lines.append(merge_line_cluster_torch(cluster, return_indices))

    return torch.stack(new_lines, dim=0)


def sample_along_line(lines, img, n_samples=10, mode="mean"):
    """Sample a fixed number of points along each line and interpolate
    an img at these points, and finally aggregate the values."""
    # Get the sample positions
    t = np.linspace(0, 1, 10)[None, :, None]
    samples = lines[:, 0][:, None] + t * (lines[:, 1][:, None] - lines[:, 0][:, None])
    samples = samples.reshape(-1, 2)

    # Interpolate the img at the samples and aggregate the values
    if mode == "mean":
        # Average
        val = bilinear_interpolate_numpy(img, samples[:, 1], samples[:, 0])
        val = np.mean(val.reshape(-1, n_samples), axis=-1)
    elif mode == "angle":
        # Average of angles
        val = nn_interpolate_numpy(img, samples[:, 1], samples[:, 0])
        val = val.reshape(-1, n_samples)
        val = np.arctan2(np.sin(val).sum(axis=-1), np.cos(val).sum(axis=-1))
    elif mode == "median":
        # Median
        val = nn_interpolate_numpy(img, samples[:, 1], samples[:, 0])
        val = np.median(val.reshape(-1, n_samples), axis=-1)
    else:
        # No aggregation
        val = nn_interpolate_numpy(img, samples[:, 1], samples[:, 0])
        val = val.reshape(-1, n_samples)

    return val


def get_line_orientation(lines, angle):
    """Get the orientation in [-pi, pi] of a line, based on the gradient."""
    grad_val = sample_along_line(lines, angle, mode="angle")
    line_ori = np.mod(
        np.arctan2(lines[:, 1, 0] - lines[:, 0, 0], lines[:, 1, 1] - lines[:, 0, 1]),
        np.pi,
    )

    pos_dist = np.minimum(
        np.abs(grad_val - line_ori), 2 * np.pi - np.abs(grad_val - line_ori)
    )
    neg_dist = np.minimum(
        np.abs(grad_val - line_ori + np.pi),
        2 * np.pi - np.abs(grad_val - line_ori + np.pi),
    )
    line_ori = np.where(pos_dist <= neg_dist, line_ori, line_ori - np.pi)
    return line_ori


def filter_outlier_lines(
    img,
    lines,
    df,
    angle,
    mode="inlier_thresh",
    use_grad=False,
    inlier_thresh=0.5,
    df_thresh=1.5,
    ang_thresh=np.pi / 6,
    n_samples=50,
):
    """Filter out outlier lines either by comparing the average DF and
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
    samples = lines[:, 0][:, None] + t * (lines[:, 1][:, None] - lines[:, 0][:, None])
    samples = samples.reshape(-1, 2)

    # Interpolate the DF and angle map
    df_samples = bilinear_interpolate_numpy(df, samples[:, 1], samples[:, 0])
    df_samples = df_samples.reshape(-1, n_samples)
    if use_grad:
        oriented_line_level = np.mod(img_grad_angle - np.pi / 2, 2 * np.pi)
    ang_samples = nn_interpolate_numpy(
        oriented_line_level, samples[:, 1], samples[:, 0]
    ).reshape(-1, n_samples)

    # Check the average value or number of inliers
    if mode == "mean":
        df_check = np.mean(df_samples, axis=1) < df_thresh
        ang_avg = np.arctan2(
            np.sin(ang_samples).sum(axis=1), np.cos(ang_samples).sum(axis=1)
        )
        ang_diff = np.minimum(
            np.abs(ang_avg - orientations), 2 * np.pi - np.abs(ang_avg - orientations)
        )
        ang_check = ang_diff < ang_thresh
        valid = df_check & ang_check
    elif mode == "inlier_thresh":
        df_check = df_samples < df_thresh
        ang_diff = np.minimum(
            np.abs(ang_samples - orientations[:, None]),
            2 * np.pi - np.abs(ang_samples - orientations[:, None]),
        )
        ang_check = ang_diff < ang_thresh
        valid = (df_check & ang_check).mean(axis=1) > inlier_thresh
    else:
        raise ValueError("Unknown filtering mode: " + mode)

    return lines[valid], valid
