from scipy.sparse.csgraph import connected_components
import torch

def project_point_to_line(line_segs, points):
    """ Given a list of line segments and a list of points (2D or 3D coordinates),
        compute the orthogonal projection of all points on all lines.
        This returns the 1D coordinates of the projection on the line,
        as well as the list of orthogonal distances. """
    # Compute the 1D coordinate of the points projected on the line
    dir_vec = (line_segs[:, 1] - line_segs[:, 0])[:, None]
    # print(line_segs[:, None, 0].shape, points[None].shape, dir_vec.shape)
    # print(((points[None] - line_segs[:, None, 0]) * dir_vec).sum(axis=2))
    coords1d = (((points[None] - line_segs[:, None, 0]) * dir_vec).sum(axis=2)
                / torch.linalg.norm(dir_vec, axis=2) ** 2)
    # coords1d is of shape (n_lines, n_points)

    
    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    dist_to_line = torch.linalg.norm(projection - points[None], axis=2)

    return coords1d, dist_to_line


def get_segment_overlap(seg_coord1d):
    """ Given a list of segments parameterized by the 1D coordinate
        of the endpoints, compute the overlap with the segment [0, 1]. """
    seg_coord1d, indices = torch.sort(seg_coord1d, axis=-1)
    overlap = ((seg_coord1d[..., 1] > 0) * (seg_coord1d[..., 0] < 1)
               * (torch.minimum(seg_coord1d[..., 1], torch.ones_like(seg_coord1d[..., 1]))
                  - torch.maximum(seg_coord1d[..., 0], torch.zeros_like(seg_coord1d[..., 0]))))
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
    weights = torch.linalg.norm(lines[:, 0] - lines[:, 1], axis=1).to(lines.device)
    weights = weights.repeat(2)[:, None]
    weights /= weights.sum()  # More weights for longer lines
    avg_points = (points * weights).sum(axis=0)
    points_bar = points - avg_points[None]
    cov = 1 / 3 * torch.einsum(
        'ij,ik->ijk', points_bar, points_bar * weights).sum(axis=0).to(lines.device)
    a, b, c = cov[0, 0], cov[0, 1], cov[1, 1]
    # Principal component of a 2x2 symmetric matrix
    if b == 0:
        u = torch.Tensor([1, 0]) if a >= c else torch.Tensor([0, 1])
        u = u.to(lines.device)
    else:
        m = (c - a + torch.sqrt((a - c) ** 2 + 4 * b ** 2)) / (2 * b)
        m = m.to(lines.device)
        u = torch.Tensor([1, m]).to(lines.device) / torch.sqrt(1 + m ** 2)
        
    # Get the center of gravity of all endpoints
    cross = torch.mean(points, axis=0)
        
    # Project the endpoints on the line defined by cross and u
    avg_line_seg = torch.stack([cross, cross + u], axis=0)
    proj = project_point_to_line(avg_line_seg[None], points)[0]
    
    # Take the two extremal projected endpoints
    new_line = torch.stack([cross + torch.amin(proj) * u,
                            cross + torch.amax(proj) * u], axis=0)
    return new_line


def merge_lines(lines, thresh=5., overlap_thresh=0.):
    """ Given a set of lines, merge close-by lines together.
    Two lines are merged when their orthogonal distance is smaller
    than a threshold and they have a positive overlap.
    Args:
        lines: a (N, 2, 2) np array.
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
        close_endpoint = torch.linalg.norm(endpoints[:, None] - endpoints[None],
                                           axis=2)
        close_endpoint = close_endpoint.reshape(n, 2, n, 2).transpose(
            0, 2, 1, 3).reshape(n, n, 4)
        close_endpoint = torch.amin(close_endpoint, axis=2)
        adjacency_mat = (((overlaps > 0) | (close_endpoint < overlap_thresh))
                         * (orth_dist < thresh))
    n_comp, components = connected_components(adjacency_mat.cpu(), directed=False)
    
    # For each cluster, merge all lines into a single one
    new_lines = []
    for i in range(n_comp):
        cluster = lines[components == i]
        new_lines.append(merge_line_cluster(cluster).to(lines.device))
        
    return torch.stack(new_lines, axis=0)