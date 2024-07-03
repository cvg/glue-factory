"""
    A set of geometry tools to handle lines in Pytorch and Numpy.
"""

import numpy as np
import torch

from gluefactory.datasets.homographies_deeplsd import warp_lines
from gluefactory.utils.image import compute_image_grad

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
    coords1d = (
        torch.sum((points[None] - line_segs[:, None, 0]) * dir_vec, dim=2)
        / torch.norm(dir_vec.float(), dim=2) ** 2
    )
    # coords1d is of shape (n_lines, n_points)

    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    dist_to_line = torch.norm(projection - points[None], dim=2)

    return coords1d, dist_to_line


def project_point_to_line_torch(line_segs, points):
    """Given a list of line segments and a list of points (2D or 3D coordinates) as torch tensors,
    compute the orthogonal projection of all points on all lines.
    This returns the 1D coordinates of the projection on the line,
    as well as the list of orthogonal distances."""
    # Compute the 1D coordinate of the points projected on the line
    dir_vec = (line_segs[:, 1] - line_segs[:, 0])[:, None]
    coords1d = (
        torch.sum((points[None] - line_segs[:, None, 0]) * dir_vec, dim=2)
        / torch.norm(dir_vec.float(), dim=2) ** 2
    )
    # coords1d is of shape (n_lines, n_points)

    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    dist_to_line = torch.norm(projection - points[None], dim=2)

    return coords1d, dist_to_line


def preprocess_angle(angle, img, mask=False):
    """Convert a grad angle field into a line level angle, using
    the image gradient to get the right orientation."""
    oriented_grad_angle, img_grad_angle = align_with_grad_angle(angle, img)
    oriented_grad_angle = np.mod(oriented_grad_angle - np.pi / 2, 2 * np.pi)
    if mask:
        oriented_grad_angle[0] = -1024
        oriented_grad_angle[:, 0] = -1024
    return oriented_grad_angle.astype(np.float64), img_grad_angle


### Line segment distances


def project_point_to_line(line_segs, points):
    """Given a list of line segments and a list of points (2D or 3D coordinates),
    compute the orthogonal projection of all points on all lines.
    This returns the 1D coordinates of the projection on the line,
    as well as the list of orthogonal distances."""
    # Compute the 1D coordinate of the points projected on the line
    dir_vec = (line_segs[:, 1] - line_segs[:, 0])[:, None]
    coords1d = (
        torch.sum((points[None] - line_segs[:, None, 0]) * dir_vec, dim=2)
        / torch.norm(dir_vec.float(), dim=2) ** 2
    )
    # coords1d is of shape (n_lines, n_points)

    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    dist_to_line = torch.norm(projection - points[None], dim=2)

    return coords1d, dist_to_line


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


def preprocess_angle(angle, img, mask=False):
    """Convert a grad angle field into a line level angle, using
    the image gradient to get the right orientation."""
    oriented_grad_angle, img_grad_angle = align_with_grad_angle(angle, img)
    oriented_grad_angle = np.mod(oriented_grad_angle - np.pi / 2, 2 * np.pi)
    if mask:
        oriented_grad_angle[0] = -1024
        oriented_grad_angle[:, 0] = -1024
    return oriented_grad_angle.astype(np.float64), img_grad_angle


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


def nn_interpolate_numpy(img, x, y):
    xi = np.clip(np.round(x).astype(int), 0, img.shape[1] - 1)
    yi = np.clip(np.round(y).astype(int), 0, img.shape[0] - 1)
    return img[yi, xi]


def align_with_grad_angle(angle, img):
    """Starting from an angle in [0, pi], find the sign of the angle based on
    the image gradient of the corresponding pixel."""
    # Image gradient
    img_grad_angle = compute_image_grad(img)[3]

    # Compute the distance of the image gradient to the angle
    # and angle - pi
    pred_grad = np.mod(angle, np.pi)  # in [0, pi]
    pos_dist = np.minimum(
        np.abs(img_grad_angle - pred_grad),
        2 * np.pi - np.abs(img_grad_angle - pred_grad),
    )
    neg_dist = np.minimum(
        np.abs(img_grad_angle - pred_grad + np.pi),
        2 * np.pi - np.abs(img_grad_angle - pred_grad + np.pi),
    )

    # Assign the new grad angle to the closest of the two
    is_pos_closest = np.argmin(np.stack([neg_dist, pos_dist], axis=-1), axis=-1).astype(
        bool
    )
    new_grad_angle = np.where(is_pos_closest, pred_grad, pred_grad - np.pi)
    return new_grad_angle, img_grad_angle


def clip_line_to_boundary(lines):
    """Clip the first coordinate of a set of lines to the lower boundary 0
        and indicate which lines are completely outside of the boundary.
    Args:
        lines: a [N, 2, 2] tensor of lines.
    Returns:
        The clipped coordinates + a mask indicating invalid lines.
    """
    updated_lines = lines.copy()

    # Detect invalid lines completely outside of the first boundary
    invalid = np.all(lines[:, :, 0] < 0, axis=1)

    # Clip the lines to the boundary and update the second coordinate
    # First endpoint
    out = lines[:, 0, 0] < 0
    denom = lines[:, 1, 0] - lines[:, 0, 0]
    denom[denom == 0] = 1e-6
    ratio = lines[:, 1, 0] / denom
    updated_y = ratio * lines[:, 0, 1] + (1 - ratio) * lines[:, 1, 1]
    updated_lines[out, 0, 1] = updated_y[out]
    updated_lines[out, 0, 0] = 0
    # Second endpoint
    out = lines[:, 1, 0] < 0
    denom = lines[:, 0, 0] - lines[:, 1, 0]
    denom[denom == 0] = 1e-6
    ratio = lines[:, 0, 0] / denom
    updated_y = ratio * lines[:, 1, 1] + (1 - ratio) * lines[:, 0, 1]
    updated_lines[out, 1, 1] = updated_y[out]
    updated_lines[out, 1, 0] = 0

    return updated_lines, invalid


def clip_line_to_boundaries(lines, img_size, min_len=10):
    """Clip a set of lines to the image boundaries and indicate
        which lines are completely outside of the boundaries.
    Args:
        lines: a [N, 2, 2] tensor of lines.
        img_size: the original image size.
    Returns:
        The clipped coordinates + a mask indicating valid lines.
    """
    new_lines = lines.copy()

    # Clip the first coordinate to the 0 boundary of img1
    new_lines, invalid_x0 = clip_line_to_boundary(lines)

    # Mirror in first coordinate to clip to the H-1 boundary
    new_lines[:, :, 0] = img_size[0] - 1 - new_lines[:, :, 0]
    new_lines, invalid_xh = clip_line_to_boundary(new_lines)
    new_lines[:, :, 0] = img_size[0] - 1 - new_lines[:, :, 0]

    # Swap the two coordinates, perform the same for y, and swap back
    new_lines = new_lines[:, :, [1, 0]]
    new_lines, invalid_y0 = clip_line_to_boundary(new_lines)
    new_lines[:, :, 0] = img_size[1] - 1 - new_lines[:, :, 0]
    new_lines, invalid_yw = clip_line_to_boundary(new_lines)
    new_lines[:, :, 0] = img_size[1] - 1 - new_lines[:, :, 0]
    new_lines = new_lines[:, :, [1, 0]]

    # Merge all the invalid lines and also remove lines that became too short
    short = np.linalg.norm(new_lines[:, 1] - new_lines[:, 0], axis=1) < min_len
    valid = np.logical_not(invalid_x0 | invalid_xh | invalid_y0 | invalid_yw | short)

    return new_lines, valid


def get_common_lines(lines0, lines1, H, img_size):
    """Extract the lines in common between two views, by warping lines1
        into lines0 frame.
    Args:
        lines0, lines1: sets of lines of size [N, 2, 2].
        H: homography relating the lines.
        img_size: size of the original images img0 and img1.
    Returns:
        Updated lines0 with a valid reprojection in img1 and warped_lines1.
    """
    # First warp lines0 to img1 to detect invalid lines
    warped_lines0 = warp_lines(lines0, H)

    # Clip them to the boundary
    warped_lines0, valid = clip_line_to_boundaries(warped_lines0, img_size)

    # Warp all the valid lines back in img0
    inv_H = np.linalg.inv(H)
    new_lines0 = warp_lines(warped_lines0[valid], inv_H)
    warped_lines1 = warp_lines(lines1, inv_H)
    warped_lines1, valid = clip_line_to_boundaries(warped_lines1, img_size)

    return new_lines0, warped_lines1[valid]
