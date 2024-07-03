"""
    A set of geometry tools to handle lines in Pytorch and Numpy.
"""

import numpy as np
import torch

from gluefactory.utils.deeplsd_utils import align_with_grad_angle

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
