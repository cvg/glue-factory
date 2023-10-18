import math
from typing import Tuple

import numpy as np
import torch

from .utils import from_homogeneous, to_homogeneous


def flat2mat(H):
    return np.reshape(np.concatenate([H, np.ones_like(H[:, :1])], axis=1), [3, 3])


# Homography creation


def create_center_patch(shape, patch_shape=None):
    if patch_shape is None:
        patch_shape = shape
    width, height = shape
    pwidth, pheight = patch_shape
    left = int((width - pwidth) / 2)
    bottom = int((height - pheight) / 2)
    right = int((width + pwidth) / 2)
    top = int((height + pheight) / 2)
    return np.array([[left, bottom], [left, top], [right, top], [right, bottom]])


def check_convex(patch, min_convexity=0.05):
    """Checks if given polygon vertices [N,2] form a convex shape"""
    for i in range(patch.shape[0]):
        x1, y1 = patch[(i - 1) % patch.shape[0]]
        x2, y2 = patch[i]
        x3, y3 = patch[(i + 1) % patch.shape[0]]
        if (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1) > -min_convexity:
            return False
    return True


def sample_homography_corners(
    shape,
    patch_shape,
    difficulty=1.0,
    translation=0.4,
    n_angles=10,
    max_angle=90,
    min_convexity=0.05,
    rng=np.random,
):
    max_angle = max_angle / 180.0 * math.pi
    width, height = shape
    pwidth, pheight = width * (1 - difficulty), height * (1 - difficulty)
    min_pts1 = create_center_patch(shape, (pwidth, pheight))
    full = create_center_patch(shape)
    pts2 = create_center_patch(patch_shape)
    scale = min_pts1 - full
    found_valid = False
    cnt = -1
    while not found_valid:
        offsets = rng.uniform(0.0, 1.0, size=(4, 2)) * scale
        pts1 = full + offsets
        found_valid = check_convex(pts1 / np.array(shape), min_convexity)
        cnt += 1

    # re-center
    pts1 = pts1 - np.mean(pts1, axis=0, keepdims=True)
    pts1 = pts1 + np.mean(min_pts1, axis=0, keepdims=True)

    # Rotation
    if n_angles > 0 and difficulty > 0:
        angles = np.linspace(-max_angle * difficulty, max_angle * difficulty, n_angles)
        rng.shuffle(angles)
        rng.shuffle(angles)
        angles = np.concatenate([[0.0], angles], axis=0)

        center = np.mean(pts1, axis=0, keepdims=True)
        rot_mat = np.reshape(
            np.stack(
                [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)],
                axis=1,
            ),
            [-1, 2, 2],
        )
        rotated = (
            np.matmul(
                np.tile(np.expand_dims(pts1 - center, axis=0), [n_angles + 1, 1, 1]),
                rot_mat,
            )
            + center
        )

        for idx in range(1, n_angles):
            warped_points = rotated[idx] / np.array(shape)
            if np.all((warped_points >= 0.0) & (warped_points < 1.0)):
                pts1 = rotated[idx]
                break

    # Translation
    if translation > 0:
        min_trans = -np.min(pts1, axis=0)
        max_trans = shape - np.max(pts1, axis=0)
        trans = rng.uniform(min_trans, max_trans)[None]
        pts1 += trans * translation * difficulty

    H = compute_homography(pts1, pts2, [1.0, 1.0])
    warped = warp_points(full, H, inverse=False)
    return H, full, warped, patch_shape


def compute_homography(pts1_, pts2_, shape):
    """Compute the homography matrix from 4 point correspondences"""
    # Rescale to actual size
    shape = np.array(shape[::-1], dtype=np.float32)  # different convention [y, x]
    pts1 = pts1_ * np.expand_dims(shape, axis=0)
    pts2 = pts2_ * np.expand_dims(shape, axis=0)

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(
        np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0)
    )
    homography = np.transpose(np.linalg.solve(a_mat, p_mat))
    return flat2mat(homography)


# Point warping utils


def warp_points(points, homography, inverse=True):
    """
    Warp a list of points with the INVERSE of the given homography.
    The inverse is used to be coherent with tf.contrib.image.transform
    Arguments:
        points: list of N points, shape (N, 2).
        homography: batched or not (shapes (B, 3, 3) and (3, 3) respectively).
    Returns: a Tensor of shape (N, 2) or (B, N, 2) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.
    """
    H = homography[None] if len(homography.shape) == 2 else homography

    # Get the points to the homogeneous format
    num_points = points.shape[0]
    # points = points.astype(np.float32)[:, ::-1]
    points = np.concatenate([points, np.ones([num_points, 1], dtype=np.float32)], -1)

    H_inv = np.transpose(np.linalg.inv(H) if inverse else H)
    warped_points = np.tensordot(points, H_inv, axes=[[1], [0]])

    warped_points = np.transpose(warped_points, [2, 0, 1])
    warped_points[np.abs(warped_points[:, :, 2]) < 1e-8, 2] = 1e-8
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]

    return warped_points[0] if len(homography.shape) == 2 else warped_points


def warp_points_torch(points, H, inverse=True):
    """
    Warp a list of points with the INVERSE of the given homography.
    The inverse is used to be coherent with tf.contrib.image.transform
    Arguments:
        points: batched list of N points, shape (B, N, 2).
        H: batched or not (shapes (B, 3, 3) and (3, 3) respectively).
        inverse: Whether to multiply the points by H or the inverse of H
    Returns: a Tensor of shape (B, N, 2) containing the new coordinates of the warps.
    """

    # Get the points to the homogeneous format
    points = to_homogeneous(points)

    # Apply the homography
    H_mat = (torch.inverse(H) if inverse else H).transpose(-2, -1)
    warped_points = torch.einsum("...nj,...ji->...ni", points, H_mat)

    warped_points = from_homogeneous(warped_points, eps=1e-5)
    return warped_points


# Line warping utils


def seg_equation(segs):
    # calculate list of start, end and midpoints points from both lists
    start_points, end_points = to_homogeneous(segs[..., 0, :]), to_homogeneous(
        segs[..., 1, :]
    )
    # Compute the line equations as ax + by + c = 0 , where x^2 + y^2 = 1
    lines = torch.cross(start_points, end_points, dim=-1)
    lines_norm = torch.sqrt(lines[..., 0] ** 2 + lines[..., 1] ** 2)[..., None]
    assert torch.all(
        lines_norm > 0
    ), "Error: trying to compute the equation of a line with a single point"
    lines = lines / lines_norm
    return lines


def is_inside_img(pts: torch.Tensor, img_shape: Tuple[int, int]):
    h, w = img_shape
    return (
        (pts >= 0).all(dim=-1)
        & (pts[..., 0] < w)
        & (pts[..., 1] < h)
        & (~torch.isinf(pts).any(dim=-1))
    )


def shrink_segs_to_img(segs: torch.Tensor, img_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Shrink an array of segments to fit inside the image.
    :param segs: The tensor of segments with shape (N, 2, 2)
    :param img_shape: The image shape in format (H, W)
    """
    EPS = 1e-4
    device = segs.device
    w, h = img_shape[1], img_shape[0]
    # Project the segments to the reference image
    segs = segs.clone()
    eqs = seg_equation(segs)
    x0, y0 = torch.tensor([1.0, 0, 0.0], device=device), torch.tensor(
        [0.0, 1, 0], device=device
    )
    x0 = x0.repeat(eqs.shape[:-1] + (1,))
    y0 = y0.repeat(eqs.shape[:-1] + (1,))
    pt_x0s = torch.cross(eqs, x0, dim=-1)
    pt_x0s = pt_x0s[..., :-1] / pt_x0s[..., None, -1]
    pt_x0s_valid = is_inside_img(pt_x0s, img_shape)
    pt_y0s = torch.cross(eqs, y0, dim=-1)
    pt_y0s = pt_y0s[..., :-1] / pt_y0s[..., None, -1]
    pt_y0s_valid = is_inside_img(pt_y0s, img_shape)

    xW = torch.tensor([1.0, 0, EPS - w], device=device)
    yH = torch.tensor([0.0, 1, EPS - h], device=device)
    xW = xW.repeat(eqs.shape[:-1] + (1,))
    yH = yH.repeat(eqs.shape[:-1] + (1,))
    pt_xWs = torch.cross(eqs, xW, dim=-1)
    pt_xWs = pt_xWs[..., :-1] / pt_xWs[..., None, -1]
    pt_xWs_valid = is_inside_img(pt_xWs, img_shape)
    pt_yHs = torch.cross(eqs, yH, dim=-1)
    pt_yHs = pt_yHs[..., :-1] / pt_yHs[..., None, -1]
    pt_yHs_valid = is_inside_img(pt_yHs, img_shape)

    # If the X coordinate of the first endpoint is out
    mask = (segs[..., 0, 0] < 0) & pt_x0s_valid
    segs[mask, 0, :] = pt_x0s[mask]
    mask = (segs[..., 0, 0] > (w - 1)) & pt_xWs_valid
    segs[mask, 0, :] = pt_xWs[mask]
    # If the X coordinate of the second endpoint is out
    mask = (segs[..., 1, 0] < 0) & pt_x0s_valid
    segs[mask, 1, :] = pt_x0s[mask]
    mask = (segs[:, 1, 0] > (w - 1)) & pt_xWs_valid
    segs[mask, 1, :] = pt_xWs[mask]
    # If the Y coordinate of the first endpoint is out
    mask = (segs[..., 0, 1] < 0) & pt_y0s_valid
    segs[mask, 0, :] = pt_y0s[mask]
    mask = (segs[..., 0, 1] > (h - 1)) & pt_yHs_valid
    segs[mask, 0, :] = pt_yHs[mask]
    # If the Y coordinate of the second endpoint is out
    mask = (segs[..., 1, 1] < 0) & pt_y0s_valid
    segs[mask, 1, :] = pt_y0s[mask]
    mask = (segs[..., 1, 1] > (h - 1)) & pt_yHs_valid
    segs[mask, 1, :] = pt_yHs[mask]

    assert (
        torch.all(segs >= 0)
        and torch.all(segs[..., 0] < w)
        and torch.all(segs[..., 1] < h)
    )
    return segs


def warp_lines_torch(
    lines, H, inverse=True, dst_shape: Tuple[int, int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param lines: A tensor of shape (B, N, 2, 2)
              where B is the batch size, N the number of lines.
    :param H: The homography used to convert the lines.
              batched or not (shapes (B, 3, 3) and (3, 3) respectively).
    :param inverse: Whether to apply H or the inverse of H
    :param dst_shape:If provided, lines are trimmed to be inside the image
    """
    device = lines.device
    batch_size = len(lines)
    lines = warp_points_torch(lines.reshape(batch_size, -1, 2), H, inverse).reshape(
        lines.shape
    )

    if dst_shape is None:
        return lines, torch.ones(lines.shape[:-2], dtype=torch.bool, device=device)

    out_img = torch.any(
        (lines < 0) | (lines >= torch.tensor(dst_shape[::-1], device=device)), -1
    )
    valid = ~out_img.all(-1)
    any_out_of_img = out_img.any(-1)
    lines_to_trim = valid & any_out_of_img

    for b in range(batch_size):
        lines_to_trim_mask_b = lines_to_trim[b]
        lines_to_trim_b = lines[b][lines_to_trim_mask_b]
        corrected_lines = shrink_segs_to_img(lines_to_trim_b, dst_shape)
        lines[b][lines_to_trim_mask_b] = corrected_lines

    return lines, valid


# Homography evaluation utils


def sym_homography_error(kpts0, kpts1, T_0to1):
    kpts0_1 = from_homogeneous(to_homogeneous(kpts0) @ T_0to1.transpose(-1, -2))
    dist0_1 = ((kpts0_1 - kpts1) ** 2).sum(-1).sqrt()

    kpts1_0 = from_homogeneous(
        to_homogeneous(kpts1) @ torch.pinverse(T_0to1.transpose(-1, -2))
    )
    dist1_0 = ((kpts1_0 - kpts0) ** 2).sum(-1).sqrt()

    return (dist0_1 + dist1_0) / 2.0


def sym_homography_error_all(kpts0, kpts1, H):
    kp0_1 = warp_points_torch(kpts0, H, inverse=False)
    kp1_0 = warp_points_torch(kpts1, H, inverse=True)

    # build a distance matrix of size [... x M x N]
    dist0 = torch.sum((kp0_1.unsqueeze(-2) - kpts1.unsqueeze(-3)) ** 2, -1).sqrt()
    dist1 = torch.sum((kpts0.unsqueeze(-2) - kp1_0.unsqueeze(-3)) ** 2, -1).sqrt()
    return (dist0 + dist1) / 2.0


def homography_corner_error(T, T_gt, image_size):
    W, H = image_size[..., 0], image_size[..., 1]
    corners0 = torch.Tensor([[0, 0], [W, 0], [W, H], [0, H]]).float().to(T)
    corners1_gt = from_homogeneous(to_homogeneous(corners0) @ T_gt.transpose(-1, -2))
    corners1 = from_homogeneous(to_homogeneous(corners0) @ T.transpose(-1, -2))
    d = torch.sqrt(((corners1 - corners1_gt) ** 2).sum(-1))
    return d.mean(-1)
