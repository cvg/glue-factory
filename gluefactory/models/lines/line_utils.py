"""
A set of geometry tools to handle lines in Pytorch and Numpy.
"""

import numpy as np
import torch
from homography_est import LineSegment, ransac_line_homography
from torch.nn.functional import pixel_shuffle, softmax

from gluefactory.datasets.homographies_deeplsd import warp_lines, warp_points
from gluefactory.geometry.homography import warp_lines_torch
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
    if isinstance(line_segs, np.ndarray):
        coords1d = (
            np.sum((points[None] - line_segs[:, None, 0]) * dir_vec, axis=2)
            / np.linalg.norm(dir_vec.astype(np.float32), axis=2) ** 2
        )
    else:
        coords1d = (
            torch.sum((points[None] - line_segs[:, None, 0]) * dir_vec, dim=2)
            / torch.norm(dir_vec.float(), dim=2) ** 2
        )
    # coords1d is of shape (n_lines, n_points)

    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    if isinstance(line_segs, np.ndarray):
        dist_to_line = np.linalg.norm(projection - points[None], axis=2)
    else:
        dist_to_line = torch.norm(projection - points[None], dim=2)

    return coords1d, dist_to_line


def get_segment_overlap(seg_coord1d):
    """Given a list of segments parameterized by the 1D coordinate
    of the endpoints, compute the overlap with the segment [0, 1]."""

    if isinstance(seg_coord1d, np.ndarray):

        seg_coord1d = np.sort(seg_coord1d, axis=-1)
        overlap = (
            (seg_coord1d[..., 1] > 0)
            * (seg_coord1d[..., 0] < 1)
            * (np.minimum(seg_coord1d[..., 1], 1) - np.maximum(seg_coord1d[..., 0], 0))
        )
    else:
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
    if isinstance(overlaps1, np.ndarray):
        min_overlaps = np.minimum(overlaps1, overlaps2)
    else:
        min_overlaps = torch.minimum(overlaps1, overlaps2)

    if return_overlap:
        return line_dists, overlaps

    # Enforce a max line distance for line segments with small overlap
    if mode == "mean":
        low_overlaps = overlaps < min_overlap
    else:
        low_overlaps = min_overlaps < min_overlap

    if isinstance(line_dists, np.ndarray):
        line_dists[low_overlaps] = np.amax(line_dists)
    else:
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


def clip_line_to_boundary(lines: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip the first coordinate of a set of lines to the lower boundary 0
        and indicate which lines are completely outside of the boundary.
    Args:
        lines: a [N, 2, 2] tensor of lines.
    Returns:
        The clipped coordinates + a mask indicating invalid lines.
    """
    # updated_lines = lines.detach().clone()
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


def clip_line_to_boundaries(
    lines: torch.Tensor, img_size: tuple[int, int], min_len=10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip a set of lines to the image boundaries and indicate
        which lines are completely outside of the boundaries.
    Args:
        lines: a [N, 2, 2] tensor of lines.
        img_size: the original image size.
    Returns:
        The clipped coordinates + a mask indicating valid lines.
    """
    # new_lines = lines.detach().clone()
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


def get_common_lines(
    lines0: torch.Tensor, lines1: torch.Tensor, H: torch.Tensor, img_size
) -> tuple[torch.Tensor, torch.Tensor]:
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
    if isinstance(H, np.ndarray):
        warped_lines0 = warp_lines(lines0, H)
    else:
        warped_lines0 = warp_lines(lines0.cpu().numpy(), H.cpu().numpy())

    # Clip them to the boundary
    warped_lines0, valid = clip_line_to_boundaries(warped_lines0, img_size)

    # Warp all the valid lines back in img0
    if isinstance(H, np.ndarray):
        inv_H = np.linalg.inv(H)
    else:
        inv_H = np.linalg.inv(H.cpu().numpy())

    new_lines0 = warp_lines(warped_lines0[valid], inv_H)

    if isinstance(H, np.ndarray):
        warped_lines1 = warp_lines(lines1, inv_H)
    else:
        warped_lines1 = warp_lines(lines1.cpu().numpy(), inv_H)

    warped_lines1, valid = clip_line_to_boundaries(warped_lines1, img_size)

    if isinstance(H, np.ndarray):
        return new_lines0, warped_lines1[valid]
    else:
        return torch.Tensor(new_lines0, device=lines0.device), torch.Tensor(
            warped_lines1[valid], device=lines1.device
        )


# Taken from SOLD2
def line_map_to_segments(junctions: np.ndarray, line_map: np.ndarray) -> np.ndarray:
    """Convert a line map to a Nx2x2 list of segments."""
    line_map_tmp = line_map.copy()

    output_segments = np.zeros([0, 2, 2])
    for idx in range(junctions.shape[0]):
        # if no connectivity, just skip it
        if line_map_tmp[idx, :].sum() == 0:
            continue
        # Record the line segment
        else:
            for idx2 in np.where(line_map_tmp[idx, :] == 1)[0]:
                p1 = junctions[idx, :]  # HW format
                p2 = junctions[idx2, :]
                single_seg = np.concatenate([p1[None, ...], p2[None, ...]], axis=0)
                output_segments = np.concatenate(
                    (output_segments, single_seg[None, ...]), axis=0
                )

                # Update line_map
                line_map_tmp[idx, idx2] = 0
                line_map_tmp[idx2, idx] = 0

    return output_segments


# Taken from SOLD2
def convert_junc_predictions(predictions, grid_size, detect_thresh=1 / 65, topk=300):
    """Convert torch predictions to numpy arrays for evaluation."""
    # Convert to probability outputs first
    junc_prob = softmax(predictions.detach(), dim=1).cpu()
    junc_pred = junc_prob[:, :-1, :, :]

    junc_prob_np = junc_prob.numpy().transpose(0, 2, 3, 1)[:, :, :, :-1]
    junc_prob_np = np.sum(junc_prob_np, axis=-1)
    junc_pred_np = (
        pixel_shuffle(junc_pred, grid_size).cpu().numpy().transpose(0, 2, 3, 1)
    )
    junc_pred_np_nms = super_nms(junc_pred_np, grid_size, detect_thresh, topk)
    junc_pred_np = junc_pred_np.squeeze(-1)

    return {
        "junc_pred": junc_pred_np,
        "junc_pred_nms": junc_pred_np_nms,
        "junc_prob": junc_prob_np,
    }


# Taken from SOLD2
def super_nms(prob_predictions, dist_thresh, prob_thresh=0.01, top_k=0):
    """Non-maximum suppression adapted from SuperPoint."""
    # Iterate through batch dimension
    im_h = prob_predictions.shape[1]
    im_w = prob_predictions.shape[2]
    output_lst = []
    for i in range(prob_predictions.shape[0]):
        # print(i)
        prob_pred = prob_predictions[i, ...]
        # Filter the points using prob_thresh
        coord = np.where(prob_pred >= prob_thresh)  # HW format
        points = np.concatenate(
            (coord[0][..., None], coord[1][..., None]), axis=1
        )  # HW format

        # Get the probability score
        prob_score = prob_pred[points[:, 0], points[:, 1]]

        # Perform super nms
        # Modify the in_points to xy format (instead of HW format)
        in_points = np.concatenate(
            (coord[1][..., None], coord[0][..., None], prob_score), axis=1
        ).T
        keep_points_, keep_inds = nms_fast(in_points, im_h, im_w, dist_thresh)
        # Remember to flip outputs back to HW format
        keep_points = np.round(np.flip(keep_points_[:2, :], axis=0).T)
        keep_score = keep_points_[-1, :].T

        # Whether we only keep the topk value
        if (top_k > 0) or (top_k is None):
            k = min([keep_points.shape[0], top_k])
            keep_points = keep_points[:k, :]
            keep_score = keep_score[:k]

        # Re-compose the probability map
        output_map = np.zeros([im_h, im_w])
        output_map[
            keep_points[:, 0].astype(np.int), keep_points[:, 1].astype(np.int)
        ] = keep_score.squeeze()

        output_lst.append(output_map[None, ...])

    return np.concatenate(output_lst, axis=0)


# Taken from SOLD2
def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1,
    rest are zeros. Iterate through all the 1's and convert them to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundary.

    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinite distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode="constant")
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad : pt[1] + pad + 1, pt[0] - pad : pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds


def get_inliers_and_reproj_error(line_seg1, line_seg2, H, tol_px=5):
    # Warp back line_seg2
    warped_line_seg2 = warp_lines(line_seg2, H)

    # Compute the line distance
    dist = np.diag(get_orth_line_dist(line_seg1, warped_line_seg2))
    inliers = dist < tol_px
    reproj_error = 0 if np.sum(inliers) == 0 else dist[inliers].mean()
    return inliers, reproj_error


def estimate_homography(line_seg1, line_seg2, tol_px=5):
    """Estimate the homography relating two sets of lines.
    Args:
        line_seg1, line_seg2: the matching set of line segments.
        tol_px: inlier threshold in RANSAC.
    Returns:
        The estimated homography, mask of inliers, and reprojection error.
    """
    # To Account for optional libraries
    from homography_est import LineSegment, ransac_line_homography

    # Initialize the line segments C++ bindings
    lines1 = [LineSegment(l[0, [1, 0]], l[1, [1, 0]]) for l in line_seg1]
    lines2 = [LineSegment(l[0, [1, 0]], l[1, [1, 0]]) for l in line_seg2]

    # Estimate the homography with RANSAC
    inliers = []
    H = ransac_line_homography(lines1, lines2, tol_px, False, inliers)
    inliers, reproj_error = get_inliers_and_reproj_error(
        line_seg1, line_seg2, H, tol_px
    )
    return H, inliers, reproj_error


def H_estimation(line_seg1, line_seg2, H_gt, img_size, reproj_thresh=3, tol_px=5):
    """Given matching line segments from pairs of images, estimate
        a homography and compare it to the ground truth homography.
    Args:
        line_seg1, line_seg2: the matching set of line segments.
        H_gt: the ground truth homography relating the two images.
        img_size: the original image size.
        reproj_thresh: error threshold to determine if a homography is valid.
        tol_px: inlier threshold in RANSAC.
    Returns:
        The percentage of correctly estimated homographies.
    """
    # Estimate the homography
    H, inliers, reproj_error = estimate_homography(line_seg1, line_seg2, tol_px)

    # Compute the homography estimation error
    corners = np.array(
        [
            [0, 0],
            [0, img_size[1] - 1],
            [img_size[0] - 1, 0],
            [img_size[0] - 1, img_size[1] - 1],
        ],
        dtype=float,
    )
    warped_corners = warp_points(corners, H_gt)
    pred_corners = warp_points(warped_corners, H)
    error = np.linalg.norm(corners - pred_corners, axis=1).mean()
    return error < reproj_thresh, np.sum(inliers), reproj_error
