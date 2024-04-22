"""
Utility functions for JPLDD extractor
- ALIKED supporting methods
- Linesegments supporting methods
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax, pixel_shuffle


def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)


def simple_nms(scores: torch.Tensor, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""

    zeros = torch.zeros_like(scores)
    max_mask = scores == torch.nn.functional.max_pool2d(
        scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
    )

    for _ in range(2):
        supp_mask = (
                torch.nn.functional.max_pool2d(
                    max_mask.float(),
                    kernel_size=nms_radius * 2 + 1,
                    stride=1,
                    padding=nms_radius,
                )
                > 0
        )
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == torch.nn.functional.max_pool2d(
            supp_scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


class InputPadder(object):
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, h: int, w: int, divis_by: int = 8):
        self.ht = h
        self.wd = w
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, x: torch.Tensor):
        assert x.ndim == 4
        return F.pad(x, self._pad, mode="replicate")

    def unpad(self, x: torch.Tensor):
        assert x.ndim == 4
        ht = x.shape[-2]
        wd = x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]: c[1], c[2]: c[3]]


# Taken from SOLD2
def line_map_to_segments(junctions, line_map):
    """ Convert a line map to a Nx2x2 list of segments. """
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
                single_seg = np.concatenate([p1[None, ...], p2[None, ...]],
                                            axis=0)
                output_segments = np.concatenate(
                    (output_segments, single_seg[None, ...]), axis=0)

                # Update line_map
                line_map_tmp[idx, idx2] = 0
                line_map_tmp[idx2, idx] = 0

    return output_segments


# Taken from SOLD2
def convert_junc_predictions(predictions, grid_size,
                             detect_thresh=1 / 65, topk=300):
    """ Convert torch predictions to numpy arrays for evaluation. """
    # Convert to probability outputs first
    junc_prob = softmax(predictions.detach(), dim=1).cpu()
    junc_pred = junc_prob[:, :-1, :, :]

    junc_prob_np = junc_prob.numpy().transpose(0, 2, 3, 1)[:, :, :, :-1]
    junc_prob_np = np.sum(junc_prob_np, axis=-1)
    junc_pred_np = pixel_shuffle(
        junc_pred, grid_size).cpu().numpy().transpose(0, 2, 3, 1)
    junc_pred_np_nms = super_nms(junc_pred_np, grid_size, detect_thresh, topk)
    junc_pred_np = junc_pred_np.squeeze(-1)

    return {"junc_pred": junc_pred_np, "junc_pred_nms": junc_pred_np_nms,
            "junc_prob": junc_prob_np}


# Taken from SOLD2
def super_nms(prob_predictions, dist_thresh, prob_thresh=0.01, top_k=0):
    """ Non-maximum suppression adapted from SuperPoint. """
    # Iterate through batch dimension
    im_h = prob_predictions.shape[1]
    im_w = prob_predictions.shape[2]
    output_lst = []
    for i in range(prob_predictions.shape[0]):
        # print(i)
        prob_pred = prob_predictions[i, ...]
        # Filter the points using prob_thresh
        coord = np.where(prob_pred >= prob_thresh)  # HW format
        points = np.concatenate((coord[0][..., None], coord[1][..., None]),
                                axis=1)  # HW format

        # Get the probability score
        prob_score = prob_pred[points[:, 0], points[:, 1]]

        # Perform super nms
        # Modify the in_points to xy format (instead of HW format)
        in_points = np.concatenate((coord[1][..., None], coord[0][..., None],
                                    prob_score), axis=1).T
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
        output_map[keep_points[:, 0].astype(np.int),
        keep_points[:, 1].astype(np.int)] = keep_score.squeeze()

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
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
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
