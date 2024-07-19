import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from gluefactory.datasets.homographies_deeplsd import warp_lines
from gluefactory.models.lines.line_distances import (
    angular_distance,
    get_area_line_dist,
    get_lip_line_dist,
    get_orth_line_dist,
    get_structural_line_dist,
    overlap_distance_sym,
)
from gluefactory.models.lines.line_utils import get_common_lines

NUM_LINES_THRESHOLDS = [10, 25, 50, 100, 300]
PIXEL_THRESHOLDS = [1, 2, 3, 4, 5]

### Line matching


def get_rep_and_loc_error(
    lines: torch.Tensor,
    warped_lines: torch.Tensor,
    Hs: torch.Tensor,
    img_size: tuple[float, float],
    num_lines_thres: list[int] = NUM_LINES_THRESHOLDS,
    pixel_thres: list[int] = PIXEL_THRESHOLDS,
) -> tuple[np.ndarray, np.ndarray]:
    (struct_rep, struct_loc_error, num_lines) = [], [], []
    for i in range(len(lines)):
        cur_lines = lines[i]
        cur_warped_lines = warped_lines[i]
        num_lines.append((len(cur_lines) + len(cur_warped_lines)) / 2)
        segs1, segs2, matched_idx1, matched_idx2, distances = match_segments_1_to_1(
            cur_lines,
            cur_warped_lines,
            Hs[i],
            img_size,
            line_dist="struct",
            dist_thresh=5,
        )
        if len(matched_idx1) == 0:
            struct_rep.append([0] * len(pixel_thres))
            struct_loc_error.append([0] * len(num_lines_thres))
        else:
            struct_rep.append(
                compute_repeatability(
                    segs1,
                    segs2,
                    matched_idx1,
                    matched_idx2,
                    distances,
                    pixel_thres,
                    rep_type="num",
                )
            )
            struct_loc_error.append(compute_loc_error(distances, num_lines_thres))
    num_lines = np.mean(num_lines)
    repeatability = np.mean(np.stack(struct_rep, axis=0), axis=0)
    loc_error = np.mean(np.stack(struct_loc_error, axis=0), axis=0)
    return repeatability, loc_error


def match_segments_1_to_1(
    line_seg1: torch.Tensor,
    line_seg2: torch.Tensor,
    H: torch.Tensor,
    img_size,
    line_dist="area",
    angular_th=(30 * np.pi / 180),
    overlap_th=0.5,
    dist_thresh=5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match segs1 and segs2 1-to-1, minimizing the chosen line distance.
    Ensure a minimum overlap and maximum angle difference."""
    HIGH_VALUE = 100000

    # Gather lines in common between the two views and warp lines1 to img0
    segs1, segs2 = get_common_lines(line_seg1, line_seg2, H, img_size)

    if len(segs1) == 0 or len(segs2) == 0:
        return (
            np.empty((0, 2, 2)),
            np.empty((0, 2, 2)),
            np.empty(0),
            np.empty(0),
            np.empty(0),
        )

    if line_dist == "struct":
        full_distance_matrix = get_structural_line_dist(segs1, segs2)
    elif line_dist == "orth":
        full_distance_matrix = get_orth_line_dist(segs1, segs2, overlap_th)
    elif line_dist == "area":
        full_distance_matrix = get_area_line_dist(segs1, segs2)
    elif line_dist == "lip":
        full_distance_matrix = get_lip_line_dist(segs1, segs2)
    else:
        raise ValueError("Unknown line distance: " + line_dist)

    # Ignore matches with a too high distance
    full_distance_matrix[full_distance_matrix > dist_thresh] = HIGH_VALUE

    # We require that they have a similar angle
    angular_dist = angular_distance(segs1, segs2)
    full_distance_matrix[angular_dist > angular_th] = HIGH_VALUE

    # We require a minimum overlap between the segments
    overlap_dist = overlap_distance_sym(segs1, segs2)
    full_distance_matrix[overlap_dist <= overlap_th] = HIGH_VALUE

    # Enforce the 1-to-1 assignation
    matched_idx1, matched_idx2 = linear_sum_assignment(
        full_distance_matrix.cpu().numpy()
    )
    matched_idx1, matched_idx2 = torch.tensor(
        matched_idx1, device=full_distance_matrix.device
    ), torch.tensor(matched_idx2, device=full_distance_matrix.device)
    # Remove invalid matches
    distances = full_distance_matrix[matched_idx1, matched_idx2]
    valid = distances < HIGH_VALUE
    matched_idx1, matched_idx2 = matched_idx1[valid], matched_idx2[valid]
    distances = distances[valid]

    # Sort by increasing distance
    sort_indices = torch.argsort(distances)
    matched_idx1 = matched_idx1[sort_indices]
    matched_idx2 = matched_idx2[sort_indices]
    distances = distances[sort_indices]

    return segs1, segs2, matched_idx1, matched_idx2, distances


### Metrics computation


def compute_repeatability(
    segs1: torch.Tensor,
    segs2: torch.Tensor,
    matched_idx1: torch.Tensor,
    matched_idx2: torch.Tensor,
    distances: torch.Tensor,
    thresholds: list[int],
    rep_type="num",
) -> list[np.ndarray]:
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
        if rep_type == "num":
            rep = (torch.sum(correct) / min(n1, n2)).cpu().numpy()
        elif rep_type == "length":
            len1 = torch.linalg.norm(segs1[:, 0] - segs1[:, 1], axis=1)
            len2 = torch.linalg.norm(segs2[:, 0] - segs2[:, 1], axis=1)
            matched_len1 = len1[matched_idx1[correct]]
            matched_len2 = len2[matched_idx2[correct]]
            rep = (matched_len1.sum() + matched_len2.sum()) / (
                len1.sum() + len2.sum()
            ).cpu().numpy()
        else:
            raise ValueError("Unknown repeatability type: " + rep_type)
        reps.append(rep)
    return reps


def compute_loc_error(
    distances: torch.Tensor, thresholds: list[int]
) -> list[np.ndarray]:
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
        valid_distances = distances[:t]
        if len(valid_distances) == 0:
            loc_errors.append(0)
        else:
            loc_errors.append(torch.mean(valid_distances).cpu().numpy())
    return loc_errors


def get_inliers_and_reproj_error(line_seg1, line_seg2, H, tol_px=5):
    # Warp back line_seg2
    warped_line_seg2 = warp_lines(line_seg2, H)

    # Compute the line distance
    dist = np.diag(get_orth_line_dist(line_seg1, warped_line_seg2))
    inliers = dist < tol_px
    reproj_error = 0 if np.sum(inliers) == 0 else dist[inliers].mean()
    return inliers, reproj_error


### Vanishing point estimation


def dist_lines_vp(lines, vp):
    """Estimate the distance between a set of lines
        and VPs in homogeneous format.
    Args:
        lines: [N, 2, 2] array in ij convention.
        vp: [M, 3] array in homogeneous format.
    Returns:
        An [N, M] distance matrix of each line to each VP.
    """
    # Center of the lines
    centers = (lines[:, 0] + lines[:, 1]) / 2
    centers = np.concatenate([centers[:, [1, 0]], np.ones_like(centers[:, :1])], axis=1)

    # Line passing through the VP and the center of the lines
    # l = cross(center, vp)
    # l is [N, M, 3]
    line_vp = np.cross(centers[:, None], vp[None])
    line_vp_norm = np.linalg.norm(line_vp[:, :, :2], axis=2)

    # Orthogonal distance of the lines to l
    endpts = np.concatenate(
        [lines[:, 0][:, [1, 0]], np.ones_like(lines[:, 0, :1])], axis=1
    )
    orth_dist = np.abs(np.sum(endpts[:, None] * line_vp, axis=2)) / line_vp_norm
    return orth_dist


def vp_consistency_check(gt_lines, line_clusters, vps, tol=3):
    """Given a set of GT lines, their GT VP clustering and estimated VPs,
    assign each cluster of line to a unique VP and compute the ratio
    of lines consistent with the assigned VP.
    Return a list of consistency, for each tolerance threshold in tol."""
    if not isinstance(tol, list):
        tol = [tol]

    # Compute the distance from all lines to all VPs
    distances = dist_lines_vp(gt_lines, vps)

    # Compute the average score for each cluster of lines
    num_vps = len(vps)
    num_lines = len(gt_lines)
    cluster_labels = np.unique(line_clusters)
    num_clusters = len(cluster_labels)
    avg_scores = np.zeros((num_clusters, num_vps))
    for i in range(num_clusters):
        curr_cluster = line_clusters == cluster_labels[i]
        avg_scores[i] = np.mean(distances[curr_cluster], axis=0)

    # Find the optimal assignment of clusters and VPs
    cluster_assignment, vp_assignment = linear_sum_assignment(avg_scores)

    # Compute the number of consistent lines within each cluster
    consistency_check = []
    for t in tol:
        num_consistent = 0
        for cl, vp in zip(cluster_assignment, vp_assignment):
            num_consistent += np.sum(
                distances[line_clusters == cluster_labels[cl]][:, vp] < t
            )
        consistency_check.append(num_consistent / num_lines)

    return consistency_check


def unproject_vp_to_world(vp, K):
    """Convert the VPs from homogenous format in the image plane
    to world direction."""
    proj_vp = (np.linalg.inv(K) @ vp.T).T
    proj_vp[:, 1] *= -1
    proj_vp /= np.linalg.norm(proj_vp, axis=1, keepdims=True)
    return proj_vp


def get_vp_error(gt_vp, pred_vp, K, max_err=10.0):
    """Compute the angular error between the predicted and GT VPs in 3D.
    The GT VPs are expected in 3D and unit normalized,
    but the predicted ones are in homogeneous format in the image."""
    # Unproject the predicted VP to world coordinates
    pred_vp_3d = pred_vp.copy()
    finite = np.abs(pred_vp_3d[:, 2]) > 1e-5
    pred_vp_3d[finite] /= pred_vp_3d[:, 2:][finite]
    pred_vp_3d = unproject_vp_to_world(pred_vp_3d, K)

    # Compute the pairwise cosine distances
    vp_dist = np.abs(np.einsum("nd,md->nm", gt_vp, pred_vp_3d))

    # Find the optimal assignment
    gt_idx, pred_idx = linear_sum_assignment(vp_dist, maximize=True)
    num_found = len(gt_idx)
    num_vp = len(gt_vp)

    # Get the error in degrees
    error = np.clip(vp_dist[gt_idx, pred_idx], 0, 1)
    error = np.arccos(error) * 180 / np.pi

    # Clip to a maximum error of 10 degrees and penalize missing VPs
    error = np.clip(error, 0, max_err)
    error = np.concatenate([error, max_err * np.ones(num_vp - num_found)])
    return np.mean(error)


def get_vp_detection_ratio(gt_vp, pred_vp, K, thresholds):
    """Compute the angular error between the predicted and GT VPs in 3D.
    The GT VPs are expected in 3D and unit normalized,
    but the predicted ones are in homogeneous format in the image.
    Count how many correct VPs are obtained for each error threshold."""
    # Unproject the predicted VP to world coordinates
    pred_vp_3d = pred_vp.copy()
    finite = np.abs(pred_vp_3d[:, 2]) > 1e-5
    pred_vp_3d[finite] /= pred_vp_3d[:, 2:][finite]
    pred_vp_3d = unproject_vp_to_world(pred_vp_3d, K)

    # Compute the pairwise cosine distances
    vp_dist = np.abs(np.einsum("nd,md->nm", gt_vp, pred_vp_3d))

    # Find the optimal assignment
    gt_idx, pred_idx = linear_sum_assignment(vp_dist, maximize=True)

    # Get the accuracy in degrees
    num_gt_vp = len(gt_vp)
    accuracy = np.clip(vp_dist[gt_idx, pred_idx], 0, 1)
    accuracy = np.arccos(accuracy) * 180 / np.pi
    scores = [np.sum(accuracy < t) / num_gt_vp for t in thresholds]

    return scores


def get_recall_AUC(gt_vp, pred_vp, K):
    """Compute the angular error between the predicted and GT VPs in 3D,
    compute the recall for different error thresholds, and compute the AUC.
    The GT VPs are expected in 3D and unit normalized,
    but the predicted ones are in homogeneous format in the image."""
    # Unproject the predicted VP to world coordinates
    pred_vp_3d = pred_vp.copy()
    finite = np.abs(pred_vp_3d[:, 2]) > 1e-5
    pred_vp_3d[finite] /= pred_vp_3d[:, 2:][finite]
    pred_vp_3d = unproject_vp_to_world(pred_vp_3d, K)

    # Compute the pairwise cosine distances
    vp_dist = np.abs(np.einsum("nd,md->nm", gt_vp, pred_vp_3d))

    # Find the optimal assignment
    gt_idx, pred_idx = linear_sum_assignment(vp_dist, maximize=True)
    num_vp = len(gt_vp)

    # Get the accuracy in degrees
    accuracy = np.clip(vp_dist[gt_idx, pred_idx], 0, 1)
    accuracy = np.arccos(accuracy) * 180 / np.pi

    # Compute the recall at various error thresholds
    step = 0.5
    error_thresholds = np.arange(0, 10.1, step=step)
    recalls = [np.sum(accuracy < t) / num_vp for t in error_thresholds]

    # Compute the AUC
    auc = np.sum(recalls) * step

    return recalls, auc
