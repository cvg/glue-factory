import numpy as np
from gluefactory.datasets.homographies_deeplsd import sample_homography
from gluefactory.models.extractors.jpldd.metrics_lines import match_segments_1_to_1,compute_repeatability
import torch
from kornia.geometry.transform import warp_perspective

default_H_params = {
    'translation': True,
    'rotation': True,
    'scaling': True,
    'perspective': True,
    'scaling_amplitude': 0.2,
    'perspective_amplitude_x': 0.2,
    'perspective_amplitude_y': 0.2,
    'patch_ratio': 0.85,
    'max_angle': 1.57,
    'allow_artifacts': True
}


def compute_tp_fp(data: np.array, pred: np.array, prob_tresh=0.3, distance_thresh=5, simplified=False, max_detections=2000):
    """
    Compute the true and false positive rates.

    :param data:  array of actual data (i.e. the ground truth)
    :param pred: array of predictions (i.e.)
    """
    # Read data
    gt = np.where(data > prob_tresh)
    gt = np.stack([gt[0], gt[1]], axis=-1)
    n_gt = len(gt)

    # Filter out predictions with near-zero probability
    flat_indices = np.argpartition(pred,-max_detections,axis=None)[-max_detections:]
    mask = np.unravel_index(flat_indices,pred.shape)
    probs = pred[mask]
    pred = np.array(mask).T

    # When several detections match the same ground truth point, only pick
    # the one with the highest score  (the others are false positive)
    sort_idx = np.argsort(probs)[::-1]
    
    probs = probs[sort_idx]
    pred = pred[sort_idx]

    diff = np.expand_dims(pred, axis=1) - np.expand_dims(gt, axis=0)
    dist = np.linalg.norm(diff, axis=-1)
    matches = np.less_equal(dist, distance_thresh)

    tp = []
    matched = np.zeros(len(gt))
    for m in matches:
        correct = np.any(m)
        if correct:
            gt_idx = np.argmax(m)
            tp.append(not matched[gt_idx])
            matched[gt_idx] = 1
        else:
            tp.append(False)
    tp = np.array(tp, bool)
    if simplified:
        tp = np.any(matches, axis=1)  # keeps multiple matches for the same gt point
        n_gt = np.sum(np.minimum(np.sum(matches, axis=0), 1))  # buggy
    fp = np.logical_not(tp)
    return tp, fp, probs, n_gt


def compute_pr(data: np.array, pred: np.array) -> tuple[np.array, np.array, np.array]:
    """
    Compute precision and recall.

    :param data: (Batched) array of actual data (i.e. the ground truth)
    :param pred: (Batched) array of predictions (i.e.)
    """
    # Gather TP and FP for all files
    tp, fp, prob, n_gt = [], [], [], 0
    for i in range(len(data)):
        t, f, p, n = compute_tp_fp(data[i], pred[i])
        tp.append(t)
        fp.append(f)
        prob.append(p)
        n_gt += n
    tp = np.concatenate(tp)
    fp = np.concatenate(fp)
    prob = np.concatenate(prob)

    # Sort in descending order of confidence
    sort_idx = np.argsort(prob)[::-1]
    tp = tp[sort_idx]
    fp = fp[sort_idx]
    prob = prob[sort_idx]

    # Cumulative
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = div0(tp_cum, n_gt)
    precision = div0(tp_cum, tp_cum + fp_cum)
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return precision, recall, prob


def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        idx = ~np.isfinite(c)
        c[idx] = np.where(a[idx] == 0, 1, 0)  # -inf inf NaN
    return c


def compute_loc_error(data: np.array, pred: np.array, prob_thresh=0.3, distance_thresh=5, max_detections=2000):
    """
    Compute the localization error.
    :param data: (Batched) array of actual data (i.e. the ground truth)
    :param pred: (Batched) array of predictions (i.e.)
    """

    def loc_error_per_image(single_data, pred_prob):
        # Read data
        gt = np.where(single_data > prob_thresh)
        gt = np.stack([gt[0], gt[1]], axis=-1)

        # Filter out predictions
        flat_indices = np.argpartition(pred_prob,-max_detections,axis=None)[-max_detections:]
        mask = np.unravel_index(flat_indices,pred_prob.shape)
        pred = np.array(mask).T
        pred_prob = pred_prob[mask]

        sort_idx = np.argsort(pred_prob)[::-1]
        pred = pred[sort_idx]
        pred_prob = pred_prob[sort_idx]

        if not len(gt) or not len(pred):
            return []

        diff = np.expand_dims(pred, axis=1) - np.expand_dims(gt, axis=0)
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.min(dist, axis=1)
        correct_dist = dist[np.less_equal(dist, distance_thresh)]
        return correct_dist

    error = []
    for i in range(len(data)):
        error.append(loc_error_per_image(data[i], pred[i]))
    errors = np.concatenate(error)
    return np.mean(errors) if len(errors) > 0 else float(distance_thresh)


def compute_repeatability(pred: np.array, warped_pred: np.array, Hs: np.array,
                            max_detections=2000,keep_k_points=300,distance_thresh=5, verbose=False):
    """
    Compute the repeatability. The experiment must contain in its output the prediction
    on 2 images, an original image and a warped version of it, plus the homography
    linking the 2 images.
    """

    def warp_keypoints(keypoints, H):
        num_points = keypoints.shape[0]
        homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                            axis=1)
        warped_points = np.dot(homogeneous_points, np.transpose(H))
        return warped_points[:, :2] / warped_points[:, 2:]

    def filter_keypoints(points, shape):
        """ Keep only the points whose coordinates are
        inside the dimensions of shape. """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) & \
               (points[:, 1] >= 0) & (points[:, 1] < shape[1])
        return points[mask, :]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H
        are still inside shape. """
        warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) & \
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :]

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    repeatability = []
    N1s = []
    N2s = []
    for i in range(len(pred)):
        cur_pred = pred[i]
        cur_warped_pred = warped_pred[i]
        shape = (cur_pred.shape[0],cur_pred.shape[1])
        # Filter out predictions
        flat_indices = np.argpartition(cur_pred,-max_detections,axis=None)[-max_detections:]
        keypoints = np.unravel_index(flat_indices,cur_pred.shape)
        prob = cur_pred[keypoints[0], keypoints[1]]
        keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
        flat_indices = np.argpartition(cur_warped_pred,-max_detections,axis=None)[-max_detections:]
        warped_keypoints = np.unravel_index(flat_indices,cur_warped_pred.shape)
        warped_prob = cur_warped_pred[warped_keypoints[0], warped_keypoints[1]]
        warped_keypoints = np.stack([warped_keypoints[0],
                                     warped_keypoints[1],
                                     warped_prob], axis=-1)
        H_np = Hs[i].cpu().numpy()
        warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H_np),
                                               shape)

        # Warp the original keypoints with the true homography
        true_warped_keypoints = warp_keypoints(keypoints[:, [1, 0]], H_np)
        true_warped_keypoints = np.stack([true_warped_keypoints[:, 1],
                                          true_warped_keypoints[:, 0],
                                          prob], axis=-1)
        true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)

        # Keep only the keep_k_points best predictions
        warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
        true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

        # Compute the repeatability
        N1 = true_warped_keypoints.shape[0]
        N2 = warped_keypoints.shape[0]
        N1s.append(N1)
        N2s.append(N2)
        true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
        warped_keypoints = np.expand_dims(warped_keypoints, 0)
        # shapes are broadcasted to N1 x N2 x 2:
        norm = np.linalg.norm(true_warped_keypoints - warped_keypoints,
                              ord=None, axis=2)
        count1 = 0
        count2 = 0
        if N2 != 0:
            min1 = np.min(norm, axis=1)
            count1 = np.sum(min1 <= distance_thresh)
        if N1 != 0:
            min2 = np.min(norm, axis=0)
            count2 = np.sum(min2 <= distance_thresh)
        if N1 + N2 > 0:
            repeatability.append((count1 + count2) / (N1 + N2))

    if verbose:
        print("Average number of points in the first image: " + str(np.mean(N1s)))
        print("Average number of points in the second image: " + str(np.mean(N2s)))
    return np.mean(repeatability) if len(repeatability) > 0 else 0.0
