import cv2
import numpy as np
from gluefactory.utils.warp import warp_points


def select_k_best(kp, desc, k):
    """ Select the k most probable points (and strip their probability).
        points has shape [n_kp, 3] with last dim = kp score. """
    order = kp[:, 2].argsort()
    sorted_prob = kp[order, :2]
    sorted_desc = desc[order, :]
    start = min(k, kp.shape[0])
    return sorted_prob[-start:, :], sorted_desc[-start:, :]


def keep_shared_points(kp, desc, H, img_shape, keep_k_points=1000):
    """
    Compute a list of keypoints from the map, filter the list of points by keeping
    only the points that once mapped by H are still inside the shape of the map
    and keep at most 'keep_k_points' keypoints in the image. """
    
    def keep_true_keypoints(kp, desc, H, shape):
        """ Keep only the points whose warped coordinates
            by H are still inside shape. """
        warped_kp = warp_points(kp[:, [1, 0]], H)
        mask = ((warped_kp[:, 0] >= 0) & (warped_kp[:, 0] < shape[0]) &
                (warped_kp[:, 1] >= 0) & (warped_kp[:, 1] < shape[1]))
        return kp[mask, :], desc[mask, :]

    selected_kp, selected_desc = keep_true_keypoints(kp, desc, H, img_shape)
    selected_kp, selected_desc = select_k_best(selected_kp, selected_desc,
                                               keep_k_points)
    return selected_kp, selected_desc


def compute_matching_score(kp0, kp1, desc0, desc1, H, img_shape,
                           keep_k_points=1000, thresh=[1,3,5]):
    """ Compute the matching score between two sets
        of keypoints with associated descriptors. """
    # Select the top common points
    kp0_corr, desc0_corr = select_k_best(kp0, desc0, keep_k_points)
    kp1_corr, desc1_corr = select_k_best(kp1, desc1, keep_k_points)

    # Nearest neighbor matching
    match_dist = np.linalg.norm(desc0_corr[:, None] - desc1_corr[None],
                                axis=2)
    nearest0 = np.argmin(match_dist, axis=1)
    nearest1 = np.argmin(match_dist, axis=0)
    mutual = nearest1[nearest0] == np.arange(len(kp0_corr))
    m_kp0 = kp0_corr[mutual]
    m_kp1 = kp1_corr[nearest0[mutual]]

    # Matching score computation
    warped_m_kp1 = warp_points(m_kp1[:, [1, 0]], np.linalg.inv(H))[:, [1, 0]]
    correct_warped = np.all(
        (warped_m_kp1 >= 0) & (warped_m_kp1 <= (np.array(img_shape) - 1)),
        axis=-1)
    dist = np.linalg.norm(warped_m_kp1 - m_kp0, axis=1)

    matching_score = []
    for thr in thresh:
        matching_score.append(np.sum((dist < thr) * correct_warped)
                        / np.maximum(correct_warped.sum(), 1))

    return matching_score

def compute_homography(kp0, kp1, desc0, desc1, H_gt,
                       img_shape, keep_k_points=1000):
    """ Compute the homography between 2 sets of keypoints and descriptors. 
        Use the homography to compute the correctness metrics (1, 3, 5). """
    # Keeps only the points shared between the two views
    kp0_corr, desc0_corr = keep_shared_points(kp0, desc0, H_gt, img_shape,
                                              keep_k_points)
    kp1_corr, desc1_corr = keep_shared_points(kp1, desc1, np.linalg.inv(H_gt),
                                              img_shape, keep_k_points)

    if kp0_corr.shape[0] < 4 or kp1_corr.shape[0] < 4:
        return 0, 0, 0

    # Nearest neighbor matching
    match_dist = np.linalg.norm(desc0_corr[:, None] - desc1_corr[None],
                                axis=2)
    nearest0 = np.argmin(match_dist, axis=1)
    nearest1 = np.argmin(match_dist, axis=0)
    mutual = nearest1[nearest0] == np.arange(len(kp0_corr))
    m_kp0 = kp0_corr[mutual]
    m_kp1 = kp1_corr[nearest0[mutual]]

    if m_kp0.shape[0] < 4 or m_kp1.shape[0] < 4:
        return 0, 0, 0

    # Estimate the homography between the matches using RANSAC
    H, _ = cv2.findHomography(m_kp0, m_kp1, cv2.RANSAC, 3, maxIters=5000)

    if H is None:
        return 0, 0, 0

    # Compute the homography correctness
    corners = np.array([[0, 0],
                        [0, img_shape[1] - 1],
                        [img_shape[0] - 1, 0],
                        [img_shape[0] - 1, img_shape[1] - 1]])
    warped_corners = warp_points(corners, H)
    warped_corners = warp_points(warped_corners, np.linalg.inv(H_gt))

    mean_dist = np.linalg.norm(corners - warped_corners, axis=1).mean()
    correctness1 = float(mean_dist <= 1)
    correctness3 = float(mean_dist <= 3)
    correctness5 = float(mean_dist <= 5)
    return correctness1, correctness3, correctness5
