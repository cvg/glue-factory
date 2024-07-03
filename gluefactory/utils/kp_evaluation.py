import numpy as np

from gluefactory.utils.warp import warp_points

def compute_rep_loc_H(kp0, kp1, H, img_shape, keep_k_points=300, thresh=3):
    """ Compute the repeatability and localization error metrics between
        2 sets of keypoints. We compute it one way, by reprojecting kp1 to
        image 0, as it is fairer in the case of synthetic homographies.
        kp0 and kp1 are expected to be [n_kp, 3] with last dim = kp score.
    """
    def filter_keypoints(points, shape):
        """ Keep only the points whose coordinates are
            inside the dimensions of shape. """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &\
               (points[:, 1] >= 0) & (points[:, 1] < shape[1])
        return points[mask]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates
            by H are still inside shape. """
        warped_points = warp_points(points[:, [1, 0]], H)
        mask = ((warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &
                (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1]))
        return points[mask]

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their probability).
            points has shape [n_kp, 3] with last dim = kp score. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:]

    # Warp points and keep only the ones inside the image
    kp0_corr = keep_true_keypoints(kp0, H, img_shape)
    warped_kp1 = warp_points(kp1[:, [1, 0]], np.linalg.inv(H))[:, [1, 0]]
    warped_kp1 = np.concatenate([warped_kp1, kp1[:, 2:]], axis=1)
    warped_kp1 = filter_keypoints(warped_kp1, img_shape)

    # Keep only the keep_k_points best predictions
    kp0_corr = select_k_best(kp0_corr, keep_k_points)
    warped_kp1 = select_k_best(warped_kp1, keep_k_points)

    # Compute the repeatability and localization error
    N0 = len(kp0_corr)
    N1 = len(warped_kp1)
    dist = np.linalg.norm(kp0_corr[:, None] - warped_kp1[None], axis=2)
    count0 = 0
    count1 = 0
    le0 = 0
    le1 = 0
    if N1 != 0:
        min0 = np.min(dist, axis=1)
        correct0 = min0 <= thresh
        count0 = np.sum(correct0)
        le0 = min0[correct0].sum()
    if N0 != 0:
        min1 = np.min(dist, axis=0)
        correct1 = (min1 <= thresh)
        count1 = np.sum(correct1)
        le1 = min1[correct1].sum()
    if N0 + N1 > 0:
        repeatability = (count0 + count1) / (N0 + N1)
    else:
        repeatability = -1
    if count0 + count1 > 0:
        loc_err = (le0 + le1) / (count0 + count1)
    else:
        loc_err = -1

    return repeatability, loc_err
