import cv2
import numpy as np
import torch


def warp_points(points, H):
    """Warp 2D points by an homography H."""
    n_points = points.shape[0]
    reproj_points = points.copy()[:, [1, 0]]
    reproj_points = np.concatenate([reproj_points, np.ones((n_points, 1))], axis=1)
    reproj_points = H.dot(reproj_points.transpose()).transpose()
    reproj_points = reproj_points[:, :2] / reproj_points[:, 2:]
    reproj_points = reproj_points[:, [1, 0]]
    return reproj_points


def warp_lines(lines, H):
    """Warp lines of the shape [N, 2, 2] by an homography H."""
    return warp_points(lines.reshape(-1, 2), H).reshape(-1, 2, 2)
