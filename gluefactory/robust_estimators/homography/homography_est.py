import numpy as np
import torch
from homography_est import (
    LineSegment,
    ransac_line_homography,
    ransac_point_homography,
    ransac_point_line_homography,
)

from ...utils.tensor import batch_to_numpy
from ..base_estimator import BaseEstimator


def H_estimation_hybrid(kpts0=None, kpts1=None, lines0=None, lines1=None, tol_px=5):
    """Estimate a homography from points and lines with hybrid RANSAC.
    All features are expected in x-y convention
    """
    # Check that we have at least 4 features
    n_features = 0
    if kpts0 is not None:
        n_features += len(kpts0) + len(kpts1)
    if lines0 is not None:
        n_features += len(lines0) + len(lines1)
    if n_features < 4:
        return None

    if lines0 is None:
        # Point-only RANSAC
        H = ransac_point_homography(kpts0, kpts1, tol_px, False, [])
    elif kpts0 is None:
        # Line-only RANSAC
        ls0 = [LineSegment(line[0], line[1]) for line in lines0]
        ls1 = [LineSegment(line[0], line[1]) for line in lines1]
        H = ransac_line_homography(ls0, ls1, tol_px, False, [])
    else:
        # Point-lines RANSAC
        ls0 = [LineSegment(line[0], line[1]) for line in lines0]
        ls1 = [LineSegment(line[0], line[1]) for line in lines1]
        H = ransac_point_line_homography(kpts0, kpts1, ls0, ls1, tol_px, False, [], [])
    if np.abs(H[-1, -1]) > 1e-8:
        H /= H[-1, -1]
    return H


class PointLineHomographyEstimator(BaseEstimator):
    default_conf = {"ransac_th": 2.0, "options": {}}

    required_data_keys = ["m_kpts0", "m_kpts1", "m_lines0", "m_lines1"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        feat = data["m_kpts0"] if "m_kpts0" in data else data["m_lines0"]
        data = batch_to_numpy(data)
        m_features = {
            "kpts0": data["m_kpts1"] if "m_kpts1" in data else None,
            "kpts1": data["m_kpts0"] if "m_kpts0" in data else None,
            "lines0": data["m_lines1"] if "m_lines1" in data else None,
            "lines1": data["m_lines0"] if "m_lines0" in data else None,
        }
        M = H_estimation_hybrid(**m_features, tol_px=self.conf.ransac_th)
        success = M is not None
        if not success:
            M = torch.eye(3, device=feat.device, dtype=feat.dtype)
        else:
            M = torch.from_numpy(M).to(feat)

        estimation = {
            "success": success,
            "M_0to1": M,
        }

        return estimation
