import cv2
import torch
import pytheia as pt

from ..base_estimator import BaseEstimator


class PytheiaHomographyEstimator(BaseEstimator):
    default_conf = {
        "ransac_th": 3.0,
    }

    required_data_keys = ["m_kpts0", "m_kpts1"]

    def _init(self, conf):
        self.params = pt.solvers.RansacParameters()
        self.params.use_mle = True 
        self.params.max_iterations = 5000
        self.params.failure_probability = 1-0.995

    def _forward(self, data):
        pts0, pts1 = data["m_kpts0"], data["m_kpts1"]

        try:
            pt.sfm.EstimateHomography(
                
            )
            success = M is not None
        except cv2.error:
            success = False
        if not success:
            M = torch.eye(3, device=pts0.device, dtype=pts0.dtype)
            inl = torch.zeros_like(pts0[:, 0]).bool()
        else:
            M = torch.tensor(M).to(pts0)
            inl = torch.tensor(mask).bool().to(pts0.device)

        return {
            "success": success,
            "M_0to1": M,
            "inliers": inl,
        }
