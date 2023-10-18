import cv2
import torch
import pytheia as pt

from ..base_estimator import BaseEstimator


class PytheiaHomographyEstimator(BaseEstimator):
    default_conf = {"ransac_th": 2.0, "options": {}}

    required_data_keys = ["m_kpts0", "m_kpts1"]

    def _init(self, conf):
        self.params = pt.solvers.RansacParameters()
        self.params.use_lo = True 
        self.params.use_mle = True 
        self.params.max_iterations = 10000
        self.params.failure_probability = 1-0.9999
        self.params.min_iterations = 1000
        self.params.lo_start_iterations = 10
        
    def _forward(self, data):
        pts0, pts1 = data["m_kpts0"], data["m_kpts1"]

        correspondences = []
        for p0, p1 in zip(pts0.numpy(), pts1.numpy()):
            correspondences.append(
                pt.matching.FeatureCorrespondence(
                    pt.sfm.Feature(p0), pt.sfm.Feature(p1)))

        self.params.error_thresh = self.conf.ransac_th

        if len(correspondences) < 4:
            success = False
        else:
            success, M, summary = pt.sfm.EstimateHomography(
                self.params, pt.sfm.RansacType(0), correspondences)

        if not success:
            M = torch.eye(3, device=pts0.device, dtype=pts0.dtype)
            inl = torch.zeros_like(pts0[:, 0]).bool()
        else:
            M = torch.tensor(M).to(pts0)
            inl = torch.tensor(summary.inliers).bool().to(pts0.device)

        return {
            "success": success,
            "M_0to1": M,
            "inliers": inl,
        }
