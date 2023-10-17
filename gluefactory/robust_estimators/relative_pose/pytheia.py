import numpy as np
import torch
import pytheia as pt

from ...geometry.utils import from_homogeneous
from ...geometry.wrappers import Pose
from ..base_estimator import BaseEstimator


class PytheiaRelativePoseEstimator(BaseEstimator):
    default_conf = {
        "ransac_th": 0.5,
    }

    required_data_keys = ["m_kpts0", "m_kpts1", "camera0", "camera1"]

    def _init(self, conf):
        self.params = pt.solvers.RansacParameters()
        self.params.use_lo = True 
        self.params.use_mle = True 
        self.params.max_iterations = 10000
        self.params.failure_probability = 1-0.99999
        self.params.min_iterations = 1000
        self.params.lo_start_iterations = 1

    def _forward(self, data):
        kpts0, kpts1 = data["m_kpts0"], data["m_kpts1"]
        camera0 = data["camera0"]
        camera1 = data["camera1"]

        M = None

        normalized_correspondences = []
        if len(kpts0) >= 5:
            f_mean = torch.cat([camera0.f, camera1.f]).mean().item()
            norm_thresh = self.conf.ransac_th / f_mean

            pts0 = from_homogeneous(camera0.image2cam(kpts0)).cpu().detach().numpy()
            pts1 = from_homogeneous(camera1.image2cam(kpts1)).cpu().detach().numpy()
        
        for p0, p1 in zip(pts0, pts1):
            normalized_correspondences.append(
                pt.matching.FeatureCorrespondence(
                    pt.sfm.Feature(p0), pt.sfm.Feature(p1)))

        self.params.error_thresh = norm_thresh

        success, rel_ori, summary = pt.sfm.EstimateRelativePose(
            self.params, pt.sfm.RansacType(2), normalized_correspondences)

        if success:
            M = Pose.from_Rt(
                torch.tensor(rel_ori.rotation).to(kpts0), 
                torch.tensor(-rel_ori.rotation@rel_ori.position).to(kpts0))

        estimation = {
            "success": M is not None,
            "M_0to1": M if M is not None else Pose.from_4x4mat(torch.eye(4).to(kpts0)),
            "inliers": torch.tensor(summary.inliers).to(kpts0),
        }

        return estimation