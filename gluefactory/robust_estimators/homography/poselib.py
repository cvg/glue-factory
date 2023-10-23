import poselib
import torch
from omegaconf import OmegaConf

from ..base_estimator import BaseEstimator


class PoseLibHomographyEstimator(BaseEstimator):
    default_conf = {"ransac_th": 2.0, "options": {}}

    required_data_keys = ["m_kpts0", "m_kpts1"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        pts0, pts1 = data["m_kpts0"], data["m_kpts1"]
        M, info = poselib.estimate_homography(
            pts0.detach().cpu().numpy(),
            pts1.detach().cpu().numpy(),
            {
                "max_reproj_error": self.conf.ransac_th,
                **OmegaConf.to_container(self.conf.options),
            },
        )
        success = M is not None
        if not success:
            M = torch.eye(3, device=pts0.device, dtype=pts0.dtype)
            inl = torch.zeros_like(pts0[:, 0]).bool()
        else:
            M = torch.tensor(M).to(pts0)
            inl = torch.tensor(info["inliers"]).bool().to(pts0.device)

        estimation = {
            "success": success,
            "M_0to1": M,
            "inliers": inl,
        }

        return estimation
