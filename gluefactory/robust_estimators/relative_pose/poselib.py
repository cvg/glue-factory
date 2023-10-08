import poselib
import torch
from omegaconf import OmegaConf

from ...geometry.wrappers import Pose
from ..base_estimator import BaseEstimator


class PoseLibRelativePoseEstimator(BaseEstimator):
    default_conf = {"ransac_th": 2.0, "options": {}}

    required_data_keys = ["m_kpts0", "m_kpts1", "camera0", "camera1"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        pts0, pts1 = data["m_kpts0"], data["m_kpts1"]
        camera0 = data["camera0"]
        camera1 = data["camera1"]
        M, info = poselib.estimate_relative_pose(
            pts0.numpy(),
            pts1.numpy(),
            camera0.to_cameradict(),
            camera1.to_cameradict(),
            {
                "max_epipolar_error": self.conf.ransac_th,
                **OmegaConf.to_container(self.conf.options),
            },
        )
        success = M is not None
        if success:
            M = Pose.from_Rt(torch.tensor(M.R), torch.tensor(M.t)).to(pts0)
        else:
            M = Pose.from_4x4mat(torch.eye(4)).to(pts0)

        estimation = {
            "success": success,
            "M_0to1": M,
            "inliers": torch.tensor(info.pop("inliers")).to(pts0),
            **info,
        }

        return estimation
