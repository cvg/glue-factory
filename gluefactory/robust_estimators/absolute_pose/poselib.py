import poselib
import torch
from omegaconf import OmegaConf

from ...geometry import reconstruction
from ..base_estimator import BaseEstimator


class PoseLibAbsolutePoseEstimator(BaseEstimator):
    default_conf = {"ransac_th": 2.0, "options": {}}

    required_data_keys = ["p3d_w", "p2d_i", "camera"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        p2d_i, p3d_w = data["p2d_i"], data["p3d_w"]
        camera = data["camera"]
        c_t_w, info = poselib.estimate_absolute_pose(
            p2d_i.detach().numpy(),
            p3d_w.detach().numpy(),
            camera.to_cameradict(),
            {
                "max_reproj_error": self.conf.ransac_th,
                **OmegaConf.to_container(self.conf.options),
            },
        )
        success = c_t_w is not None
        if success:
            M = reconstruction.Pose.from_Rt(
                torch.tensor(c_t_w.R), torch.tensor(c_t_w.t)
            ).to(p2d_i)
        else:
            M = reconstruction.Pose.from_4x4mat(torch.eye(4)).to(p2d_i)

        estimation = {
            "success": success,
            "c_t_w": M,
            "inliers": torch.tensor(info.pop("inliers")).to(p2d_i),
            **info,
        }

        return estimation
