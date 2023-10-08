import pycolmap
import torch
from omegaconf import OmegaConf

from ...geometry.wrappers import Pose
from ..base_estimator import BaseEstimator


class PycolmapTwoViewEstimator(BaseEstimator):
    default_conf = {
        "ransac_th": 4.0,
        "options": {**pycolmap.TwoViewGeometryOptions().todict()},
    }

    required_data_keys = ["m_kpts0", "m_kpts1", "camera0", "camera1"]

    def _init(self, conf):
        opts = OmegaConf.to_container(conf.options)
        self.options = pycolmap.TwoViewGeometryOptions(opts)
        self.options.ransac.max_error = conf.ransac_th

    def _forward(self, data):
        pts0, pts1 = data["m_kpts0"], data["m_kpts1"]
        camera0 = data["camera0"]
        camera1 = data["camera1"]
        info = pycolmap.two_view_geometry_estimation(
            pts0.numpy(),
            pts1.numpy(),
            camera0.to_cameradict(),
            camera1.to_cameradict(),
            self.options,
        )
        success = info["success"]
        if success:
            R = pycolmap.qvec_to_rotmat(info["qvec"])
            t = info["tvec"]
            M = Pose.from_Rt(torch.tensor(R), torch.tensor(t)).to(pts0)
            inl = torch.tensor(info.pop("inliers")).to(pts0)
        else:
            M = Pose.from_4x4mat(torch.eye(4)).to(pts0)
            inl = torch.zeros_like(pts0[:, 0]).bool()

        estimation = {
            "success": success,
            "M_0to1": M,
            "inliers": inl,
            "type": str(
                info.get("configuration_type", pycolmap.TwoViewGeometry.UNDEFINED)
            ),
        }

        return estimation
