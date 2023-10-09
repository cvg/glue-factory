import cv2
import numpy as np
import torch

from ...geometry.utils import from_homogeneous
from ...geometry.wrappers import Pose
from ..base_estimator import BaseEstimator


class OpenCVRelativePoseEstimator(BaseEstimator):
    default_conf = {
        "ransac_th": 0.5,
        "options": {"confidence": 0.99999, "method": "ransac"},
    }

    required_data_keys = ["m_kpts0", "m_kpts1", "camera0", "camera1"]

    def _init(self, conf):
        self.solver = {"ransac": cv2.RANSAC, "usac_magsac": cv2.USAC_MAGSAC}[
            self.conf.options.method
        ]

    def _forward(self, data):
        kpts0, kpts1 = data["m_kpts0"], data["m_kpts1"]
        camera0 = data["camera0"]
        camera1 = data["camera1"]
        M, inl = None, torch.zeros_like(kpts0[:, 0]).bool()

        if len(kpts0) >= 5:
            f_mean = torch.cat([camera0.f, camera1.f]).mean().item()
            norm_thresh = self.conf.ransac_th / f_mean

            pts0 = from_homogeneous(camera0.image2cam(kpts0)).cpu().detach().numpy()
            pts1 = from_homogeneous(camera1.image2cam(kpts1)).cpu().detach().numpy()

            E, mask = cv2.findEssentialMat(
                pts0,
                pts1,
                np.eye(3),
                threshold=norm_thresh,
                prob=self.conf.options.confidence,
                method=self.solver,
            )

            if E is not None:
                best_num_inliers = 0
                for _E in np.split(E, len(E) / 3):
                    n, R, t, _ = cv2.recoverPose(
                        _E, pts0, pts1, np.eye(3), 1e9, mask=mask
                    )
                    if n > best_num_inliers:
                        best_num_inliers = n
                        inl = torch.tensor(mask.ravel() > 0)
                        M = Pose.from_Rt(
                            torch.tensor(R).to(kpts0), torch.tensor(t[:, 0]).to(kpts0)
                        )

        estimation = {
            "success": M is not None,
            "M_0to1": M if M is not None else Pose.from_4x4mat(torch.eye(4).to(kpts0)),
            "inliers": inl.to(device=kpts0.device),
        }

        return estimation
