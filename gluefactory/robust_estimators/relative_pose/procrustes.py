from ...geometry import depth, epipolar
from ..base_estimator import BaseEstimator


class ProcrustesRelativePoseEstimator(BaseEstimator):
    default_conf = {"ransac_th": 2.0, "options": {}}

    required_data_keys = ["m_kpts0", "m_kpts1", "camera0", "camera1"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        pts0, pts1 = data["m_kpts0"], data["m_kpts1"]
        camera0 = data["camera0"]
        camera1 = data["camera1"]

        depth0 = data["m_depth0"]
        depth1 = data["m_depth1"]

        xyz0 = camera0.image2cam(pts0) * depth0[..., None]
        xyz1 = camera1.image2cam(pts1) * depth1[..., None]

        c0_t_c1, scale, _ = depth.align_pointclouds(
            xyz0, xyz1, weights=data.get("m_scores")
        )

        c1_t_c0 = c0_t_c1.inv() if c0_t_c1 is not None else None

        c1_F_c0 = epipolar.E_to_F(camera0, camera1, c1_t_c0.E)
        inliers = (
            epipolar.sym_epipolar_distance(pts0, pts1, c1_F_c0, squared=False)
            < self.conf.ransac_th
        )

        estimation = {
            "success": c0_t_c1 is not None,
            "M_0to1": c1_t_c0,
            "inliers": inliers,
            "scale": scale,
        }

        return estimation
