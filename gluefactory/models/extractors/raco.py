"""Wrapper for RaCo keypoint extractor."""

from raco.raco import RaCo as RaCo_

from ...utils import misc
from ..base_model import BaseModel


class RaCo(BaseModel):
    default_conf = {
        "name": "raco",
        "weights": "https://github.com/cvg/RaCo/releases/download/v1.0.0/raco.pth",
        "max_num_keypoints": 2048,
        "nms_radius": 3,
        "subpixel_sampling": True,
        "subpixel_temp": 0.5,
        "detection_threshold": -1,
        "ranker": True,
        "covariance_estimator": True,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        self.model_ = RaCo_(
            max_num_keypoints=conf.max_num_keypoints,
            nms_radius=conf.nms_radius,
            subpixel_sampling=conf.subpixel_sampling,
            subpixel_temp=conf.subpixel_temp,
            detection_threshold=conf.detection_threshold,
            ranker=conf.ranker,
            covariance_estimator=conf.covariance_estimator,
        )
        self.set_initialized()

    def _forward(self, data):
        pred = self.model_(data)
        return pred

    def loss(self, pred, data):
        raise NotImplementedError
