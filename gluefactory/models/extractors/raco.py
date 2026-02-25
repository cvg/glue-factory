"""Wrapper for RaCo keypoint extractor."""

import torch

from raco.raco import RaCo as RaCo_

from ...utils.misc import sample_random_keypoints
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
        "ranker": False,
        "covariance_estimator": False,
        "add_random_keypoints": 0,
        "remove_borders": 0,
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
            remove_borders=conf.remove_borders,
        )
        self.set_initialized()

    def _forward(self, data):
        pred = self.model_(data)
        if self.conf.add_random_keypoints > 0:
            delta = self.conf.add_random_keypoints
            kpts = pred["keypoints"]  # (B, N, 2)
            B, dev = kpts.shape[0], kpts.device
            if "depth" in data:
                from .grid_extractor import _bbox_from_mask

                h, w = data["depth"].shape[-2:]
                wmin, hmin, wmax, hmax = _bbox_from_mask(data["depth"] > 0, h, w)
                bbox = torch.stack([wmin, hmin, wmax, hmax], dim=-1).to(dev)
                rand_kpts = sample_random_keypoints(delta, None, None, dev, bbox)
            else:
                rand_kpts = torch.stack(
                    [
                        sample_random_keypoints(
                            delta, data["transform"], data["original_image_size"], dev
                        )
                        for _ in range(B)
                    ]
                )
            pred["keypoints"] = torch.cat([kpts, rand_kpts], dim=1)
            if "keypoint_scores" in pred:
                pad = pred["keypoint_scores"].new_zeros(B, delta)
                pred["keypoint_scores"] = torch.cat(
                    [pred["keypoint_scores"], pad], dim=1
                )
        return pred

    def loss(self, pred, data):
        raise NotImplementedError
