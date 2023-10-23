import kornia
import torch

from ..base_model import BaseModel


class KorniaSIFT(BaseModel):
    default_conf = {
        "has_detector": True,
        "has_descriptor": True,
        "max_num_keypoints": -1,
        "detection_threshold": None,
        "rootsift": True,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        self.sift = kornia.feature.SIFTFeature(
            num_features=self.conf.max_num_keypoints, rootsift=self.conf.rootsift
        )
        self.set_initialized()

    def _forward(self, data):
        lafs, scores, descriptors = self.sift(data["image"])
        keypoints = kornia.feature.get_laf_center(lafs)
        scales = kornia.feature.get_laf_scale(lafs).squeeze(-1).squeeze(-1)
        oris = kornia.feature.get_laf_orientation(lafs).squeeze(-1)
        pred = {
            "keypoints": keypoints,  # @TODO: confirm keypoints are in corner convention
            "scales": scales,
            "oris": oris,
            "keypoint_scores": scores,
        }

        if self.conf.has_descriptor:
            pred["descriptors"] = descriptors

        pred = {k: pred[k].to(device=data["image"].device) for k in pred.keys()}

        pred["scales"] = pred["scales"]
        pred["oris"] = torch.deg2rad(pred["oris"])
        return pred

    def loss(self, pred, data):
        raise NotImplementedError
