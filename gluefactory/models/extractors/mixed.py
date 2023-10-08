import torch.nn.functional as F
from omegaconf import OmegaConf

from .. import get_model
from ..base_model import BaseModel

to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class MixedExtractor(BaseModel):
    default_conf = {
        "detector": {"name": None},
        "descriptor": {"name": None},
        "interpolate_descriptors_from": None,  # field name
    }

    required_data_keys = ["image"]
    required_cache_keys = []

    def _init(self, conf):
        if conf.detector.name:
            self.detector = get_model(conf.detector.name)(to_ctr(conf.detector))
        else:
            self.required_data_keys += ["cache"]
            self.required_cache_keys += ["keypoints"]

        if conf.descriptor.name:
            self.descriptor = get_model(conf.descriptor.name)(to_ctr(conf.descriptor))
        else:
            self.required_data_keys += ["cache"]
            self.required_cache_keys += ["descriptors"]

    def _forward(self, data):
        if self.conf.detector.name:
            pred = self.detector(data)
        else:
            pred = data["cache"]
        if self.conf.detector.name:
            pred = {**pred, **self.descriptor({**pred, **data})}

        if self.conf.interpolate_descriptors_from:
            h, w = data["image"].shape[-2:]
            kpts = pred["keypoints"]
            pts = (kpts / kpts.new_tensor([[w, h]]) * 2 - 1)[:, None]
            pred["descriptors"] = (
                F.grid_sample(
                    pred[self.conf.interpolate_descriptors_from],
                    pts,
                    align_corners=False,
                    mode="bilinear",
                )
                .squeeze(-2)
                .transpose(-2, -1)
                .contiguous()
            )

        return pred

    def loss(self, pred, data):
        losses = {}
        metrics = {}
        total = 0

        for k in ["detector", "descriptor"]:
            apply = True
            if "apply_loss" in self.conf[k].keys():
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    losses_, metrics_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}
                total = losses_["total"] + total
        return {**losses, "total": total}, metrics
