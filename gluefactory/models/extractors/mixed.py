from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from .. import get_model
from ..base_model import BaseModel

to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class LazyMLP(nn.Sequential):
    def __init__(self, dims: int, norm: bool = True):
        layers = []
        norm_cls = nn.LayerNorm if norm else nn.Identity
        for i, dim in enumerate(dims):
            layers.append(nn.LazyLinear(dim))
            if i < len(dims) - 1:
                layers.append(norm_cls(dim))
                layers.append(nn.ReLU())
        super().__init__(*layers)


class MixedExtractor(BaseModel):
    default_conf = {
        "detector": {"name": None},
        "descriptor": {"name": None},
        "interpolate_descriptors_from": None,  # field name, str or list
        "fusion_mlp": None,
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

        self.interpolate_descriptors_from = conf.interpolate_descriptors_from
        if isinstance(self.interpolate_descriptors_from, str):
            self.interpolate_descriptors_from = [self.interpolate_descriptors_from]
            self.fusion = lambda x: x[0]
        elif isinstance(self.interpolate_descriptors_from, Sequence):
            if len(self.interpolate_descriptors_from) > 1:
                assert self.conf.fusion_mlp is not None
                self.fusion_mlp = LazyMLP(self.conf.fusion_mlp, norm=True)
                self.fusion = lambda x: self.fusion_mlp(torch.cat(x, dim=-1))
            else:
                self.fusion = lambda x: x[0]

    def _forward(self, data):
        if self.conf.detector.name:
            pred = self.detector(data)
        else:
            pred = data["cache"]
        if self.conf.detector.name:
            pred = {**pred, **self.descriptor({**pred, **data})}

        if self.interpolate_descriptors_from:
            h, w = data["image"].shape[-2:]
            kpts = pred["keypoints"].clone()
            kpts[..., 0] = kpts[..., 0] * 2 / w - 1
            kpts[..., 1] = kpts[..., 1] * 2 / h - 1

            kpts = kpts[:, None]

            all_descriptors = [
                self.interpolate_descriptors(pred[fmap_key], kpts)
                for fmap_key in self.interpolate_descriptors_from
            ]
            pred["descriptors"] = self.fusion(all_descriptors)

        return pred

    def interpolate_descriptors(self, fmap, kpts):
        return (
            F.grid_sample(
                fmap,
                kpts,
                align_corners=False,
                mode="bilinear",
            )
            .squeeze(-2)
            .transpose(-2, -1)
            .contiguous()
        )

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
