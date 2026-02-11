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
        "refiner": {"name": None},  # Always runs
        "interpolate_descriptors_from": None,  # field name, str or list
        "fusion_mlp": None,
        "allow_no_detect": False,
    }

    required_data_keys = ["image"]
    required_cache_keys = []

    def _init(self, conf):
        if conf.detector.name:
            self.detector = get_model(conf.detector.name)(to_ctr(conf.detector))
        if conf.descriptor.name:
            self.descriptor = get_model(conf.descriptor.name)(to_ctr(conf.descriptor))
        if conf.refiner.name:
            self.refiner = get_model(conf.refiner.name)(to_ctr(conf.refiner))

        self.interpolate_descriptors_from = conf.interpolate_descriptors_from
        if isinstance(self.interpolate_descriptors_from, str):
            self.interpolate_descriptors_from = [self.interpolate_descriptors_from]
            self.fusion = lambda x: x[0]
        elif isinstance(self.interpolate_descriptors_from, Sequence):
            if (
                len(self.interpolate_descriptors_from) > 1
                and conf.fusion_mlp is not None
            ):
                self.fusion_mlp = LazyMLP(self.conf.fusion_mlp, norm=True)
                self.fusion = lambda x: self.fusion_mlp(torch.cat(x, dim=-1))
            else:
                self.fusion = lambda x: sum(x) / len(x)

    def _forward(self, data):
        skip_detect = len(data.get("cache", {})) > 0 and self.conf.allow_no_detect
        if self.conf.detector.name and not skip_detect:
            pred = self.detector(data)
        else:
            pred = data["cache"]
        if self.conf.descriptor.name:
            pred = {**pred, **self.descriptor({**pred, **data})}

        if self.conf.refiner.name:
            pred = {**pred, **self.refiner({**pred, **data})}

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
        if fmap.ndim == 3 and fmap.shape[-2] == kpts.shape[-2]:
            # Already interpolated
            return fmap
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

        for k in ["detector", "descriptor", "refiner"]:
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

    def compile(self, *args, **kwargs) -> BaseModel:
        if self.conf.compile:
            return super().compile(*args, **kwargs)

        for k in ["detector", "descriptor", "refiner"]:
            if self.conf[k].name:
                setattr(self, k, getattr(self, k).compile(*args, **kwargs))
        return self
