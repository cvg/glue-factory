"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

import torch
from omegaconf import OmegaConf

from ..utils import misc
from . import get_model
from .base_model import BaseModel

to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class TwoViewPipeline(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "matcher": {"name": None},
        "filter": {"name": None},
        "solver": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
        "from_triplet": False,
    }
    required_data_keys = ["view0", "view1"]
    strict_conf = False  # need to pass new confs to children models
    components = [
        "extractor",
        "matcher",
        "filter",
        "solver",
        "ground_truth",
    ]

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))

        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))

        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(
                to_ctr(conf.ground_truth)
            )

    def _precompute_covisible_bboxes(self, data, num_views):
        """Inject covisible bounding boxes into each view's data dict."""
        from ..geometry.depth import covisible_bbox

        pad = self.conf.extractor.get("cell_size", 16)
        for i in range(num_views):
            j = 1 - i
            vi, vj = data[f"view{i}"], data[f"view{j}"]
            if "depth" in vi and "camera" in vi and f"T_{i}to{j}" in data:
                vi["covisible_bbox"] = covisible_bbox(
                    vi["depth"], vi["camera"], vj["camera"],
                    data[f"T_{i}to{j}"], pad=pad,
                )

    def extract_view(self, data_i):
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        if self.conf.extractor.name and not skip_extract:
            pred_i = {**pred_i, **self.extractor(data_i)}
        elif self.conf.extractor.name and not self.conf.allow_no_extract:
            pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}
        return pred_i

    def triplet_to_pairs(self, data):
        # Convert triplet to three pairs (inplace because easier)
        tv_datas = [misc.get_twoview(data, idx) for idx in ["0to1", "1to2", "0to2"]]
        data.clear()
        data.update(misc.concat_tree(tv_datas))
        return data

    def _forward(self, data):
        num_views = len([k for k in data.keys() if k.startswith("view")])
        if self.conf.extractor.get("bias_to", None) == "covisible" and num_views == 2:
            self._precompute_covisible_bboxes(data, num_views)
        if self.conf.get("extract_parallel", False) and self.training:
            bs = data["view0"]["image"].shape[0]
            vdata = misc.concat_tree(misc.iterelements(data, pattern="view{i}"))
            vpred = self.extract_view(vdata)
            preds = misc.split_tree(vpred, bs, num_views)
        else:
            preds = [self.extract_view(data[f"view{i}"]) for i in range(num_views)]

        pred = {**{f"{k}{i}": v for i, p in enumerate(preds) for k, v in p.items()}}

        if num_views > 2:
            assert num_views == 3, "Only support triplets for now"
            # Convert triplet to three pairs (inplace because easier)
            self.triplet_to_pairs(data)
            self.triplet_to_pairs(pred)

        if self.conf.ground_truth.name and self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
        if self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred})}
        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}
        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}
        return pred

    def loss(self, pred, data):
        if "view2" in data:
            self.triplet_to_pairs(data)
        losses = {}
        metrics = {}
        total = 0

        # get labels
        if self.conf.ground_truth.name and not self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        for k in self.components:
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
                if "total" in losses_:
                    total = losses_["total"] + total
        if isinstance(total, torch.Tensor):
            losses["total"] = total
        return losses, metrics

    def visualize(self, pred, data, **kwargs):
        """Visualize the matches."""
        if "view2" in data:
            self.triplet_to_pairs(data)
        figures = {}
        for k in self.components:
            if self.conf[k].name and self.conf[k].get("visualize", True):
                figures.update(getattr(self, k).visualize(pred, data, **kwargs))
        return figures

    def pr_metrics(self, pred, data):
        """Compute precision-recall metrics."""
        pr_metrics = {}
        if "view2" in data:
            self.triplet_to_pairs(data)
        for k in self.components:
            if self.conf[k].name and hasattr(getattr(self, k), "pr_metrics"):
                pr_metrics.update(getattr(self, k).pr_metrics(pred, data))
        return pr_metrics

    def compile(self, *args, **kwargs) -> BaseModel:
        if self.conf.compile:
            return super().compile(*args, **kwargs)
        for k in self.components:
            if self.conf[k].name:
                setattr(self, k, getattr(self, k).compile(*args, **kwargs))
        return self
