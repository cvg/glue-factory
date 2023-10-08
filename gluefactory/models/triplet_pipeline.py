"""
A two-view sparse feature matching pipeline on triplets.

If a triplet is found, runs the extractor on three images and
then runs matcher/filter/solver for all three pairs.

Losses and metrics get accumulated accordingly.

If no triplet is found, this falls back to two_view_pipeline.py
"""

import torch

from ..utils.misc import get_twoview, stack_twoviews, unstack_twoviews
from .two_view_pipeline import TwoViewPipeline


def has_triplet(data):
    # we already check for image0 and image1 in required_keys
    return "view2" in data.keys()


class TripletPipeline(TwoViewPipeline):
    default_conf = {"batch_triplets": True, **TwoViewPipeline.default_conf}

    def _forward(self, data):
        if not has_triplet(data):
            return super()._forward(data)
        # the two-view outputs are stored in
        # pred['0to1'],pred['0to2'], pred['1to2']

        assert not self.conf.run_gt_in_forward
        pred0 = self.extract_view(data, "0")
        pred1 = self.extract_view(data, "1")
        pred2 = self.extract_view(data, "2")

        pred = {}
        pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
            **{k + "2": v for k, v in pred2.items()},
        }

        def predict_twoview(pred, data):
            # forward pass
            if self.conf.matcher.name:
                pred = {**pred, **self.matcher({**data, **pred})}

            if self.conf.filter.name:
                pred = {**pred, **self.filter({**m_data, **pred})}

            if self.conf.solver.name:
                pred = {**pred, **self.solver({**m_data, **pred})}
            return pred

        if self.conf.batch_triplets:
            B = data["image1"].shape[0]
            # stack on batch dimension
            m_data = stack_twoviews(data)
            m_pred = stack_twoviews(pred)

            # forward pass
            m_pred = predict_twoview(m_pred, m_data)

            # unstack
            pred = {**pred, **unstack_twoviews(m_pred, B)}
        else:
            for idx in ["0to1", "0to2", "1to2"]:
                m_data = get_twoview(data, idx)
                m_pred = get_twoview(pred, idx)
                pred[idx] = predict_twoview(m_pred, m_data)
        return pred

    def loss(self, pred, data):
        if not has_triplet(data):
            return super().loss(pred, data)
        if self.conf.batch_triplets:
            m_data = stack_twoviews(data)
            m_pred = stack_twoviews(pred)
            losses, metrics = super().loss(m_pred, m_data)
        else:
            losses = {}
            metrics = {}
            for idx in ["0to1", "0to2", "1to2"]:
                data_i = get_twoview(data, idx)
                pred_i = pred[idx]
                losses_i, metrics_i = super().loss(pred_i, data_i)
                for k, v in losses_i.items():
                    if k in losses.keys():
                        losses[k] = losses[k] + v
                    else:
                        losses[k] = v
                for k, v in metrics_i.items():
                    if k in metrics.keys():
                        metrics[k] = torch.cat([metrics[k], v], 0)
                    else:
                        metrics[k] = v

        return losses, metrics
