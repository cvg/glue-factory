"""
Nearest neighbor matcher for normalized descriptors.
Optionally apply the mutual check and threshold the distance or ratio.
"""

import logging

import torch
import torch.nn.functional as F

from ..base_model import BaseModel
from ..utils.metrics import matcher_metrics


@torch.no_grad()
def find_nn(sim, ratio_thresh, distance_thresh):
    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2) * dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    return matches


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    inds1 = torch.arange(m1.shape[-1], device=m1.device)
    loop0 = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    loop1 = torch.gather(m0, -1, torch.where(m1 > -1, m1, m1.new_tensor(0)))
    m0_new = torch.where((m0 > -1) & (inds0 == loop0), m0, m0.new_tensor(-1))
    m1_new = torch.where((m1 > -1) & (inds1 == loop1), m1, m1.new_tensor(-1))
    return m0_new, m1_new


class NearestNeighborMatcher(BaseModel):
    default_conf = {
        "ratio_thresh": None,
        "distance_thresh": None,
        "mutual_check": True,
        "loss": None,
    }
    required_data_keys = ["descriptors0", "descriptors1"]

    def _init(self, conf):
        if conf.loss == "N_pair":
            temperature = torch.nn.Parameter(torch.tensor(1.0))
            self.register_parameter("temperature", temperature)

    def _forward(self, data):
        sim = torch.einsum("bnd,bmd->bnm", data["descriptors0"], data["descriptors1"])
        matches0 = find_nn(sim, self.conf.ratio_thresh, self.conf.distance_thresh)
        matches1 = find_nn(
            sim.transpose(1, 2), self.conf.ratio_thresh, self.conf.distance_thresh
        )
        if self.conf.mutual_check:
            matches0, matches1 = mutual_check(matches0, matches1)
        b, m, n = sim.shape
        la = sim.new_zeros(b, m + 1, n + 1)
        la[:, :-1, :-1] = F.log_softmax(sim, -1) + F.log_softmax(sim, -2)
        mscores0 = (matches0 > -1).float()
        mscores1 = (matches1 > -1).float()
        return {
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "similarity": sim,
            "log_assignment": la,
        }

    def loss(self, pred, data):
        losses = {}
        if self.conf.loss == "N_pair":
            sim = pred["similarity"]
            if torch.any(sim > (1.0 + 1e-6)):
                logging.warning(f"Similarity larger than 1, max={sim.max()}")
            scores = torch.sqrt(torch.clamp(2 * (1 - sim), min=1e-6))
            scores = self.temperature * (2 - scores)
            assert not torch.any(torch.isnan(scores)), torch.any(torch.isnan(sim))
            prob0 = torch.nn.functional.log_softmax(scores, 2)
            prob1 = torch.nn.functional.log_softmax(scores, 1)

            assignment = data["gt_assignment"].float()
            num = torch.max(assignment.sum((1, 2)), assignment.new_tensor(1))
            nll0 = (prob0 * assignment).sum((1, 2)) / num
            nll1 = (prob1 * assignment).sum((1, 2)) / num
            nll = -(nll0 + nll1) / 2
            losses["n_pair_nll"] = losses["total"] = nll
            losses["num_matchable"] = num
            losses["n_pair_temperature"] = self.temperature[None]
        else:
            raise NotImplementedError
        metrics = {} if self.training else matcher_metrics(pred, data)
        return losses, metrics
