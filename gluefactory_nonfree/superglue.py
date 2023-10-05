"""
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

Described in:
    SuperGlue: Learning Feature Matching with Graph Neural Networks,
    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz,
    Andrew Rabinovich, CVPR 2020.

Original code: github.com/MagicLeapResearch/SuperPointPretrainedNetwork

Adapted by Philipp Lindenberger (Phil26AT)
"""

import torch
from torch import nn
from copy import deepcopy
import logging
from torch.utils.checkpoint import checkpoint

from gluefactory.models.base_model import BaseModel


def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(kpts, size=None, shape=None):
    if size is None:
        assert shape is not None
        _, _, h, w = shape
        one = kpts.new_tensor(1)
        size = torch.stack([one * w, one * h])[None]

    shift = size.float().to(kpts) / 2
    scale = size.max(1).values.float().to(kpts) * 0.7
    kpts = (kpts - shift[:, None]) / scale[:, None, None]
    return kpts


class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers, use_scores=True):
        super().__init__()
        self.use_scores = use_scores
        c = 3 if use_scores else 2
        self.encoder = MLP([c] + list(layers) + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        if self.use_scores:
            inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        else:
            inputs = [kpts.transpose(1, 2)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dim = d_model // h
        self.h = h
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [
            layer(x).view(b, self.dim, self.h, -1)
            for layer, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(b, self.dim * self.h, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, num_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim * 2, num_dim * 2, num_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_names):
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))]
        )
        self.names = layer_names

    def forward(self, desc0, desc1):
        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            layer.attn.prob = []
            if self.training:
                delta0, delta1 = checkpoint(
                    self._forward, layer, desc0, desc1, name, preserve_rng_state=False
                )
            else:
                delta0, delta1 = self._forward(layer, desc0, desc1, name)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
            del delta0, delta1
        return desc0, desc1

    def _forward(self, layer, desc0, desc1, name):
        if name == "self":
            return layer(desc0, desc0), layer(desc1, desc1)
        elif name == "cross":
            return layer(desc0, desc1), layer(desc1, desc0)
        else:
            raise ValueError(name)


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters):
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat(
        [torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1
    )

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(BaseModel):
    default_conf = {
        "descriptor_dim": 256,
        "weights": "outdoor",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "num_sinkhorn_iterations": 50,
        "filter_threshold": 0.2,
        "use_scores": True,
        "loss": {
            "nll_balancing": 0.5,
        },
    }
    required_data_keys = [
        "view0",
        "view1",
        "keypoints0",
        "keypoints1",
        "descriptors0",
        "descriptors1",
        "keypoint_scores0",
        "keypoint_scores1",
    ]

    checkpoint_url = "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_{}.pth"  # noqa: E501

    def _init(self, conf):
        self.kenc = KeypointEncoder(
            conf.descriptor_dim, conf.keypoint_encoder, conf.use_scores
        )

        self.gnn = AttentionalGNN(conf.descriptor_dim, conf.GNN_layers)

        self.final_proj = nn.Conv1d(
            conf.descriptor_dim, conf.descriptor_dim, kernel_size=1, bias=True
        )
        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)

        if conf.weights:
            assert conf.weights in ["indoor", "outdoor"]
            url = self.checkpoint_url.format(conf.weights)
            self.load_state_dict(torch.hub.load_state_dict_from_url(url))
            logging.info(f"Loading SuperGlue trained for {conf.weights}.")

    def _forward(self, data):
        desc0 = data["descriptors0"].transpose(-1, -2)
        desc1 = data["descriptors1"].transpose(-1, -2)
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                "matches0": kpts0.new_full(shape0, -1, dtype=torch.int),
                "matches1": kpts1.new_full(shape1, -1, dtype=torch.int),
                "matching_scores0": kpts0.new_zeros(shape0),
                "matching_scores1": kpts1.new_zeros(shape1),
            }
        view0, view1 = data["view0"], data["view1"]
        kpts0 = normalize_keypoints(
            kpts0, size=view0.get("image_size"), shape=view0["image"].shape
        )
        kpts1 = normalize_keypoints(
            kpts1, size=view1.get("image_size"), shape=view1["image"].shape
        )
        assert torch.all(kpts0 >= -1) and torch.all(kpts0 <= 1)
        assert torch.all(kpts1 >= -1) and torch.all(kpts1 <= 1)
        desc0 = desc0 + self.kenc(kpts0, data["keypoint_scores0"])
        desc1 = desc1 + self.kenc(kpts1, data["keypoint_scores1"])

        desc0, desc1 = self.gnn(desc0, desc1)

        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        cost = scores / self.conf.descriptor_dim**0.5

        scores = log_optimal_transport(
            cost, self.bin_score, iters=self.conf.num_sinkhorn_iterations
        )

        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        m0, m1 = max0.indices, max1.indices
        mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
        valid0 = mutual0 & (mscores0 > self.conf.filter_threshold)
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))
        m1 = torch.where(valid1, m1, m1.new_tensor(-1))

        return {
            "sinkhorn_cost": cost,
            "log_assignment": scores,
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
        }

    def loss(self, pred, data):
        losses = {"total": 0}

        positive = data["gt_assignment"].float()
        num_pos = torch.max(positive.sum((1, 2)), positive.new_tensor(1))
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()
        num_neg = torch.max(neg0.sum(1) + neg1.sum(1), neg0.new_tensor(1))

        log_assignment = pred["log_assignment"]
        nll_pos = -(log_assignment[:, :-1, :-1] * positive).sum((1, 2))
        nll_pos /= num_pos
        nll_neg0 = -(log_assignment[:, :-1, -1] * neg0).sum(1)
        nll_neg1 = -(log_assignment[:, -1, :-1] * neg1).sum(1)
        nll_neg = (nll_neg0 + nll_neg1) / num_neg
        nll = (
            self.conf.loss.nll_balancing * nll_pos
            + (1 - self.conf.loss.nll_balancing) * nll_neg
        )
        losses["assignment_nll"] = nll
        losses["total"] = nll

        losses["nll_pos"] = nll_pos
        losses["nll_neg"] = nll_neg

        # Some statistics
        losses["num_matchable"] = num_pos
        losses["num_unmatchable"] = num_neg
        losses["bin_score"] = self.bin_score[None]

        return losses

    def metrics(self, pred, data):
        raise NotImplementedError
