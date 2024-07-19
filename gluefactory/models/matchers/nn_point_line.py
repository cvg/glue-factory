"""
Nearest neighbor matcher for normalized descriptors.
Optionally apply the mutual check and threshold the distance or ratio.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from ..base_model import BaseModel
from ..utils.metrics import matcher_metrics

MATCH_THRESHOLD = 0.01
SINKHORN_ITERATIONS = 30
DESC_SIZE = 128
DEBUG = False


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


def mutual_match(m0, m1, scores):

    matches = []
    m1_set = set(tuple(x) for x in m1[:, [1, 0]])
    for idx in range(len(m0)):
        matches.append(idx)

    matches = m0[np.array(matches)]
    scores = scores[matches[:, 0], matches[:, 1]]
    return matches, scores


## Ref : https://github.com/cvg/limap/blob/main/limap/point2d/superglue/superglue.py LINE 263


def solve_optimal_transport(scores):
    bin_score = torch.nn.Parameter(torch.tensor(1.0)).to(scores.device)
    return log_optimal_transport(scores, bin_score, iters=SINKHORN_ITERATIONS)


## Ref : https://github.com/cvg/limap/blob/main/limap/point2d/superglue/superglue.py LINE 267


def log_sinkhorn_iterations(
    Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int
) -> torch.Tensor:
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


## Ref : https://github.com/cvg/limap/blob/main/limap/point2d/superglue/superglue.py LINE 275


def log_optimal_transport(
    scores: torch.Tensor, alpha: torch.Tensor, iters: int
) -> torch.Tensor:
    """Perform Differentiable Optimal Transport in Log-space for stability"""
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


## Ref : https://github.com/cvg/limap/blob/main/limap/point2d/superglue/superglue.py LINE 144


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


## Ref : https://github.com/cvg/limap/blob/main/limap/point2d/superglue/superglue.py LINE 297


def get_matches(scores_mat):
    max0, max1 = scores_mat[:, :-1, :-1].max(2), scores_mat[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
    mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)
    zero = scores_mat.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > MATCH_THRESHOLD)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, m0.new_tensor(-1))
    m1 = torch.where(valid1, m1, m1.new_tensor(-1))
    return m0, m1, mscores0, mscores1


## REF https://github.com/cvg/limap/blob/main/limap/line2d/endpoints/matcher.py#L33-L53


def line_match(desc1, desc2):
    with torch.no_grad():
        # Shaping endpoints
        desc1 = desc1.reshape(-1, DESC_SIZE)
        desc2 = desc2.reshape(-1, DESC_SIZE)

        # Run the point matching
        scores = desc1 @ desc2.t()

        # Retrieve the best matching score of the line endpoints
        n_lines1 = scores.shape[0] // 2
        n_lines2 = scores.shape[1] // 2
        scores = scores.reshape(n_lines1, 2, n_lines2, 2)
        scores = 0.5 * torch.maximum(
            scores[:, 0, :, 0] + scores[:, 1, :, 1],
            scores[:, 0, :, 1] + scores[:, 1, :, 0],
        )

        # Run the Sinkhorn algorithm and get the line matches
        # TODO figure out why this causes issues 0 matches
        # scores = solve_optimal_transport(scores[None])
        matches = get_matches(scores[None])[0].cpu().numpy()[0]

    # Transform matches to [n_matches, 2]
    id_list_1 = np.arange(0, matches.shape[0])[matches != -1]
    id_list_2 = matches[matches != -1]
    matches_t = np.stack([id_list_1, id_list_2], 1)
    return matches_t, scores.cpu().numpy()
    # return matches_t, scores.cpu().numpy()[0]


def match_segs_with_descinfo_topk(desc1, desc2, topk=1):
    with torch.no_grad():
        # Run the point matching
        desc1 = desc1.reshape(-1, DESC_SIZE)
        desc2 = desc2.reshape(-1, DESC_SIZE)
        scores = desc1 @ desc2.t()

        # Retrieve the best matching score of the line endpoints
        n_lines1 = scores.shape[0] // 2
        n_lines2 = scores.shape[1] // 2
        scores = scores.reshape(n_lines1, 2, n_lines2, 2)
        scores = 0.5 * torch.maximum(
            scores[:, 0, :, 0] + scores[:, 1, :, 1],
            scores[:, 0, :, 1] + scores[:, 1, :, 0],
        )

        # For each line in img1, retrieve the topk matches in img2
        matches = torch.argsort(scores, dim=1)[:, -topk:]
        matches = torch.flip(matches, dims=(1,))
        matches = matches.cpu().numpy()

    # Transform matches to [n_matches, 2]
    n_lines = matches.shape[0]
    topk = matches.shape[1]
    matches_t = np.stack([np.arange(n_lines).repeat(topk), matches.flatten()], axis=1)
    return matches_t, scores.cpu().numpy()


class NearestNeighborPointLineMatcher(BaseModel):
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

        # Lines
        line_matches0 = []
        line_matches1 = []
        line_matching_scores0 = []
        line_matching_scores1 = []

        # TODO Batching makes no sense, always 1
        for batch_idx in range(len(data["lines0"])):

            lm0 = np.array([])
            lm1 = np.array([])
            lms0 = np.array([])
            lms1 = np.array([])
            desc1 = torch.stack(
                [
                    data["descriptors0"][batch_idx][data["lines0"][batch_idx][:, 0]],
                    data["descriptors0"][batch_idx][data["lines0"][batch_idx][:, 1]],
                ],
                1,
            )

            desc2 = torch.stack(
                [
                    data["descriptors1"][batch_idx][data["lines1"][batch_idx][:, 0]],
                    data["descriptors1"][batch_idx][data["lines1"][batch_idx][:, 1]],
                ],
                1,
            )

            if DEBUG:
                print(f"lines0 : {data['lines0'][batch_idx].shape}")
                print(f"lines1 : {data['lines1'][batch_idx].shape}")
                print(f"desc1 : {desc1.shape}")
                print(f"desc2 : {desc2.shape}")

            if desc1.shape[0] * desc2.shape[0] > 0:
                # matches, scores = line_match(desc1, desc2)

                matches_1, scores1 = match_segs_with_descinfo_topk(desc1, desc2)
                matches_2, scores2 = match_segs_with_descinfo_topk(desc2, desc1)

                if DEBUG:
                    print(f"Matches 1 : {matches_1.shape}")
                    print(f"Matches 2 : {matches_2.shape}")

                matches = np.array(
                    [
                        x
                        for x in set(tuple(x) for x in matches_1)
                        & set(tuple(x) for x in matches_2[:, [1, 0]])
                    ]
                )
                if DEBUG:
                    print(f"Matches : {matches.shape}")
                # match_idx = (matches_1[:, None] == matches_2[:, [1,0]]).all(-1).argmax(0)
                # matches, scores = mutual_match(matches_1, matches_2, line_matching_scores0)

                lm0 = np.ones(data["lines0"][batch_idx].shape[0], dtype=np.int32) * (-1)
                lm0[matches[:, 0]] = matches[:, 1]
                lm1 = np.ones(data["lines1"][batch_idx].shape[0], dtype=np.int32) * (-1)
                lm1[matches[:, 1]] = matches[:, 0]
                # lm0 = matches[:, 0]
                # lm1 = matches[:, 1]
                lms0 = np.ones(data["lines0"][batch_idx].shape[0]) * (-1)
                lms1 = np.ones(data["lines1"][batch_idx].shape[0]) * (-1)
                lms0[matches[:, 0]] = lms1[matches[:, 1]] = scores1[
                    matches[:, 0], matches[:, 1]
                ]
            else:
                print(f'Lines0 Detected Issue:{data["lines0"][batch_idx].shape}')
                print(f'Lines1 Detected Issue:{data["lines1"][batch_idx].shape}')

            # TODO Batching makes no sense, always 1
            # line_matches0.append(torch.from_numpy(lm0).to(data['descriptors0'][batch_idx].device))
            # line_matches1.append(torch.from_numpy(lm1).to(data['descriptors0'][batch_idx].device))
            # line_matching_scores0.append(torch.from_numpy(lms0).to(data['descriptors0'][batch_idx].device))
            # line_matching_scores1.append(torch.from_numpy(lms1).to(data['descriptors0'][batch_idx].device))

            line_matches0 = (
                torch.from_numpy(lm0)
                .to(data["descriptors0"][batch_idx].device)
                .unsqueeze(0)
            )
            line_matches1 = (
                torch.from_numpy(lm1)
                .to(data["descriptors0"][batch_idx].device)
                .unsqueeze(0)
            )
            line_matching_scores0 = (
                torch.from_numpy(lms0)
                .to(data["descriptors0"][batch_idx].device)
                .unsqueeze(0)
            )
            line_matching_scores1 = (
                torch.from_numpy(lms1)
                .to(data["descriptors0"][batch_idx].device)
                .unsqueeze(0)
            )

        return {
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "similarity": sim,
            "log_assignment": la,
            "line_matches0": line_matches0,
            "line_matches1": line_matches1,
            "line_matching_scores0": line_matching_scores0,
            "line_matching_scores1": line_matching_scores1,
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
