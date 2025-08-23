import torch
import torch.nn as nn
from omegaconf import OmegaConf

def weight_loss(log_assignment, weights, gamma=0.0):
    b, m, n = log_assignment.shape
    m -= 1
    n -= 1

    loss_sc = log_assignment * weights

    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)

    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)

    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)

    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0

def weight_loss_one2many(log_assignment_pos, pos_certainties, log_assignment_neg, \
                         positive, negative, gamma=0.0, alpha=0.0):
    b, m, n = log_assignment_pos.shape

    loss_sc_pos = log_assignment_pos * positive
    pos_certainties = pos_certainties * positive

    loss_sc_neg = log_assignment_neg * negative

    num_neg0 = negative[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = negative[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = positive[:, :m, :n].sum((-1, -2)).clamp(min=1.0)
    num_neg = negative[:, :m, :n].sum((-1, -2)).clamp(min=1.0)

    nll_pos = -loss_sc_pos[:, :m, :n].sum((-1, -2)) - pos_certainties.sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)

    nll_neg = -loss_sc_neg[:, :m, :n].sum((-1, -2))
    nll_neg /= num_neg.clamp(min=1.0)

    nll_neg0 = -loss_sc_neg[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc_neg[:, -1, :n].sum(-1)

    nll_neg = nll_neg + (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

    return alpha * nll_pos + (1 - alpha) * nll_neg, nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0

def focal_loss_one2many(log_assignment_pos, pos_certainties, log_assignment_neg, \
                        positive, negative, gamma=0.0, alpha=0.0):
    b, m, n = log_assignment_pos.shape

    loss_sc_pos = log_assignment_pos * positive
    pos_certainties = pos_certainties * positive
    loss_sc_neg = log_assignment_neg * negative

    focal_weight_pos = alpha * (1 - log_assignment_pos.exp()) ** gamma
    focal_weight_neg = (1 - alpha) * (1 - log_assignment_neg[:, :m, :n].exp()) ** gamma

    num_neg0 = negative[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = negative[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = positive[:, :m, :n].sum((-1, -2)).clamp(min=1.0)
    num_neg = negative[:, :m, :n].sum((-1, -2)).clamp(min=1.0)

    nll_pos = - (focal_weight_pos * loss_sc_pos[:, :m, :n] + pos_certainties).sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)

    nll_neg = - (focal_weight_neg * loss_sc_neg[:, :m, :n]).sum((-1, -2))
    nll_neg /= num_neg.clamp(min=1.0)

    nll_neg0 = -loss_sc_neg[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc_neg[:, -1, :n].sum(-1)

    nll_neg = nll_neg + (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

    return nll_pos + nll_neg, nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0

class NLLLoss(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.loss_fn = self.nll_loss

    def forward(self, pred, data, weights=None, prefix=""):
        log_assignment = pred[prefix + "log_assignment"]
        if weights is None:
            weights = self.loss_fn(log_assignment, data, prefix)
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(
            log_assignment, weights, gamma=self.conf.gamma_f
        )
        nll = (
            self.conf.nll_balancing * nll_pos + (1 - self.conf.nll_balancing) * nll_neg
        )

        return (
            nll,
            weights,
            {
                prefix + "assignment_nll": nll,
                prefix + "nll_pos": nll_pos,
                prefix + "nll_neg": nll_neg,
                prefix + "num_matchable": num_pos,
                prefix + "num_unmatchable": num_neg,
            },
        )

    def nll_loss(self, log_assignment, data, prefix=""):
        m, n = data["gt_" + prefix + "matches0"].size(-1), data["gt_" + prefix + "matches1"].size(-1)
        positive = data["gt_" + prefix + "assignment"].float()
        neg0 = (data["gt_" + prefix + "matches0"] == -1).float()
        neg1 = (data["gt_" + prefix + "matches1"] == -1).float()

        weights = torch.zeros_like(log_assignment)
        weights[:, :m, :n] = positive

        weights[:, :m, -1] = neg0
        weights[:, -1, :n] = neg1
        return weights

class NLLLoss_one2many(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.loss_fn = self.nll_loss

    def forward(self, pred, data, positive=None, negative=None, prefix=""):
        log_assignment_pos = pred[prefix + "log_assignment"]
        log_assignment_neg = pred[prefix + "neg_log_assignment"]
        pos_certainties = pred[prefix + "certainties"]

        if positive is None or negative is None:
            positive, negative = self.loss_fn(log_assignment_neg, data, prefix)
        nll, nll_pos, nll_neg, num_pos, num_neg = focal_loss_one2many(
            log_assignment_pos, pos_certainties, log_assignment_neg, positive, negative, \
            gamma=self.conf.gamma_f, alpha=self.conf.nll_balancing
        )
        return (
            nll,
            positive,
            negative,
            {
                prefix + "assignment_nll": nll,
                prefix + "nll_pos": nll_pos,
                prefix + "nll_neg": nll_neg,
                prefix + "num_matchable": num_pos,
                prefix + "num_unmatchable": num_neg,
            },
        )

    def nll_loss(self, log_assignment_neg, data, prefix=""):
        m, n = data["gt_" + prefix + "matches0"].size(-1), data["gt_" + prefix + "matches1"].size(-1)
        positive = data["gt_" + prefix + "assignment"].float()

        neg0 = (data["gt_" + prefix + "matches0"] == -1).float()
        neg1 = (data["gt_" + prefix + "matches1"] == -1).float()

        ignore0 = (data["gt_" + prefix + "matches0"] == -2)
        ignore1 = (data["gt_" + prefix + "matches1"] == -2)

        negative = torch.zeros_like(log_assignment_neg)

        # Handle unmatchable cases
        negative[:, :m, -1] = neg0
        negative[:, -1, :n] = neg1

        # Set ignore labels to 0 in the negative mask
        negative[:, :m, :n] = 1 - positive

        batch_indices, row_indices = torch.where(ignore0)
        negative[batch_indices, row_indices, :] = 0

        batch_indices, col_indices = torch.where(ignore1)
        negative[batch_indices, :, col_indices] = 0

        return positive, negative