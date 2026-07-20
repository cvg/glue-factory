"""Wrap two models, running one during training and the other during eval.

Generic: works for any pair of BaseModel components (extractors, matchers,
solvers, ...) — e.g. training with a cheap/randomized extractor while
evaluating with the "real" one. Dispatches purely on self.training, the
standard nn.Module flag already toggled by the trainer, so no changes are
needed anywhere else in the pipeline.
"""

import logging

from omegaconf import OmegaConf

from . import get_model
from .base_model import BaseModel

to_ctr = OmegaConf.to_container
logger = logging.getLogger(__name__)


class TrainEvalSwitch(BaseModel):
    default_conf = {
        "train": {"name": None},
        "eval": {"name": None},
    }

    def _init(self, conf):
        assert conf.train.name, "TrainEvalSwitch requires conf.train.name to be set."
        assert conf.eval.name, "TrainEvalSwitch requires conf.eval.name to be set."
        self.train_model = get_model(conf.train.name)(to_ctr(conf.train))
        self.eval_model = get_model(conf.eval.name)(to_ctr(conf.eval))
        # eval_model never runs during training (see _forward), so it should
        # never receive gradients either — freeze it regardless of its own
        # config's `trainable` setting.
        for p in self.eval_model.parameters():
            p.requires_grad = False

    def _forward(self, data):
        model = self.train_model if self.training else self.eval_model
        return model(data)

    def loss(self, pred, data):
        model = self.train_model if self.training else self.eval_model
        return model.loss(pred, data)


def build_with_train_alternative(conf):
    """Build a component from `conf`, or wrap it in a TrainEvalSwitch if
    `conf.train_alternative.name` is set.

    `conf` itself (minus `train_alternative`) becomes the eval branch. The
    train branch is `conf` merged with `train_alternative` on top (via
    OmegaConf.merge), so only fields that actually differ (e.g. `name`) need
    to be specified in `train_alternative` — everything else (e.g.
    `max_num_keypoints`) carries over from the base config automatically.
    This lets any pipeline component opt into a train-only alternative with
    a single override — e.g. `model.extractor.train_alternative.name=random`
    — instead of restructuring its whole config into explicit train/eval
    blocks.
    """
    train_alt = conf.get("train_alternative", None)
    if not train_alt or not train_alt.get("name", None):
        return get_model(conf.name)(to_ctr(conf))
    eval_conf = to_ctr(conf)
    eval_conf.pop("train_alternative", None)
    train_conf = to_ctr(OmegaConf.merge(eval_conf, to_ctr(train_alt)))
    switch = TrainEvalSwitch({"train": train_conf, "eval": eval_conf})
    logger.info(
        "Wrapping %r in a TrainEvalSwitch:\n%s", conf.name, OmegaConf.to_yaml(switch.conf)
    )
    return switch
