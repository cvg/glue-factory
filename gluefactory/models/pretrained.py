"""Pretrained model wrapper.

Loads a model from a checkpoint and extracts a submodel at a given path.
This allows integrating any pretrained component into a new pipeline.

Example config::

    model:
      solver:
        name: pretrained
        checkpoint: my_experiment
        path: solver
        n_iters: 8  # forwarded as override to solver.n_iters in the loaded model
"""

import logging

from omegaconf import OmegaConf

from gluefactory.models import BaseModel

logger = logging.getLogger(__name__)


class Pretrained(BaseModel):
    default_conf = {
        "checkpoint": "???",  # experiment name or path to .tar
        "path": "???",  # dot-separated path to submodel, e.g. "solver"
    }
    strict_conf = False
    required_data_keys = []

    def _init(self, conf):
        from gluefactory.utils.experiments import load_experiment

        # Separate own keys from config overrides
        own_keys = set(self.base_default_conf.keys()) | {"checkpoint", "path"}
        conf_dict = OmegaConf.to_container(conf, resolve=True)
        overrides = {k: v for k, v in conf_dict.items() if k not in own_keys}

        # Nest overrides under the path so they apply to the right submodel
        path_parts = conf.path.split(".")
        nested = overrides
        for part in reversed(path_parts):
            nested = {part: nested}

        logger.info(
            f"Loading pretrained '{conf.path}' from checkpoint '{conf.checkpoint}'"
        )
        full_model = load_experiment(conf.checkpoint, conf=nested)

        # Navigate to the submodel
        submodel = full_model
        for part in path_parts:
            submodel = getattr(submodel, part)

        self.submodel = submodel

    def _forward(self, data):
        return self.submodel(data)

    def loss(self, pred, data):
        return self.submodel.loss(pred, data)
