from copy import copy

from omegaconf import OmegaConf


class BaseEstimator:
    base_default_conf = {
        "name": "???",
        "ransac_th": "???",
    }
    test_thresholds = [1.0]
    required_data_keys = []

    strict_conf = False

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        default_conf = OmegaConf.merge(
            self.base_default_conf, OmegaConf.create(self.default_conf)
        )
        if self.strict_conf:
            OmegaConf.set_struct(default_conf, True)

        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)
        self.required_data_keys = copy(self.required_data_keys)
        self._init(conf)

    def __call__(self, data):
        return self._forward(data)
