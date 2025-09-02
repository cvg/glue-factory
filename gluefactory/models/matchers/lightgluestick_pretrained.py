from lightgluestick import LightGlueStick as LightGlueStick_
from omegaconf import OmegaConf

from ..base_model import BaseModel


class LightGlueStick(BaseModel):
    default_conf = {"features": "superpoint", **LightGlueStick_.default_conf}

    def _init(self, conf):
        dconf = OmegaConf.to_container(conf)
        self.net = LightGlueStick_(dconf)
        self.set_initialized()

    def _forward(self, data):
        return self.net(data)

    def loss(pred, data):
        raise NotImplementedError
