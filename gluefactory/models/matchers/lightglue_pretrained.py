from ..base_model import BaseModel
from lightglue import LightGlue as LightGlue_
from omegaconf import OmegaConf


class LightGlue(BaseModel):
    default_conf = {"features": "superpoint", **LightGlue_.default_conf}
    required_data_keys = [
        "view0",
        "keypoints0",
        "descriptors0",
        "view1",
        "keypoints1",
        "descriptors1",
    ]

    def _init(self, conf):
        dconf = OmegaConf.to_container(conf)
        self.net = LightGlue_(dconf.pop("features"), **dconf).cuda()
        # self.net.compile()

    def _forward(self, data):
        view0 = {
            **{k: data[k + "0"] for k in ["keypoints", "descriptors"]},
            **data["view0"],
        }
        view1 = {
            **{k: data[k + "1"] for k in ["keypoints", "descriptors"]},
            **data["view1"],
        }
        return self.net({"image0": view0, "image1": view1})

    def loss(pred, data):
        raise NotImplementedError
