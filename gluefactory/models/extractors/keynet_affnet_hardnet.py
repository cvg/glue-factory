import kornia
import torch

from ..base_model import BaseModel
from ..utils.misc import pad_to_length


class KeyNetAffNetHardNet(BaseModel):
    default_conf = {
        "max_num_keypoints": None,
        "desc_dim": 128,
        "upright": False,
        "scale_laf": 1.0,
        "chunk": 4,  # for reduced VRAM in training
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        self.model = kornia.feature.KeyNetHardNet(
            num_features=conf.max_num_keypoints,
            upright=conf.upright,
            scale_laf=conf.scale_laf,
        )
        self.set_initialized()

    def _forward(self, data):
        image = data["image"]
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        lafs, scores, descs = [], [], []
        im_size = data.get("image_size")
        for i in range(image.shape[0]):
            img_i = image[i : i + 1, :1]
            if im_size is not None:
                img_i = img_i[:, :, : im_size[i, 1], : im_size[i, 0]]
            laf, score, desc = self.model(img_i)
            xn = pad_to_length(
                kornia.feature.get_laf_center(laf),
                self.conf.max_num_keypoints,
                pad_dim=-2,
                mode="random_c",
                bounds=(0, min(img_i.shape[-2:])),
            )
            laf = torch.cat(
                [
                    laf,
                    kornia.feature.laf_from_center_scale_ori(xn[:, score.shape[-1] :]),
                ],
                -3,
            )
            lafs.append(laf)
            scores.append(pad_to_length(score, self.conf.max_num_keypoints, -1))
            descs.append(pad_to_length(desc, self.conf.max_num_keypoints, -2))

        lafs = torch.cat(lafs, 0)
        scores = torch.cat(scores, 0)
        descs = torch.cat(descs, 0)
        keypoints = kornia.feature.get_laf_center(lafs)
        scales = kornia.feature.get_laf_scale(lafs)[..., 0]
        oris = kornia.feature.get_laf_orientation(lafs)
        pred = {
            "keypoints": keypoints,
            "scales": scales.squeeze(-1),
            "oris": oris.squeeze(-1),
            "lafs": lafs,
            "keypoint_scores": scores,
            "descriptors": descs,
        }

        return pred

    def loss(self, pred, data):
        raise NotImplementedError
