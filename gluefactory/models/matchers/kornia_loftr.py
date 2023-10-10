import kornia
import torch

from ...models import BaseModel


class LoFTRModule(BaseModel):
    default_conf = {
        "topk": None,
        "zero_pad": False,
    }
    required_data_keys = ["view0", "view1"]

    def _init(self, conf):
        self.net = kornia.feature.LoFTR(pretrained="outdoor")
        self.set_initialized()

    def _forward(self, data):
        image0 = data["view0"]["image"]
        image1 = data["view1"]["image"]
        if self.conf.zero_pad:
            image0, mask0 = self.zero_pad(image0)
            image1, mask1 = self.zero_pad(image1)
            res = self.net(
                {"image0": image0, "image1": image1, "mask0": mask0, "mask1": mask1}
            )
            res = self.net({"image0": image0, "image1": image1})
        else:
            res = self.net({"image0": image0, "image1": image1})
        topk = self.conf.topk
        if topk is not None and res["confidence"].shape[-1] > topk:
            _, top = torch.topk(res["confidence"], topk, -1)
            m_kpts0 = res["keypoints0"][None][:, top]
            m_kpts1 = res["keypoints1"][None][:, top]
            scores = res["confidence"][None][:, top]
        else:
            m_kpts0 = res["keypoints0"][None]
            m_kpts1 = res["keypoints1"][None]
            scores = res["confidence"][None]

        m0 = torch.arange(0, scores.shape[-1]).to(scores.device)[None]
        m1 = torch.arange(0, scores.shape[-1]).to(scores.device)[None]
        return {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": scores,
            "keypoints0": m_kpts0,
            "keypoints1": m_kpts1,
            "keypoint_scores0": scores,
            "keypoint_scores1": scores,
            "matching_scores1": scores,
        }

    def zero_pad(self, img):
        b, c, h, w = img.shape
        if h == w:
            return img
        s = max(h, w)
        image = torch.zeros((b, c, s, s)).to(img)
        image[:, :, :h, :w] = img
        mask = torch.zeros_like(image)
        mask[:, :, :h, :w] = 1.0
        return image, mask.squeeze(0).float()

    def loss(self, pred, data):
        return NotImplementedError
