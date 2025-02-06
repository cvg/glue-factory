import kornia.feature as KF
import torch

from ..base_model import BaseModel


class SOLD2(BaseModel):
    required_data_keys = ["image"]

    def _init(self, conf):
        self.net = KF.SOLD2(pretrained=True, config=None)
        self.set_initialized()

    def _forward(self, data):
        image = data["image"]
        if image.shape[1] == 3:
            # Convert to grayscale
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        # Forward pass
        with torch.no_grad():
            segs = self.net(image)["line_segments"]

        lines = []
        for i in range(len(segs)):
            cur = segs[i]
            lines.append(cur[:, :, [1, 0]])

        return {"lines": torch.stack(lines, dim=0)}

    def loss(self, pred, data):
        raise NotImplementedError
