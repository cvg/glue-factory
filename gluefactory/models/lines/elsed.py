import pyelsed
import torch


from ..base_model import BaseModel


class ELSED(BaseModel):
    """
    ELSED wrapper to be able to run inference with ELSED (https://github.com/iago-suarez/ELSED)
    Use requires make-install of ELSED as described in the github repo's README
    """
    required_data_keys = ["image"]

    def _init(self, conf):
        self.set_initialized()

    def _forward(self, data):
        # TODO: how to deal with batches -> implement
        image = data["image"]
        if image.shape[1] == 3:
            # Convert to grayscale
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        # Forward pass
        with torch.no_grad():
            segs, scores = pyelsed.detect(image)

        lines = []
        for i in range(len(segs)):
            cur = segs[i]
            lines.append(cur[:, :, [1, 0]])

        return {"lines": lines, "lines_scores": scores}

    def loss(self, pred, data):
        raise NotImplementedError