import pyelsed
import torch


from ..base_model import BaseModel


class ELSED(BaseModel):
    """
    ELSED wrapper to be able to run inference with ELSED (https://github.com/iago-suarez/ELSED)
    Use requires make-install of ELSED as described in the github repo's README
    
    Input: image of shape B x (1|3) x h x w ?  -> assert batched input of batch size 1!
    
    Output: lines of shape N x 2 x 2 
    """
    required_data_keys = ["image"]

    def _init(self, conf):
        self.set_initialized()

    def _forward(self, data):
        image = data["image"]
        assert image.ndim == 4  # assert batched input
        assert image.shape[0] == 1  # assert batches of 1

        if image.shape[1] == 3:
            # Convert to grayscale
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        image = image.squeeze(1).squeeze(0)
        
        # Forward pass
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        segs, scores = pyelsed.detect(image)

        lines = segs.reshape(-1,2,2)  # N x 4 -> N x 2 x 2

        return {"lines": lines, "lines_scores": scores}

    def loss(self, pred, data):
        raise NotImplementedError