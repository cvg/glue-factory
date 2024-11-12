import pyelsed
import torch
import numpy as np


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
        device = image.device
        assert image.ndim == 4  # assert batched input
        assert image.shape[0] == 1  # assert batches of 1

        if image.shape[1] == 3:
            # Convert to grayscale
            rgb = (image[0]*255).to(torch.int32)
            r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
            image = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        # Forward pass
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy().astype(np.uint8)
        segs, scores = pyelsed.detect(img = image)

        lines = torch.tensor(segs.reshape(-1,2,2)).to(device)  # N x 4 -> N x 2 x 2
        scores = torch.tensor(scores).to(device)
        
        # add artificial batch dimension to lines again
        lines = lines.unsqueeze(0)
        return {"lines": lines, "lines_scores": scores}

    def loss(self, pred, data):
        raise NotImplementedError