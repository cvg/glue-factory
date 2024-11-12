import numpy as np
import torch
import pyelsed

from gluefactory.models.utils.metrics_lines import match_segments_1_to_1, match_segments_lbd
from ..base_model import BaseModel


class LineMatcher_LBD(BaseModel):
    default_conf = {
        "line_dist": "orth",
        "ELSED": False,
        "angular_th": (30 * np.pi / 180),
        "overlap_th": 0.5,
        "dist_thresh": 5,
        "min_length": 20,
    }

    def _init(self, conf):
        # TODO (iago): Is this just boilerplate code?
        pass

    required_data_keys = ["H_0to1", "lines0", "lines1", "view0"]

    def _forward(self, data):
        device = data["lines0"][0].device
        img_size = data["view0"]["image"].shape[2], data["view0"]["image"].shape[3]
        result = {}
        if self.conf.ELSED:
            rgb = (data["view0"]["image"][0]*255).to(torch.int32)
            r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]

            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

            segments0, scores = pyelsed.detect(img = gray.cpu().numpy().astype(np.uint8))

            rgb = (data["view1"]["image"][0]*255).to(torch.int32)
            r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]

            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

            segments1, scores = pyelsed.detect(img = gray.cpu().numpy().astype(np.uint8))

            lines0 = torch.from_numpy(segments0.reshape(-1, 2, 2))[:, :, [1, 0]].cpu()
            lines1 = torch.from_numpy(segments1.reshape(-1, 2, 2))[:, :, [1, 0]].cpu()
        else:
            lines0 = data["lines0"][0][:, :, [1, 0]].cpu()
            lines1 = data["lines1"][0][:, :, [1, 0]].cpu()

        rgb = (data["view0"]["image"][0]*255).to(torch.int32)
        r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]

        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        segs1, segs2, matched_idx1, matched_idx2 = match_segments_lbd(
        gray.cpu().numpy().astype(np.uint8), lines0.numpy(), lines1.numpy(), data["H_0to1"][0].cpu().numpy(), 
        img_size)
        
        result["orig_lines0"] = lines0[:, :, [1, 0]].unsqueeze(0).to(device)
        result["orig_lines1"] = lines1[:, :, [1, 0]].unsqueeze(0).to(device)

        result["lines0"] = torch.Tensor(segs1).unsqueeze(0).to(device)
        result["lines1"] = torch.Tensor(segs2).unsqueeze(0).to(device)
        result["line_matches0"] = torch.Tensor(matched_idx1).unsqueeze(0).to(device)
        result["line_matches1"] = torch.Tensor(matched_idx2).unsqueeze(0).to(device)

        return result

    def loss(self, pred, data):
        raise NotImplementedError