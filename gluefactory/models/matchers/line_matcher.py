import numpy as np
import torch

from gluefactory.models.utils.metrics_lines import match_segments_1_to_1
from ..base_model import BaseModel


class LineMatcher(BaseModel):
    default_conf = {
        "line_dist": "orth",
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
        lines0 = data["lines0"][0][:, :, [1, 0]].cpu()
        lines1 = data["lines1"][0][:, :, [1, 0]].cpu()
        lines0 = lines0[
            torch.linalg.norm(lines0[:, 1] - lines0[:, 0], axis=1) > self.conf.min_length
        ]
        lines1 = lines1[
            torch.linalg.norm(lines1[:, 1] - lines1[:, 0], axis=1) > self.conf.min_length
        ]
        # The data elements come in lists and therefore they are unpacked
        segs1, segs2, matched_idx1, matched_idx2, distances = match_segments_1_to_1(
            lines0,
            lines1,
            data["H_0to1"][0].cpu(),
            img_size,
            self.conf.line_dist,
            self.conf.angular_th,
            self.conf.overlap_th,
            self.conf.dist_thresh,
        )
        # print(f'{len(data["lines0"][0]): {len(matched_idx1)}}')
        result["orig_lines0"] = torch.Tensor(lines0).unsqueeze(0).to(device)
        result["orig_lines1"] = torch.Tensor(lines1).unsqueeze(0).to(device)
        result["lines0"] = torch.Tensor(segs1).unsqueeze(0).to(device)
        result["lines1"] = torch.Tensor(segs2).unsqueeze(0).to(device)
        result["line_matches0"] = torch.Tensor(matched_idx1).unsqueeze(0).to(device)
        result["line_matches1"] = torch.Tensor(matched_idx2).unsqueeze(0).to(device)
        result["line_matching_scores0"] = (
            torch.Tensor(distances).unsqueeze(0).to(device)
        )
        result["line_matching_scores1"] = (
            torch.Tensor(distances).unsqueeze(0).to(device)
        )
        return result

    def loss(self, pred, data):
        raise NotImplementedError
