import math

import torch

from ..base_model import BaseModel


def to_sequence(map):
    return map.flatten(-2).transpose(-1, -2)


def to_map(sequence):
    n = sequence.shape[-2]
    e = math.isqrt(n)
    assert e * e == n
    assert e * e == n
    sequence.transpose(-1, -2).unflatten(-1, [e, e])


class GridExtractor(BaseModel):
    default_conf = {"cell_size": 14}
    required_data_keys = ["image"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        b, c, h, w = data["image"].shape

        cgrid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(
                        h // self.conf.cell_size,
                        dtype=torch.float32,
                        device=data["image"].device,
                    ),
                    torch.arange(
                        w // self.conf.cell_size,
                        dtype=torch.float32,
                        device=data["image"].device,
                    ),
                    indexing="ij",
                )[::-1],
                dim=0,
            )
            .unsqueeze(0)
            .repeat([b, 1, 1, 1])
            * self.conf.cell_size
            + self.conf.cell_size / 2
        )
        pred = {
            "grid": cgrid + 0.5,
            "keypoints": to_sequence(cgrid) + 0.5,
        }

        return pred

    def loss(self, pred, data):
        raise NotImplementedError
