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
    default_conf = {
        "cell_size": 16,
        "sample_offset": False,
        "avoid_borders": False,
        "max_num_keypoints": None,
        "bias_to_depth": True,
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        b, c, h, w = data["image"].shape
        dtype = data["image"].dtype
        device = data["image"].device
        hc, wc = h // self.conf.cell_size, w // self.conf.cell_size
        if self.conf.avoid_borders:
            hrange = torch.arange(1, hc - 1, device=device, dtype=dtype)
            wrange = torch.arange(1, wc - 1, device=device, dtype=dtype)
        else:
            hrange = torch.arange(hc, device=device, dtype=dtype)
            wrange = torch.arange(wc, device=device, dtype=dtype)
        cgrid = (
            torch.stack(
                torch.meshgrid(
                    hrange,
                    wrange,
                    indexing="ij",
                )[::-1],
                dim=0,
            )
            .unsqueeze(0)
            .repeat([b, 1, 1, 1])
            * self.conf.cell_size
            + self.conf.cell_size / 2
        ) + 0.5

        if self.conf.bias_to_depth and "depth" in data:
            valid = data["depth"] > 0
            row_valid = valid.any(-1).int()
            col_valid = valid.any(-2).int()
            hmin = row_valid.argmax(-1)
            hmax = h - row_valid.flip(-1).argmax(-1)
            wmin = col_valid.argmax(-1)
            wmax = w - col_valid.flip(-1).argmax(-1)
            cgrid = cgrid / torch.tensor([w, h], device=device)[None, :, None, None]
            cgrid = (
                cgrid
                * torch.stack(
                    [(wmax - wmin), (hmax - hmin)],
                    dim=1,
                )[:, :, None, None]
            )
            cgrid = (
                cgrid
                + torch.stack(
                    [wmin, hmin],
                    dim=1,
                )[:, :, None, None]
            )
        elif "image_size" in data:
            cgrid = cgrid * (
                data["image_size"][:, :, None, None]
                / torch.tensor([w, h], device=device)[None, :, None, None]
            )
        pred = {
            "grid": cgrid,
            "keypoints": to_sequence(cgrid),
        }

        if self.conf.sample_offset and self.training:
            offset = (torch.rand_like(pred["keypoints"])) * self.conf.cell_size * 0.5
            pred["keypoints"] = pred["keypoints"] + offset

        if self.conf.max_num_keypoints is not None and self.training:
            pred["keypoints"] = torch.vmap(
                lambda pk: pk[
                    torch.randperm(pk.shape[0], device=pk.device)[
                        : self.conf.max_num_keypoints
                    ]
                ],
                randomness="different",
            )(pred["keypoints"])

        return pred

    def loss(self, pred, data):
        raise NotImplementedError
