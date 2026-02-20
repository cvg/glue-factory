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


def _bbox_from_mask(valid, h, w):
    """Bounding box (wmin, hmin, wmax, hmax) from a (B, H, W) bool mask."""
    row_valid = valid.any(-1).int()
    col_valid = valid.any(-2).int()
    hmin = row_valid.argmax(-1).float()
    hmax = (h - row_valid.flip(-1).argmax(-1)).float()
    wmin = col_valid.argmax(-1).float()
    wmax = (w - col_valid.flip(-1).argmax(-1)).float()
    return wmin, hmin, wmax, hmax


def _remap_grid(cgrid, wmin, hmin, wmax, hmax, w, h):
    """Remap a [0..w, 0..h] grid into the bbox [wmin..wmax, hmin..hmax]."""
    cgrid = cgrid / torch.tensor([w, h], device=cgrid.device)[None, :, None, None]
    cgrid = cgrid * torch.stack([(wmax - wmin), (hmax - hmin)], dim=1)[:, :, None, None]
    cgrid = cgrid + torch.stack([wmin, hmin], dim=1)[:, :, None, None]
    return cgrid


class GridExtractor(BaseModel):
    default_conf = {
        "cell_size": 16,
        "sample_offset": False,
        "avoid_borders": False,
        "max_num_keypoints": None,
        "bias_to": "depth",  # "depth" | "covisible" | "image" | None
        # Deprecated, kept for backward compat
        "bias_to_depth": None,
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        b, c, h, w = data["image"].shape
        dtype = data["image"].dtype
        device = data["image"].device
        hc, wc = h // self.conf.cell_size, w // self.conf.cell_size

        # Backward compat: bias_to_depth overrides bias_to if explicitly set
        bias_to = self.conf.bias_to
        if self.conf.bias_to_depth is not None:
            bias_to = "depth" if self.conf.bias_to_depth else None

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
        )
        cgrid = (cgrid + 0.5) * self.conf.cell_size
        if bias_to == "depth" and "depth" in data:
            wmin, hmin, wmax, hmax = _bbox_from_mask(data["depth"] > 0, h, w)
            cgrid = _remap_grid(cgrid, wmin, hmin, wmax, hmax, w, h)
        elif bias_to == "covisible" and "covisible_bbox" in data:
            bbox = data["covisible_bbox"].float()
            wmin, hmin, wmax, hmax = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
            cgrid = _remap_grid(cgrid, wmin, hmin, wmax, hmax, w, h)
        elif bias_to == "image" and "image_size" in data:
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
