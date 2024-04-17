import torch
from torch import nn
import torch.nn.functional as F


# Helper used in SDDH
def get_patches(
        tensor: torch.Tensor, required_corners: torch.Tensor, ps: int
) -> torch.Tensor:
    c, h, w = tensor.shape
    corner = (required_corners - ps / 2 + 1).long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - ps)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - ps)
    offset = torch.arange(0, ps)

    kw = {"indexing": "ij"} if torch.__version__ >= "1.10" else {}
    x, y = torch.meshgrid(offset, offset, **kw)
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
    patches = patches.to(corner) + corner[None, None]
    pts = patches.reshape(-1, 2)
    sampled = tensor.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    sampled = sampled.reshape(ps, ps, -1, c)
    assert sampled.shape[:3] == patches.shape[:3]
    return sampled.permute(2, 3, 0, 1)


class SDDH(nn.Module):
    def __init__(
            self,
            dims: int,
            kernel_size: int = 3,
            n_pos: int = 8,
            gate=nn.ReLU(),
            conv2D=False,
            mask=False,
    ):
        super(SDDH, self).__init__()
        self.kernel_size = kernel_size
        self.n_pos = n_pos
        self.conv2D = conv2D
        self.mask = mask

        self.get_patches_func = get_patches

        # estimate offsets
        self.channel_num = 3 * n_pos if mask else 2 * n_pos
        self.offset_conv = nn.Sequential(
            nn.Conv2d(
                dims,
                self.channel_num,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True,
            ),
            gate,
            nn.Conv2d(
                self.channel_num,
                self.channel_num,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        # sampled feature conv
        self.sf_conv = nn.Conv2d(
            dims, dims, kernel_size=1, stride=1, padding=0, bias=False
        )

        # convM
        if not conv2D:
            # deformable desc weights
            agg_weights = torch.nn.Parameter(torch.rand(n_pos, dims, dims))
            self.register_parameter("agg_weights", agg_weights)
        else:
            self.convM = nn.Conv2d(
                dims * n_pos, dims, kernel_size=1, stride=1, padding=0, bias=False
            )

    def forward(self, x, keypoints):
        # x: [B,C,H,W]
        # keypoints: list, [[N_kpts,2], ...] (w,h)
        b, c, h, w = x.shape
        wh = torch.tensor([[w - 1, h - 1]], device=x.device)
        max_offset = max(h, w) / 4.0

        offsets = []
        descriptors = []
        # get offsets for each keypoint
        for ib in range(b):
            xi, kptsi = x[ib], keypoints[ib]
            kptsi_wh = (kptsi / 2 + 0.5) * wh
            N_kpts = len(kptsi)

            if self.kernel_size > 1:
                patch = self.get_patches_func(
                    xi, kptsi_wh.long(), self.kernel_size
                )  # [N_kpts, C, K, K]
            else:
                kptsi_wh_long = kptsi_wh.long()
                patch = (
                    xi[:, kptsi_wh_long[:, 1], kptsi_wh_long[:, 0]]
                    .permute(1, 0)
                    .reshape(N_kpts, c, 1, 1)
                )

            offset = self.offset_conv(patch).clamp(
                -max_offset, max_offset
            )  # [N_kpts, 2*n_pos, 1, 1]
            if self.mask:
                offset = (
                    offset[:, :, 0, 0].view(N_kpts, 3, self.n_pos).permute(0, 2, 1)
                )  # [N_kpts, n_pos, 3]
                offset = offset[:, :, :-1]  # [N_kpts, n_pos, 2]
                mask_weight = torch.sigmoid(offset[:, :, -1])  # [N_kpts, n_pos]
            else:
                offset = (
                    offset[:, :, 0, 0].view(N_kpts, 2, self.n_pos).permute(0, 2, 1)
                )  # [N_kpts, n_pos, 2]
            offsets.append(offset)  # for visualization

            # get sample positions
            pos = kptsi_wh.unsqueeze(1) + offset  # [N_kpts, n_pos, 2]
            pos = 2.0 * pos / wh[None] - 1
            pos = pos.reshape(1, N_kpts * self.n_pos, 1, 2)

            # sample features
            features = F.grid_sample(
                xi.unsqueeze(0), pos, mode="bilinear", align_corners=True
            )  # [1,C,(N_kpts*n_pos),1]
            features = features.reshape(c, N_kpts, self.n_pos, 1).permute(
                1, 0, 2, 3
            )  # [N_kpts, C, n_pos, 1]
            if self.mask:
                features = torch.einsum("ncpo,np->ncpo", features, mask_weight)

            features = torch.selu_(self.sf_conv(features)).squeeze(
                -1
            )  # [N_kpts, C, n_pos]
            # convM
            if not self.conv2D:
                descs = torch.einsum(
                    "ncp,pcd->nd", features, self.agg_weights
                )  # [N_kpts, C]
            else:
                features = features.reshape(N_kpts, -1)[
                           :, :, None, None
                           ]  # [N_kpts, C*n_pos, 1, 1]
                descs = self.convM(features).squeeze()  # [N_kpts, C]

            # normalize
            descs = F.normalize(descs, p=2.0, dim=1)
            descriptors.append(descs)

        return descriptors, offsets

