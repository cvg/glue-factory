"""
Contains a lightweight ALIKED model that only consists of Backbone encoder and descriptor branch.
Why need this model? As descriptors are sparsely computed(no full map) we need to be able to generate descriptors
on demand for detected keypoints. As we don't know beforehand which points that will be, we cannot generate
ground-truth beforehand.

Input=Img+Keypoint-locations
Output=descriptors
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.models import resnet
from gluefactory.models.backbones.backbone_encoder import ResBlock,ConvBlock

from gluefactory.models.base_model import BaseModel

# coordinates system
#  ------------------------------>  [ x: range=-1.0~1.0; w: range=0~W ]
#  | -----------------------------
#  | |                           |
#  | |                           |
#  | |                           |
#  | |         image             |
#  | |                           |
#  | |                           |
#  | |                           |
#  | |---------------------------|
#  v
# [ y: range=-1.0~1.0; h: range=0~H ]


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


def simple_nms(scores: torch.Tensor, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""

    zeros = torch.zeros_like(scores)
    max_mask = scores == torch.nn.functional.max_pool2d(
        scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
    )

    for _ in range(2):
        supp_mask = (
            torch.nn.functional.max_pool2d(
                max_mask.float(),
                kernel_size=nms_radius * 2 + 1,
                stride=1,
                padding=nms_radius,
            )
            > 0
        )
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == torch.nn.functional.max_pool2d(
            supp_scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


class InputPadder(object):
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, h: int, w: int, divis_by: int = 8):
        self.ht = h
        self.wd = w
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, x: torch.Tensor):
        assert x.ndim == 4
        return F.pad(x, self._pad, mode="replicate")

    def unpad(self, x: torch.Tensor):
        assert x.ndim == 4
        ht = x.shape[-2]
        wd = x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        mask=False,
    ):
        super(DeformableConv2d, self).__init__()

        self.padding = padding
        self.mask = mask

        self.channel_num = (
            3 * kernel_size * kernel_size if mask else 2 * kernel_size * kernel_size
        )
        self.offset_conv = nn.Conv2d(
            in_channels,
            self.channel_num,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.0

        out = self.offset_conv(x)
        if self.mask:
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)
        else:
            offset = out
            mask = None
        offset = offset.clamp(-max_offset, max_offset)
        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=mask,
        )
        return x


def get_conv(
    inplanes,
    planes,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
    conv_type="conv",
    mask=False,
):
    if conv_type == "conv":
        conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    elif conv_type == "dcn":
        conv = DeformableConv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=_pair(padding),
            bias=bias,
            mask=mask,
        )
    else:
        raise TypeError
    return conv


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


class ALIKED_Light(BaseModel):
    default_conf = {
        "model_name": "aliked-n16",
        "max_num_keypoints": -1,
        "detection_threshold": 0.2,
        "force_num_keypoints": False,
        "pretrained": True,
        "nms_radius": 2,
    }

    checkpoint_url = "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth"

    n_limit_max = 20000

    cfgs = {
        "aliked-t16": {
            "c1": 8,
            "c2": 16,
            "c3": 32,
            "c4": 64,
            "dim": 64,
            "K": 3,
            "M": 16,
        },
        "aliked-n16": {
            "c1": 16,
            "c2": 32,
            "c3": 64,
            "c4": 128,
            "dim": 128,
            "K": 3,
            "M": 16,
        },
        "aliked-n16rot": {
            "c1": 16,
            "c2": 32,
            "c3": 64,
            "c4": 128,
            "dim": 128,
            "K": 3,
            "M": 16,
        },
        "aliked-n32": {
            "c1": 16,
            "c2": 32,
            "c3": 64,
            "c4": 128,
            "dim": 128,
            "K": 3,
            "M": 32,
        },
    }

    required_data_keys = ["image", "keypoints"]

    def _init(self, conf):
        if conf.force_num_keypoints:
            assert conf.detection_threshold <= 0 and conf.max_num_keypoints > 0
        # get configurations
        c1, c2, c3, c4, dim, K, M = [v for _, v in self.cfgs[conf.model_name].items()]
        conv_types = ["conv", "conv", "dcn", "dcn"]
        conv2D = False
        mask = False

        # build model
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d
        self.gate = nn.SELU(inplace=True)
        self.block1 = ConvBlock(3, c1, self.gate, self.norm, conv_type=conv_types[0])
        self.block2 = ResBlock(
            c1,
            c2,
            1,
            nn.Conv2d(c1, c2, 1),
            gate=self.gate,
            norm_layer=self.norm,
            conv_type=conv_types[1],
        )
        self.block3 = ResBlock(
            c2,
            c3,
            1,
            nn.Conv2d(c2, c3, 1),
            gate=self.gate,
            norm_layer=self.norm,
            conv_type=conv_types[2],
            mask=mask,
        )
        self.block4 = ResBlock(
            c3,
            c4,
            1,
            nn.Conv2d(c3, c4, 1),
            gate=self.gate,
            norm_layer=self.norm,
            conv_type=conv_types[3],
            mask=mask,
        )
        self.conv1 = resnet.conv1x1(c1, dim // 4)
        self.conv2 = resnet.conv1x1(c2, dim // 4)
        self.conv3 = resnet.conv1x1(c3, dim // 4)
        self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.upsample32 = nn.Upsample(
            scale_factor=32, mode="bilinear", align_corners=True
        )
        self.desc_head = SDDH(dim, K, M, gate=self.gate, conv2D=conv2D, mask=mask)

        # load pretrained
        if conf.pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                self.checkpoint_url.format(conf.model_name), map_location="cpu"
            )
            self.load_state_dict(state_dict, strict=False)

    def extract_dense_map(self, image):
        # Pads images such that dimensions are divisible by
        div_by = 2**5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)
        image = padder.pad(image)

        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool4(x2)
        x3 = self.block3(x3)  # B x c3 x H/8 x W/8
        x4 = self.pool4(x3)
        x4 = self.block4(x4)  # B x dim x H/32 x W/32
        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//8 x W//8
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//32 x W//32
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample8(x3)  # B x dim//4 x H x W
        x4_up = self.upsample32(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)
        # ================================== score head
        feature_map = torch.nn.functional.normalize(x1234, p=2, dim=1)

        # Unpads images
        feature_map = padder.unpad(feature_map)

        return feature_map

    def _forward(self, data):
        image = data["image"]
        keypoints = data["keypoints"]
        feature_map = self.extract_dense_map(image)
        descriptors, offsets = self.desc_head(feature_map, keypoints)

        _, _, h, w = image.shape
        # no padding required,
        # we can set detection_threshold=-1 and conf.max_num_keypoints
        return {
            "aliked_descriptors": torch.stack(descriptors),  # B N D
        }

    def loss(self, pred, data):
        raise NotImplementedError
