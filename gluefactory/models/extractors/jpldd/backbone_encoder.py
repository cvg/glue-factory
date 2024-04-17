"""
This file contains backbone encoders that compute a hidden feature map

- ALIKED (configurable)
"""
import torch
from torch import nn
from torchvision.models import resnet

from gluefactory.models.extractors.jpldd.common_components import ConvBlock, ResBlock

aliked_cfgs = {
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


class AlikedEncoder(nn.Module):
    def _init(self, c1, c2, c3, c4, dim):
        # get configurations
        conv_types = ["conv", "conv", "dcn", "dcn"]
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
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

        return x1234
