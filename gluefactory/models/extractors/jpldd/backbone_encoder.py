"""
This file contains backbone encoders that compute a hidden feature map

- ALIKED (configurable)
"""
import torch
from torch import nn
from torchvision.models import resnet
import torchvision
from torch.nn.modules.utils import _pair
from typing import Optional, Callable


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
    def __init__(self, conf):
        super().__init__()
        # get configurations
        c1, c2, c3, c4, dim = conf["c1"], conf["c2"], conf["c3"], conf["c4"], conf["dim"]
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


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            conv_type: str = "conv",
            mask: bool = False,
    ):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = get_conv(
            in_channels, out_channels, kernel_size=3, conv_type=conv_type, mask=mask
        )
        self.bn1 = norm_layer(out_channels)
        self.conv2 = get_conv(
            out_channels, out_channels, kernel_size=3, conv_type=conv_type, mask=mask
        )
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


# modified based on torchvision\models\resnet.py#27->BasicBlock
class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            conv_type: str = "conv",
            mask: bool = False,
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("ResBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = get_conv(
            inplanes, planes, kernel_size=3, conv_type=conv_type, mask=mask
        )
        self.bn1 = norm_layer(planes)
        self.conv2 = get_conv(
            planes, planes, kernel_size=3, conv_type=conv_type, mask=mask
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out
