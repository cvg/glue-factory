import torch
import torchvision
from torch import nn
from torch.nn.modules.utils import _pair
from typing import Optional, Callable


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







