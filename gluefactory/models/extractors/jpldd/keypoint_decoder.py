"""
keypoint_decoder.py: contains possible keypoint decoders

- ALIKED SMH (ScoreMapHead)
"""
import torch
from torch import nn
from torchvision.models import resnet


class SMH(nn.Module):
    def __init__(self, input_dim):
        super(SMH, self).__init__()
        self.score_head = nn.Sequential(
            resnet.conv1x1(input_dim, 8),
            self.gate,
            resnet.conv3x3(8, 4),
            self.gate,
            resnet.conv3x3(4, 4),
            self.gate,
            resnet.conv3x3(4, 1),
        )

    def forward(self, x):
        # expects feature map not normalized
        return torch.sigmoid(self.score_head(x))
