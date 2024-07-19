"""
Contains a lightweight ALIKED model that only consists of Backbone encoder and descriptor branch.
Why need this model? As descriptors are sparsely computed(no full map) we need to be able to generate descriptors
on demand for detected keypoints. As we don't know beforehand which points that will be, we cannot generate
ground-truth beforehand.

Input=Img+Keypoint-locations
Output=descriptors
"""

import torch
from torch import nn
from torchvision.models import resnet

from gluefactory.models.backbones.backbone_encoder import ConvBlock, ResBlock
from gluefactory.models.base_model import BaseModel
from gluefactory.models.extractors.aliked import SDDH, InputPadder

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
