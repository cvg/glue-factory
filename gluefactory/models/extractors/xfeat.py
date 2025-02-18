"""
"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import logging
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from gluefactory.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class InterpolateSparse2d(nn.Module):
    """Efficiently interpolate tensor at given sparse 2D positions."""

    def __init__(self, mode="bicubic", align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Normalize coords to [-1,1]."""
        return (
            2.0 * (x / (torch.tensor([W - 1, H - 1], device=x.device, dtype=x.dtype)))
            - 1.0
        )

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode=self.mode, align_corners=False)
        return x.permute(0, 2, 3, 1).squeeze(-2)


class BasicLayer(nn.Module):
    """
    Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class ModelName(Enum):
    XFEAT = "xfeat"
    XFEAT_DENSE = "xfeat-dense"


class XFeatModel(nn.Module):
    """
    Implementation of keypoint detection and description architecture described in
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self, model_name: str = ModelName.XFEAT.value):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)
        self.model_name = model_name

        ########### ⬇️ CNN Backbone & Heads ⬇️ ###########
        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride=4), nn.Conv2d(1, 24, 1, stride=1, padding=0)
        )

        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1),
            BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        )

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )
        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )

        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0),
        )

        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        ########### ⬇️ Fine Matcher MLP ⬇️ ###########
        if self.model_name == ModelName.XFEAT_DENSE.value:
            self.fine_matcher = nn.Sequential(
                nn.Linear(128, 512),
                nn.BatchNorm1d(512, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(512, 64),
            )

    def _unfold2d(self, x: torch.Tensor, ws: int = 2) -> torch.Tensor:
        """
        Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(B, C, H // ws, W // ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)

    def forward(self, x: torch.Tensor) -> dict:
        """
        input:
            x -> torch.Tensor(B, C, H, W) grayscale or rgb images
        return:
            desc_map ->  torch.Tensor(B, 64, H/8, W/8) dense local features for descriptors
            kpt_logit_map ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
            relmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

        """
        # dont backprop through normalization
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        # main backbone
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        # pyramid fusion
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
        desc_map = self.block_fusion(x3 + x4 + x5)

        # heads
        rel_map = self.heatmap_head(desc_map)  # Reliability map
        kpt_logit_map = self.keypoint_head(
            self._unfold2d(x, ws=8)
        )  # Keypoint map logits

        return {
            "dense_descriptors": desc_map,
            "keypoint_logit_map": kpt_logit_map,
            "reliability_map": rel_map,
        }


class XFeat(BaseModel):
    """
    Implements the inference module for XFeat and XFeat-Dense models.
    """

    default_conf = {
        "model_name": ModelName.XFEAT.value,
        "max_num_keypoints": 4096,
        "detection_threshold": 0.05,
        "pretrained": True,
        "get_sparse_outputs": True,
        "force_num_keypoints": True,
        "NMS": {"threshold": 0.05, "kernel_size": 5},
        "preprocess": False,  # Preprocess input images to be divisible by 32
    }

    checkpoint_url = (
        "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt"
    )

    required_data_keys = ["image"]

    def _init(self, conf: OmegaConf):

        if self.conf.preprocess:
            assert (
                self.conf.get_sparse_outputs
            ), "Preprocessing requires get_sparse_outputs=True"
        if self.conf.model_name not in [m.value for m in ModelName]:
            raise ValueError(f"Invalid model_name: {self.conf.model_name}")

        self.net = XFeatModel(self.conf.model_name)
        self.top_k = self.conf.max_num_keypoints
        self.detection_threshold = self.conf.detection_threshold

        if self.conf.pretrained:
            state_dict = torch.hub.load_state_dict_from_url(self.checkpoint_url)

            if self.conf.model_name != ModelName.XFEAT_DENSE.value:
                state_dict = {
                    k: v for k, v in state_dict.items() if "fine_matcher" not in k
                }

            if isinstance(state_dict, str):
                logger.info("loading state_dict from: " + state_dict)
                self.net.load_state_dict(torch.load(state_dict), strict=True)
            else:
                self.net.load_state_dict(state_dict, strict=True)
            logger.info("XFeat loaded with pretrained weights.")
            self.set_initialized()

        self.interpolator = InterpolateSparse2d("bicubic")

    def preprocess_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Guarantee that image is divisible by 32 to avoid aliasing artifacts."""
        if isinstance(x, np.ndarray) and len(x.shape) == 3:
            x = torch.tensor(x).permute(2, 0, 1)[None]
        x = x.float()

        H, W = x.shape[-2:]
        _H, _W = (H // 32) * 32, (W // 32) * 32
        rh, rw = H / _H, W / _W

        x = F.interpolate(x, (_H, _W), mode="bilinear", align_corners=False)
        return x, rh, rw

    def get_kpts_heatmap(
        self, kpts: torch.Tensor, softmax_temp: float = 1.0
    ) -> torch.Tensor:
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap

    def NMS(self, x: torch.Tensor) -> torch.Tensor:
        """
        Non-Maximum Suppression for keypoint detection.

        input:
            x -> torch.Tensor(B, 1, H, W): keypoint heatmap
        return:
            torch.Tensor(B, N, 2): keypoints (x,y)
        """
        kernel_size = self.conf.NMS.kernel_size
        threshold = self.conf.NMS.threshold

        B, _, H, W = x.shape
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

        # Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, : len(pos_batched[b]), :] = pos_batched[b]

        return pos

    def get_sparse_outputs(
        self, data: dict, net_output: dict, rh: float = 1.0, rw: float = 1.0
    ) -> dict:
        """
        Compute sparse keypoints & descriptors from dense features.
        Supports both single and multi-image inputs.

        input:
            data -> Dict: contains 'image' key with torch.Tensor(B, C, H, W) grayscale or rgb images
            net_output -> Dict: contains 'dense_descriptors', 'score_map', 'reliability_map' keys
            rh, rw -> float: height and width ratios for preprocessing
        return:
            Dict:
                'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
                'scores'       ->   torch.Tensor(N,): keypoint scores
                'descriptors'  ->   torch.Tensor(N, 64): local features
        """
        x = data["image"]

        desc_map = net_output["dense_descriptors"]
        kpt_logit_map = net_output["keypoint_logit_map"]
        rel_map = net_output["reliability_map"]

        B, _, H, W = x.shape

        desc_map = F.normalize(desc_map, dim=1)

        # Convert logits to heatmap and extract kpts
        kpt_score_map = self.get_kpts_heatmap(kpt_logit_map)
        mkpts = self.NMS(kpt_score_map)

        # Compute reliability scores
        _nearest = InterpolateSparse2d("nearest")
        _bilinear = InterpolateSparse2d("bilinear")

        scores = (
            _nearest(kpt_score_map, mkpts, H, W) * _bilinear(rel_map, mkpts, H, W)
        ).squeeze(-1)

        # Filter out keypoints that are (0,0)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        # Select top-k features
        idxs = torch.argsort(-scores)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, : self.top_k]
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, : self.top_k]
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, : self.top_k]

        # Interpolate descriptors at kpts positions
        feats = self.interpolator(desc_map, mkpts, H=H, W=W)

        # L2-Normalize
        feats = F.normalize(feats, dim=-1)

        # Bring keypoints to original resolution
        if self.conf.preprocess:
            mkpts = mkpts * torch.tensor([rw, rh], device=mkpts.device).view(1, 1, -1)

        valid = scores > 0

        keypoints_list = []
        scores_list = []
        descriptors_list = []

        if not self.conf.force_num_keypoints:
            for b in range(B):
                keypoints_list.append(mkpts[b][valid[b]])
                scores_list.append(scores[b][valid[b]])
                descriptors_list.append(feats[b][valid[b]])

            return {
                "keypoints": keypoints_list,
                "keypoint_scores": scores_list,
                "descriptors": descriptors_list,
            }
        else:
            return {
                "keypoints": mkpts,
                "keypoint_scores": scores,
                "descriptors": feats,
            }

    def _forward(self, data: dict) -> dict:
        """
        input:
            data -> Dict: contains 'image' key with torch.Tensor(B, C, H, W) grayscale or rgb images
        return:
            Dict: contains 'dense_descriptors', 'keypoint_logit_map', 'reliability_map' keys
                  additionally contains 'keypoints', 'scores', 'descriptors' keys if get_sparse_outputs is True
        """
        image = data["image"]

        rh, rw = 1.0, 1.0
        if self.conf.preprocess:
            image, rh, rw = self.preprocess_tensor(image)

        pred = self.net(image)

        if self.conf.get_sparse_outputs:
            pred_sparse = self.get_sparse_outputs(data, pred, rh, rw)
            pred = {**pred, **pred_sparse}

        return pred

    def loss(self, pred, data):
        raise NotImplementedError("Loss function not implemented for XFeat model.")
