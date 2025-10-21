import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from ..base_model import BaseModel
from .lightglue import (
    CrossBlock,
    LearnableFourierPositionalEncoding,
    SelfBlock,
    TokenConfidence,
    apply_cached_rotary_emb,
    filter_matches,
    normalize_keypoints,
)
from .lightgluestick import Attention, MatchAssignment

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True
ETH_EPS = 1e-8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_mask(lines_junc_idx, eye_mask, num_nodes):
    # Get batch size and number of connections
    bs = lines_junc_idx.shape[0]
    mask = eye_mask[:, : num_nodes, : num_nodes].clone()
    # Extract the start and end nodes
    start_nodes = lines_junc_idx[:, 0::2]  # Even indexed nodes
    end_nodes = lines_junc_idx[:, 1::2]  # Odd indexed nodes

    # Use broadcasting to fill the mask
    mask[torch.arange(bs).unsqueeze(1), start_nodes, end_nodes] = 1.0
    mask[torch.arange(bs).unsqueeze(1), end_nodes, start_nodes] = 1.0  # Ensure symmetry

    return mask

class LineLayer(nn.Module):
    def __init__(
            self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
            self,
            x: torch.Tensor,
            encoding: torch.Tensor,
            mask: Optional[torch.Tensor] = None,

    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))

        return x + self.ffn(torch.cat([x, message], -1))
    
class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.line_layer = LineLayer(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        desc0 = self.self_attn(desc0, encoding0)
        desc1 = self.self_attn(desc1, encoding1)

        n_endpoints0 = mask0.shape[-1]
        n_endpoints1 = mask1.shape[-1]

        desc0[:, : n_endpoints0, :] = self.line_layer(desc0[:, : n_endpoints0, :], \
                                                      encoding0[:, :, :, : n_endpoints0, :], mask0)
        desc1[:, : n_endpoints1, :] = self.line_layer(desc1[:, : n_endpoints1, :], \
                                                      encoding1[:, :, :, : n_endpoints1, :], mask1)

        return self.cross_attn(desc0, desc1)
    
class LightGlueStick(BaseModel):
    default_conf = {
        "name": "lightgluestick",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,  # either a path or the name of pretrained weights (disk, ...)
        "max_num_lines": 600,
    }

    required_data_keys = [
        "view0",
        "view1",
        "keypoints0",
        "keypoints1",
        "descriptors0",
        "descriptors1",
        "keypoint_scores0",
        "keypoint_scores1",
        "lines0",
        "lines1",
        "lines_junc_idx0",
        "lines_junc_idx1",
        "line_scores0",
        "line_scores1",
    ]

    url = (
        "https://github.com/aubingazhib/LightGlueStick/"
        "releases/download/v1.0.0/lightgluestick.tar"
    )

    def _init(self, conf) -> None:
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, conf.flash) for _ in range(n)]
        )
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n)])

        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor(
                [self.confidence_threshold(i) for i in range(self.conf.n_layers)]
            ),
        )

        self.eye_mask = (
            torch.eye(self.conf.max_num_lines * 2, dtype=torch.float32)
            .unsqueeze(0)
            .to(DEVICE)
        )
        state_dict = None

        if conf.weights is not None:
            weights_path = Path(conf.weights)
            if weights_path.exists() and weights_path.is_file():
                # Load directly from provided weight file
                state_dict = torch.load(str(weights_path), map_location="cpu")
            else:
                # Download into default torch cache (~/.cache/torch/hub/checkpoints)
                state_dict = torch.hub.load_state_dict_from_url(
                    self.url, map_location="cpu"
                )
        else:
            # No weights provided -> use default torch cache
            state_dict = torch.hub.load_state_dict_from_url(
                self.url, map_location="cpu"
            )

        if state_dict:
            state_dict = state_dict["model"]
            state_dict = {
                k[8:]: v for k, v in state_dict.items() if k.startswith("matcher.")
            }
            self.load_state_dict(state_dict, strict=False)

    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )

    def _forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape

        device = kpts0.device

        n_lines0, n_lines1 = data["lines0"].shape[1], data["lines1"].shape[1]

        pred = {}

        if m == 0 or n == 0:
            # No detected keypoints nor lines
            pred["log_assignment"] = torch.zeros(
                b, m, n, dtype=torch.float, device=device
            )
            pred["matches0"] = torch.full((b, m), -1, device=device, dtype=torch.int64)
            pred["matches1"] = torch.full((b, n), -1, device=device, dtype=torch.int64)
            pred["matching_scores0"] = torch.zeros(
                (b, m), device=device, dtype=torch.float32
            )
            pred["matching_scores1"] = torch.zeros(
                (b, n), device=device, dtype=torch.float32
            )
            pred["line_log_assignment"] = torch.zeros(
                b, n_lines0, n_lines1, dtype=torch.float, device=device
            )
            pred["line_matches0"] = torch.full(
                (b, n_lines0), -1, device=device, dtype=torch.int64
            )
            pred["line_matches1"] = torch.full(
                (b, n_lines1), -1, device=device, dtype=torch.int64
            )
            pred["line_matching_scores0"] = torch.zeros(
                (b, n_lines0), device=device, dtype=torch.float32
            )
            pred["line_matching_scores1"] = torch.zeros(
                (b, n_lines1), device=device, dtype=torch.float32
            )
            return pred

        # [b, num_lines * 2]
        lines_junc_idx0 = data["lines_junc_idx0"].flatten(1, 2)
        lines_junc_idx1 = data["lines_junc_idx1"].flatten(1, 2)

        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")

        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0 and not self.training
        do_point_pruning = self.conf.width_confidence > 0 and not self.training

        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None

        n_endpoints0 = lines_junc_idx0.max() + 1
        n_endpoints1 = lines_junc_idx1.max() + 1

        # pre-compute masks for LG-LMP
        mask0 = (
            create_mask(lines_junc_idx0, self.eye_mask, n_endpoints0)
            .unsqueeze(1)
            .bool()
        )
        mask1 = (
            create_mask(lines_junc_idx1, self.eye_mask, n_endpoints1)
            .unsqueeze(1)
            .bool()
        )

        for i in range(self.conf.n_layers):
            desc0, desc1 = self.transformers[i](
                desc0, desc1, encoding0, encoding1, mask0, mask1
            )

            # only for eval
            if do_early_stop:
                assert b == 1
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m, :], token1[..., :n, :], i, m + n):
                    break
            if do_point_pruning:
                assert b == 1
                scores0 = self.log_assignment[i].get_matchability(desc0)

                scores0[0, :n_endpoints0] = 1.0
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
                scores1 = self.log_assignment[i].get_matchability(desc1)

                scores1[0, :n_endpoints1] = 1.0
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _, line_scores, raw_line_scores = self.log_assignment[i](
            desc0, desc1, lines_junc_idx0, lines_junc_idx1
        )
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "log_assignment": scores,
            "prune0": prune0,
            "prune1": prune1,
            "early_exit_layer_idx": i + 1,
        }

        if n_lines0 > 0 and n_lines1 > 0:
            m0_lines, m1_lines, mscores0_lines, mscores1_lines = filter_matches(
                line_scores, self.conf.filter_threshold
            )
            pred["line_log_assignment"] = line_scores
            pred["line_matches0"] = m0_lines
            pred["line_matches1"] = m1_lines
            pred["line_matching_scores0"] = mscores0_lines
            pred["line_matching_scores1"] = mscores1_lines
            pred["raw_line_scores"] = raw_line_scores
        else:
            line_scores = torch.zeros(
                b, n_lines0, n_lines1, dtype=torch.float, device=device
            )
            m0_lines = torch.full((b, n_lines0), -1, device=device, dtype=torch.int64)
            m1_lines = torch.full((b, n_lines1), -1, device=device, dtype=torch.int64)
            mscores0_lines = torch.zeros(
                (b, n_lines0), device=device, dtype=torch.float32
            )
            mscores1_lines = torch.zeros(
                (b, n_lines1), device=device, dtype=torch.float32
            )
            raw_line_scores = torch.zeros(
                b, n_lines0, n_lines1, dtype=torch.float, device=device
            )

        return pred

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]

    def loss(self, pred, data):
        raise NotImplementedError()

    def metrics(self, pred, data):
        raise NotImplementedError()
