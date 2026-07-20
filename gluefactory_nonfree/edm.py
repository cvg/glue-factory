"""Wrapper for the EDM (Efficient Deep Matching) model."""

import logging
import sys

import torch
from kornia.color import rgb_to_grayscale
from yacs.config import CfgNode

from gluefactory import settings
from gluefactory.models.base_model import BaseModel

logger = logging.getLogger(__name__)

edm_root = settings.THIRD_PARTY_PATH / "EDM"
sys.path.append(str(edm_root))

# EDM's src.utils.misc imports lightning which we don't need.
# Mock it before importing EDM to avoid the heavy dependency.
import types

if "lightning" not in sys.modules:
    _lightning = types.ModuleType("lightning")
    _pytorch = types.ModuleType("lightning.pytorch")
    _utilities = types.ModuleType("lightning.pytorch.utilities")
    _utilities.rank_zero_only = lambda fn: fn
    _pytorch.utilities = _utilities
    _lightning.pytorch = _pytorch
    sys.modules["lightning"] = _lightning
    sys.modules["lightning.pytorch"] = _pytorch
    sys.modules["lightning.pytorch.utilities"] = _utilities

from src.config.default import get_cfg_defaults  # noqa: E402
from src.edm.edm import EDM  # noqa: E402


def lower_config(yacs_cfg):
    """Recursively convert YACS CfgNode keys to lowercase."""
    if not isinstance(yacs_cfg, CfgNode):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


class EdmMatcher(BaseModel):
    default_conf = {
        "weights": None,
        "test_res": [832, 832],  # [H, W] test resolution for RoPE scaling
        "mconf_thr": 0.2,
        "topk": 2048,
        "zero_pad": True,  # pad to square with mask (EDM's MegaDepth pipeline)
    }
    required_data_keys = ["view0", "view1"]

    def _init(self, conf):
        cfg = get_cfg_defaults()
        cfg.defrost()
        cfg.EDM.DEPLOY = False
        cfg.EDM.COARSE.MCONF_THR = conf.mconf_thr
        cfg.EDM.COARSE.TOPK = conf.topk
        # Train res is always 832×832; test res controls RoPE PE scaling
        cfg.EDM.TRAIN_RES_H = 832
        cfg.EDM.TRAIN_RES_W = 832
        cfg.EDM.TEST_RES_H = conf.test_res[0]
        cfg.EDM.TEST_RES_W = conf.test_res[1]
        cfg.EDM.NECK.NPE = [832, 832, conf.test_res[0], conf.test_res[1]]
        cfg.freeze()

        _config = lower_config(cfg)
        self.net = EDM(config=_config["edm"])

        if conf.weights is not None:
            weights_path = settings.root / conf.weights
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"EDM weights not found at {weights_path}. "
                    "Download from the EDM repo (Google Drive links in README)."
                )
            ckpt = torch.load(
                str(weights_path),
                map_location="cpu",
                weights_only=False,
            )
            self.net.load_state_dict(ckpt["state_dict"])
            logger.info("Loaded EDM weights from %s", weights_path)

        self.set_initialized()

    def _pad_to_square(self, img):
        """Match EDM's original MegaDepth data pipeline exactly:
        1. Crop to dimensions divisible by 8 (backbone stride, MGDPT_DF=8)
        2. Zero-pad bottom-right to square (pad_bottom_right)
        3. Return mask at 1/8 coarse resolution (coarse_scale=0.125)
        """
        B, C, H, W = img.shape
        # Round down to divisible by 8 — matches get_divisible_wh(w, h, df=8)
        Hc, Wc = H // 8 * 8, W // 8 * 8
        if Hc != H or Wc != W:
            img = img[:, :, :Hc, :Wc]

        S = max(Hc, Wc)
        if Hc == S and Wc == S:
            # Already square and divisible by 8 — no mask needed
            return img, None

        # Bottom-right zero padding to square — matches pad_bottom_right()
        padded = img.new_zeros(B, C, S, S)
        padded[:, :, :Hc, :Wc] = img

        # Mask at 1/8 coarse resolution — matches megadepth.py:155-162
        # F.interpolate(mask, scale_factor=coarse_scale, mode="nearest")
        mask = img.new_zeros(B, S // 8, S // 8)
        mask[:, :Hc // 8, :Wc // 8] = 1.0

        return padded, mask

    def _forward(self, data):
        img0 = data["view0"]["image"]  # (B, C, H, W)
        img1 = data["view1"]["image"]

        # EDM expects grayscale (B, 1, H, W) in [0, 1]
        if img0.shape[1] == 3:
            img0 = rgb_to_grayscale(img0)
        if img1.shape[1] == 3:
            img1 = rgb_to_grayscale(img1)

        batch = {"image0": img0, "image1": img1}

        if self.conf.zero_pad:
            img0, mask0 = self._pad_to_square(img0)
            img1, mask1 = self._pad_to_square(img1)
            batch = {"image0": img0, "image1": img1}
            if mask0 is not None or mask1 is not None:
                B = img0.shape[0]
                S0 = img0.shape[2]
                S1 = img1.shape[2]
                batch["mask0"] = mask0 if mask0 is not None else img0.new_ones(B, S0 // 8, S0 // 8)
                batch["mask1"] = mask1 if mask1 is not None else img1.new_ones(B, S1 // 8, S1 // 8)

        # Run EDM — mutates batch dict with mkpts0_f, mkpts1_f, mconf, m_bids
        self.net(batch)

        mkpts0 = batch["mkpts0_f"]  # (M, 2)
        mkpts1 = batch["mkpts1_f"]  # (M, 2)
        mconf = batch["mconf"]  # (M,)
        M = mkpts0.shape[0]

        pred = {
            "keypoints0": mkpts0[None].float(),
            "keypoints1": mkpts1[None].float(),
            "matches0": torch.arange(M, device=img0.device)[None],
            "matches1": torch.arange(M, device=img0.device)[None],
            "matching_scores0": mconf[None],
            "matching_scores1": mconf[None],
        }
        return pred

    def loss(self, pred, data):
        raise NotImplementedError("EDM wrapper is for evaluation only.")
