"""Wrapper around the UFM model for image matching.

Paper: UniFlowMatch: A Unified Framework for Optical Flow and Wide-Baseline Matching.
Project page: https://uniflowmatch.github.io
Code: https://github.com/UniFlowMatch/UFM

License: CC BY-NC-SA 4.0 (non-commercial)

Main differences to the original demo code:
- Unified gluefactory API (BaseModel, warp/certainty output format).
- Handles arbitrary processing and output resolutions.
- Interface to match sparse keypoints via dense correspondences.
"""

import logging

import torch
import torch.nn.functional as F

try:
    from uniflowmatch.models.ufm import UniFlowMatchConfidence
except ImportError:
    raise ImportError(
        "Please install the 'uniflowmatch' package to use UFM models: "
        "`cd UFM && pip install --no-deps -e UniCeption && pip install --no-deps -e .`"
    )

from ...utils import misc
from .. import base_model

logger = logging.getLogger(__name__)


def flow_to_warp(flow: torch.Tensor, certainty: torch.Tensor) -> dict:
    """Convert UFM's pixel-offset flow to gluefactory's normalized warp format.

    Args:
        flow: (B, 2, H, W) pixel offsets (dx, dy) from source to target.
        certainty: (B, H, W) covisibility confidence in [0, 1].

    Returns:
        dict with:
            "warp": (B, H, W, 2) normalized target coords in [-1, 1].
            "certainty": (B, H, W) in [0, 1], zeroed where out-of-bounds.
    """
    B, _, H, W = flow.shape
    device = flow.device

    # Normalized pixel-center grid — same align_corners=False convention as RoMa.
    gy = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device)
    gx = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

    # Convert pixel offsets → normalized offsets (1 px = 2/W in x, 2/H in y).
    warp_x = grid_x[None] + flow[:, 0] * (2.0 / W)
    warp_y = grid_y[None] + flow[:, 1] * (2.0 / H)
    warp = torch.stack([warp_x, warp_y], dim=-1)  # (B, H, W, 2)

    certainty = certainty.clone()
    certainty[(warp.abs() > 1).any(dim=-1)] = 0.0
    return {"warp": warp.clamp(-1, 1), "certainty": certainty}


class UFM(base_model.BaseModel):
    """UFM (UniFlowMatch) dense matcher."""

    default_conf = {
        "weights": "UFM-Refine",  # HF model ID suffix after "infinity1096/"
        "output_hw": None,  # if set + upsample_preds, bilinear-upsample flow here
        "upsample_preds": False,
        "add_cycle_error": False,
        "sample_num_matches": 0,  # >0: sample N sparse matches from dense field
        "sample_mode": "threshold_balanced",
        "filter_threshold": 0.05,
        "max_kp_error": 2.0,
        "mutual_check": True,
        "sparse_to_dense": False,
    }
    required_data_keys = ["view0", "view1"]

    def _init(self, conf):
        self._matcher = UniFlowMatchConfidence.from_pretrained(
            f"infinity1096/{conf.weights}"
        ).eval()
        # inference_resolution is [W, H]; convert to (H, W) for F.interpolate.
        w, h = self._matcher.inference_resolution[0]
        self._internal_hw = (h, w)
        self.set_initialized(True)

    def _forward(self, data):
        pred_qtos = self.estimate_warp(data["view0"]["image"], data["view1"]["image"])
        pred_stoq = self.estimate_warp(data["view1"]["image"], data["view0"]["image"])
        pred = {**misc.to_view(pred_qtos, "0"), **misc.to_view(pred_stoq, "1")}

        if self.conf.add_cycle_error:
            pred["cycle_error0"] = misc.cycle_dist(pred["warp0"], pred["warp1"])
            pred["cycle_error1"] = misc.cycle_dist(pred["warp1"], pred["warp0"])

        if self.conf.sample_num_matches > 0:
            if "keypoints0" in data:
                logger.warning(
                    "'sample_num_matches' is set, therefore keypoints will be ignored. "
                    "Using dense match sampling instead."
                )
            pred.update(self.sample_matches(pred, data, self.conf.sample_num_matches))
        elif "keypoints0" in data:
            pred.update(
                misc.match_keypoints_dense(
                    pred,
                    data,
                    self.conf.max_kp_error,
                    self.conf.filter_threshold,
                    self.conf.mutual_check,
                    sparse_to_dense=self.conf.sparse_to_dense,
                )
            )
        return pred

    def estimate_warp(self, image0: torch.Tensor, image1: torch.Tensor) -> dict:
        """Run UFM on one image pair and return a warp/certainty dict."""
        img0 = F.interpolate(
            image0, size=self._internal_hw, mode="bilinear", align_corners=False
        )
        img1 = F.interpolate(
            image1, size=self._internal_hw, mode="bilinear", align_corners=False
        )

        result = self._matcher.predict_correspondences_batched(
            source_image=img0,
            target_image=img1,
            # Images are float32 in [0, 1] — "identity" = mean 0, std 1.
            data_norm_type="identity",
        )
        flow = result.flow.flow_output  # (B, 2, H, W), pixel offsets
        cert = result.covisibility.mask  # (B, H, W), [0, 1]

        if self.conf.upsample_preds and self.conf.output_hw is not None:
            output_hw = tuple(self.conf.output_hw)
            scale_y = output_hw[0] / self._internal_hw[0]
            scale_x = output_hw[1] / self._internal_hw[1]
            cert = F.interpolate(
                cert[:, None], size=output_hw, mode="bilinear", align_corners=False
            )[:, 0]
            flow = F.interpolate(
                flow, size=output_hw, mode="bilinear", align_corners=False
            )
            flow = flow * flow.new_tensor([scale_x, scale_y])[:, None, None]

        return flow_to_warp(flow, cert)

    def sample_matches(self, pred: dict, data: dict, num_matches: int) -> dict:
        """Sample sparse matches from the predicted dense warps."""
        warp0, warp1 = pred["warp0"], pred["warp1"]
        cert0, cert1 = pred["certainty0"], pred["certainty1"]
        img0, img1 = data["view0"]["image"], data["view1"]["image"]

        assert warp0.shape[0] == 1, "Batch size must be 1 for sampling matches."

        coords0 = misc.get_pixel_grid(fmap=warp0, normalized=True)
        coords1 = misc.get_pixel_grid(fmap=warp1, normalized=True)

        # Each row: (src_x, src_y, tgt_x, tgt_y) in normalized coords.
        matches0 = torch.cat([coords0, warp0], dim=-1).reshape(-1, 4)
        matches1 = torch.cat([warp1, coords1], dim=-1).reshape(-1, 4)
        all_matches = torch.cat([matches0, matches1], dim=0)
        all_scores = torch.cat([cert0.reshape(-1), cert1.reshape(-1)], dim=0)

        # Threshold then take top-k (threshold_balanced approximation).
        keep = all_scores > self.conf.filter_threshold
        if keep.sum() < num_matches:
            keep = torch.ones_like(keep, dtype=torch.bool)
        all_matches = all_matches[keep]
        all_scores = all_scores[keep]

        num_matches = min(num_matches, all_matches.shape[0])
        top_idx = torch.topk(all_scores, num_matches).indices
        m_kpts = all_matches[top_idx]
        scores = all_scores[top_idx].reshape(1, -1)

        return {
            "keypoints0": misc.denormalize_coords(
                m_kpts[:, :2], img0.shape[-2:]
            ).reshape(1, -1, 2),
            "keypoints1": misc.denormalize_coords(
                m_kpts[:, 2:], img1.shape[-2:]
            ).reshape(1, -1, 2),
            "matching_scores0": scores,
            "matching_scores1": scores,
            "keypoint_scores0": scores,
            "keypoint_scores1": scores,
            "matches0": torch.arange(num_matches, device=scores.device)[None],
            "matches1": torch.arange(num_matches, device=scores.device)[None],
        }

    def loss(self, pred, data):
        raise NotImplementedError("Training is currently not supported.")


if __name__ == "__main__":
    """Inference example with UFM matcher."""
    torch.set_grad_enabled(False)
    import warnings

    warnings.filterwarnings("ignore")
    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt

    from ...utils.preprocess import ImagePreprocessor
    from ...visualization import viz2d

    parser = argparse.ArgumentParser()
    parser.add_argument("--image0", type=str, default="assets/boat1.png")
    parser.add_argument("--image1", type=str, default="assets/boat2.png")
    args = parser.parse_args()

    loader = ImagePreprocessor({"resize": 480})
    image0 = loader.load_image(Path(args.image0))["image"]
    image1 = loader.load_image(Path(args.image1))["image"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UFM({"sample_num_matches": 2000}).eval().to(device)
    data = {
        "view0": {"image": image0.to(device)[None]},
        "view1": {"image": image1.to(device)[None]},
    }

    pred = misc.rbd(model(data))
    print("warp0 shape:", pred["warp0"].shape)
    print("certainty0 shape:", pred["certainty0"].shape)
    print(
        "certainty0 range:",
        pred["certainty0"].min().item(),
        "-",
        pred["certainty0"].max().item(),
    )

    if "keypoints0" in pred:
        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        viz2d.plot_images([image0, image1])
        viz2d.plot_matches(kpts0, kpts1, a=0.2)
        plt.savefig("ufm_matches.png")
        print(f"Saved {kpts0.shape[0]} dense matches to ufm_matches.png")

    # SuperPoint keypoint matching via UFM dense field
    from lightglue import SuperPoint

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))

    sp_data = {
        **data,
        "keypoints0": feats0["keypoints"],
        "keypoints1": feats1["keypoints"],
    }
    sp_model = (
        UFM({"sample_num_matches": 0, "sparse_to_dense": False}).eval().to(device)
    )
    sp_pred = misc.rbd(sp_model(sp_data))

    valid = sp_pred["matches0"] > -1
    mkpts0 = sp_pred["keypoints0"][valid]
    mkpts1 = sp_pred["keypoints1"][sp_pred["matches0"][valid]]
    viz2d.plot_images([image0, image1])
    viz2d.plot_matches(mkpts0, mkpts1, a=0.2)
    plt.savefig("ufm_sp_matches.png")
    print(f"Saved {valid.sum().item()} SP+UFM matches to ufm_sp_matches.png")
