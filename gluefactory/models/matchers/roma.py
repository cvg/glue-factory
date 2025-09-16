"""Wrapper around the RoMA model for image matching.

Paper: RoMa: Robust Dense Feature Matching, CVPR 2024.
Authors: Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten Wadenbäck, Michael Felsberg.
Arxiv: https://arxiv.org/abs/2305.15404
Code: https://github.com/Parskatt/RoMa

License: MIT

Main differences to the original code:
- Warps are by default the same size as the input images.
- Unified API from gluefactory.
- Upsampled resolution for image0 and image1 can differ.
- Compute cycle distance as additional filter.
- Interface to match keypoints via dense correspondences.
"""

import logging

import torch
import torch.nn.functional as F

try:
    from romatch.models.model_zoo.roma_models import roma_model
except ImportError:
    raise ImportError(
        "Please install the 'romatch' package to use RoMA models: "
        "`pip install romatch @ git+https://github.com/Parskatt/RoMa`."
    )

from ...utils import misc
from .. import base_model

logger = logging.getLogger(__name__)


def flow_to_warp(
    query_to_support: torch.Tensor,
    dense_certainty: torch.Tensor,
    lr_certainty: torch.Tensor | None = None,
    extract_query_coords: bool = False,
) -> dict:
    b = dense_certainty.shape[0]
    hs, ws = dense_certainty.shape[-2], dense_certainty.shape[-1]
    device = dense_certainty.device
    if lr_certainty is not None:
        lr_certainty = F.interpolate(
            lr_certainty,
            size=(hs, ws),
            align_corners=False,
            mode="bilinear",
        )
        cert_clamp = 0
        factor = 0.5
        lr_certainty = factor * lr_certainty * (lr_certainty < cert_clamp)
        dense_certainty = dense_certainty - lr_certainty
    # Get certainty interpolation
    query_to_support = query_to_support.permute(0, 2, 3, 1)
    dense_certainty = dense_certainty.sigmoid()  # logits -> probs

    if (query_to_support.abs() > 1).any():
        wrong = (query_to_support.abs() > 1).sum(dim=-1) > 0
        dense_certainty[wrong[:, None]] = 0

    query_to_support = torch.clamp(query_to_support, -1, 1)
    pred = {
        "warp": query_to_support,  # b x h x w x 2
        "certainty": dense_certainty.squeeze(-3),  # b x h x w
    }

    if extract_query_coords:
        # Create im1 meshgrid
        query_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
            )
        )
        query_coords = torch.stack((query_coords[1], query_coords[0]))
        query_coords = query_coords[None].expand(b, 2, hs, ws)
        query_coords = query_coords.permute(0, 2, 3, 1)

        pred["q_coords"] = query_coords

    return pred


class RoMa(base_model.BaseModel):
    """
    RoMa model for matching images.
    """

    default_conf = {
        "weights": "outdoor",
        "upsample_preds": True,
        "symmetric": True,
        "internal_hw": (560, 560),
        "output_hw": None,  # like input image
        "sample": False,
        "mixed_precision": True,  # mixed precision
        "add_cycle_error": False,
        "sample_num_matches": 0,  # sample X sparse matches, <=0 means no sampling
        "sample_mode": "threshold_balanced",
        "filter_threshold": 0.05,  # threshold for filtering matches
        "max_kp_error": 2.0,  # maximum distance for matching keypoints (px)
        "mutual_check": True,  # check mutual NN in keypoint matching
    }
    required_data_keys = ["view0", "view1"]

    weight_urls = {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",  # noqa: E501
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",  # noqa: E501
        "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",  # noqa: E501
    }

    def _init(self, conf):
        weights = torch.hub.load_state_dict_from_url(
            self.weight_urls[self.conf.weights], map_location="cpu"
        )
        dinov2_weights = torch.hub.load_state_dict_from_url(
            self.weight_urls["dinov2_vitl14"], map_location="cpu"
        )

        self._matcher = roma_model(
            resolution=self.conf.internal_hw,
            upsample_preds=self.conf.upsample_preds,
            weights=weights,
            dinov2_weights=dinov2_weights,
            amp_dtype=torch.float16 if self.conf.mixed_precision else torch.float32,
        )
        self._matcher.symmetric = self.conf.symmetric
        self._matcher.sample_thresh = self.conf.filter_threshold

        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

        self.set_initialized(True)

    def _forward(self, data):
        data0, data1 = data["view0"], data["view1"]
        if self._matcher.symmetric:
            pred_qtos, pred_stoq = self.estimate_warp_symmetric(
                data0["image"], data1["image"]
            )
        else:
            pred_qtos = self.estimate_warp(data0["image"], data1["image"])
            pred_stoq = self.estimate_warp(data1["image"], data0["image"])
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
            # Match existing keypoints
            pred.update(
                misc.match_keypoints_dense(
                    pred,
                    data,
                    self.conf.max_kp_error,
                    self.conf.filter_threshold,
                    self.conf.mutual_check,
                )
            )
        return pred

    def process_image(
        self,
        image: torch.Tensor,
        resize: tuple[int, int] | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Process the input image tensor by resizing and normalizing it.
        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W).
            resize (tuple[int, int], optional): Target size for resizing the image.
            normalize (bool, optional): Whether to normalize the image.
        Returns:
            torch.Tensor: Processed image tensor.
        """
        if resize is not None:
            image = F.interpolate(
                image, size=tuple(resize), mode="bilinear", align_corners=False
            )
        if normalize:
            image = (image - self.rgb_mean) / self.rgb_std
        return image

    def upsample_flow_siamese(
        self,
        q: torch.Tensor,
        s: torch.Tensor,
        init_flow: torch.Tensor,
        init_certainty: torch.Tensor,
        stack=False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Upsample flow and certainty to match the input image sizes."""
        f_q_pyramid, f_s_pyramid = self._matcher.extract_backbone_features(
            {"im_A": q, "im_B": s}, upsample=True, batched=False
        )

        dflow_q, dflow_s = init_flow.chunk(2)
        dcert_q, dcert_s = init_certainty.chunk(2)

        def sf(img):
            hr, wr = init_flow.shape[-2:]
            return (img.shape[-2] * img.shape[-1] / (hr * wr)) ** 0.5

        dc_qtos = self._matcher.decoder(
            f_q_pyramid,
            f_s_pyramid,
            upsample=True,
            flow=dflow_q,
            certainty=dcert_q,
            scale_factor=sf(q),
        )

        dc_stoq = self._matcher.decoder(
            f_s_pyramid,
            f_q_pyramid,
            upsample=True,
            flow=dflow_s,
            certainty=dcert_s,
            scale_factor=sf(s),
        )

        if stack:
            out = {
                scale: {
                    k: torch.cat(
                        [
                            dc_qtos[scale][k],
                            dc_stoq[scale][k],
                        ]
                    )
                    for k in ["flow", "certainty"]
                }
                for scale in dc_qtos.keys()
            }
            return out
        else:
            return dc_qtos, dc_stoq

    def estimate_warp_symmetric(
        self, image0: torch.Tensor, image1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = image0.device
        internal_hw = self.conf.internal_hw
        query = self.process_image(image0, resize=internal_hw, normalize=True)
        support = self.process_image(image1, resize=internal_hw, normalize=True)
        batch = {"im_A": query, "im_B": support}

        finest_scale = 1
        # Run matcher
        dense_corresps = self._matcher.forward_symmetric(batch, batched=True)
        lr_certainty0, lr_certainty1 = dense_corresps[16]["certainty"].chunk(2)

        if self._matcher.upsample_preds:
            output_hw = self.conf.output_hw
            query = self.process_image(image0, resize=output_hw, normalize=True)
            support = self.process_image(image1, resize=output_hw, normalize=True)
            query, support = query.to(device), support.to(device)

            dc_qtos, dc_stoq = self.upsample_flow_siamese(
                query,
                support,
                dense_corresps[finest_scale]["flow"],
                dense_corresps[finest_scale]["certainty"],
                stack=False,
            )
        else:
            raise NotImplementedError

        pred_qtos = flow_to_warp(
            dc_qtos[finest_scale]["flow"],
            dc_qtos[finest_scale]["certainty"],
            lr_certainty0,
        )
        pred_stoq = flow_to_warp(
            dc_stoq[finest_scale]["flow"],
            dc_stoq[finest_scale]["certainty"],
            lr_certainty1,
        )

        return pred_qtos, pred_stoq

    def estimate_warp(
        self, query: torch.Tensor, support: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = query.device
        internal_hw = self.conf.internal_hw
        query = self.process_image(query, resize=internal_hw, normalize=True)
        support = self.process_image(support, resize=internal_hw, normalize=True)
        batch = {"im_A": query, "im_B": support}

        finest_scale = 1
        # Run matcher
        corresps = self._matcher.forward(batch, batched=True)
        low_res_certainty = corresps[16]["certainty"]

        if self._matcher.upsample_preds:
            size = self.conf.output_hw
            query = self.process_image(query, resize=size, normalize=True)
            support = self.process_image(support, resize=size, normalize=True)
            f_q_pyramid, f_s_pyramid = self._matcher.extract_backbone_features(
                {"im_A": query.to(device), "im_B": support.to(device)},
                upsample=True,
                batched=False,
            )

            corresps = self._matcher.decoder(
                f_q_pyramid,
                f_s_pyramid,
                upsample=True,
                flow=corresps[finest_scale]["flow"],
                certainty=corresps[finest_scale]["certainty"],
            )
        query_to_support = corresps[finest_scale]["flow"]
        certainty = corresps[finest_scale]["certainty"]

        return flow_to_warp(query_to_support, certainty, low_res_certainty)

    def sample_matches(self, pred: dict, data: dict, num_matches: int) -> dict:
        """Sample sparse matches from the predicted warps."""
        warp0, warp1 = pred["warp0"], pred["warp1"]
        img0, img1 = data["view0"]["image"], data["view1"]["image"]

        assert warp0.shape[0] == 1, "Batch size must be 1 for sampling matches."
        certainty0, certainty1 = pred["certainty0"], pred["certainty1"]
        coords0 = misc.get_pixel_grid(fmap=warp0, normalized=True)
        coords1 = misc.get_pixel_grid(fmap=warp1, normalized=True)

        matches0 = torch.cat([coords0, warp0], dim=-1)
        matches1 = torch.cat([warp1, coords1], dim=-1)

        matches = torch.cat([matches0.reshape(-1, 4), matches1.reshape(-1, 4)], dim=0)
        scores = torch.cat([certainty0.reshape(-1), certainty1.reshape(-1)], dim=0)

        m_kpts, scores = self._matcher.sample(
            matches,
            scores,
            num=num_matches,
        )  # N x 4, N

        scores = scores.reshape(1, -1)
        sparse_pred = {
            # In COLMAP coordinates, i.e. [0, 0] is the corner of the top-left pixel
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
            "matches0": torch.arange(0, scores.shape[-1]).to(scores.device)[None],
            "matches1": torch.arange(0, scores.shape[-1]).to(scores.device)[None],
        }

        return sparse_pred

    def loss(self, pred, data):
        raise NotImplementedError("Training is currently not supported.")


if __name__ == "__main__":
    """Inference example with RoMA matcher."""
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

    image_loader = ImagePreprocessor({"resize": 480})

    image_path0 = Path(args.image0)
    image_path1 = Path(args.image1)

    data0, data1 = image_loader.load_image(image_path0), image_loader.load_image(
        image_path1
    )
    image0, image1 = data0["image"], data1["image"]

    device = "cuda"
    dkm_model = RoMa({"symmetric": True, "sample_num_matches": 0}).eval().to(device)
    data = {
        "view0": {"image": image0.to(device)[None]},
        "view1": {"image": image1.to(device)[None]},
    }

    # Example to match SuperPoint keypoints with RoMa (use LG interface)
    from lightglue import SuperPoint

    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))

    data.update({k + "0": v for k, v in feats0.items()})
    data.update({k + "1": v for k, v in feats1.items()})

    pred = misc.rbd(dkm_model(data))
    certainty0 = pred["certainty0"]
    certainty1 = pred["certainty1"]

    q_coords0 = misc.get_pixel_grid(fmap=pred["warp0"], normalized=True)
    q_coords1 = misc.get_pixel_grid(fmap=pred["warp1"], normalized=True)

    image_0to0 = misc.grid_sample(image0, q_coords0)
    image_1to1 = misc.grid_sample(image1, q_coords1)
    image_1to0 = misc.grid_sample(image1, pred["warp0"])
    image_0to1 = misc.grid_sample(image0, pred["warp1"])

    white0, white1 = torch.ones_like(certainty0).to(device), torch.ones_like(
        certainty1
    ).to(device)
    visible_0to0 = certainty0 * image_0to0 + (1 - certainty0) * white0
    visible_1to1 = certainty1 * image_1to1 + (1 - certainty1) * white1
    viz2d.plot_images([visible_0to0, visible_1to1])

    plt.savefig("roma_covisible.png")
    visible_1to0 = certainty0 * image_1to0 + (1 - certainty0) * white0
    visible_0to1 = certainty1 * image_0to1 + (1 - certainty1) * white1
    viz2d.plot_images(
        [visible_1to0, visible_0to1],
    )

    plt.savefig("roma_warp.png")

    if "keypoints0" in pred:
        kpts0 = pred["keypoints0"]
        kpts1 = pred["keypoints1"]

        viz2d.plot_images(
            [image0, image1],
        )

        valid = pred["matches0"] > -1
        kpts1 = kpts1[pred["matches0"]]
        kpts0, kpts1 = kpts0[valid], kpts1[valid]
        viz2d.plot_matches(kpts0, kpts1, a=0.2)

        plt.savefig("roma_matches.png")
