"""Wrapper around the RoMA model for image matching.

Paper: RoMa: Robust Dense Feature Matching, CVPR 2024.
Authors: Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten Wadenbäck, Michael Felsberg.
Arxiv: https://arxiv.org/abs/2305.15404
Code: https://github.com/Parskatt/RoMa

License: MIT

Main differences to the original code:
- Warps are by default the same size as the input images.
- Unified API from gluefactory.
"""

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


class RoMA(base_model.BaseModel):
    """
    RoMA model for matching images.
    """

    default_conf = {
        "weights": "outdoor",
        "upsample_preds": True,
        "symmetric": True,
        "internal_shape": (560, 560),
        "output_shape": None,  # like input image
        "sample": False,
        "mixed_precision": True,  # mixed precision
    }

    required_data_keys = ["view0", "view1"]

    weight_urls = {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",  # noqa: E501
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",  # noqa: E501
        "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",  # noqa: E501
    }

    def _init(self, conf):
        weights = torch.hub.load_state_dict_from_url(
            self.weight_urls[self.conf.weights], map_location=device
        )
        dinov2_weights = torch.hub.load_state_dict_from_url(
            self.weight_urls["dinov2_vitl14"], map_location=device
        )

        self._matcher = roma_model(
            resolution=self.conf.internal_shape,
            upsample_preds=self.conf.upsample_preds,
            weights=weights,
            dinov2_weights=dinov2_weights,
            device=device,
            amp_dtype=torch.float16 if self.conf.mixed_precision else torch.float32,
        )
        self._matcher.symmetric = self.conf.symmetric

        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

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
                image, size=resize, mode="bilinear", align_corners=False
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

    def _forward(self, data):
        data0, data1 = data["view0"], data["view1"]
        if self._matcher.symmetric:
            pred_qtos, pred_stoq = self.estimate_warp_symmetric(
                data0["image"], data1["image"]
            )
        else:
            pred_qtos = self.estimate_warp(data0["image"], data1["image"])
            pred_stoq = self.estimate_warp(data1["image"], data0["image"])
        data = {**misc.to_view(pred_qtos, "0"), **misc.to_view(pred_stoq, "1")}
        return data

    def estimate_warp_symmetric(
        self, image0: torch.Tensor, image1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = image0.device
        internal_shape = tuple(self.conf.internal_shape)
        query = self.process_image(image0, resize=internal_shape, normalize=True)
        support = self.process_image(image1, resize=internal_shape, normalize=True)
        batch = {"im_A": query, "im_B": support}

        finest_scale = 1
        # Run matcher
        dense_corresps = self._matcher.forward_symmetric(batch, batched=True)
        lr_certainty0, lr_certainty1 = dense_corresps[16]["certainty"].chunk(2)

        if self._matcher.upsample_preds:
            size = self.conf.output_shape
            query = self.process_image(image0, resize=size, normalize=True)
            support = self.process_image(image1, resize=size, normalize=True)
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
        internal_shape = tuple(self.conf.internal_shape)
        query = self.process_image(query, resize=internal_shape, normalize=True)
        support = self.process_image(support, resize=internal_shape, normalize=True)
        batch = {"im_A": query, "im_B": support}

        finest_scale = 1
        # Run matcher
        corresps = self._matcher.forward(batch, batched=True)
        low_res_certainty = corresps[16]["certainty"]

        if self._matcher.upsample_preds:
            size = self.conf.output_shape
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

    def loss(self, pred, data):
        raise NotImplementedError("Training is currently not supported.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt

    from ...utils.image import ImagePreprocessor, get_pixel_grid, sample_rgb
    from ...utils.tensor import rbd
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
    dkm_model = RoMA({"symmetric": True}).eval().to(device)
    data = {
        "view0": {"image": image0.to(device)[None]},
        "view1": {"image": image1.to(device)[None]},
    }
    pred = rbd(dkm_model(data))
    certainty0 = pred["certainty0"]
    certainty1 = pred["certainty1"]

    q_coords0 = get_pixel_grid(fmap=pred["warp0"], normalized=True)
    q_coords1 = get_pixel_grid(fmap=pred["warp1"], normalized=True)

    image_0to0 = sample_rgb(image0, q_coords0)
    image_1to1 = sample_rgb(image1, q_coords1)
    image_1to0 = sample_rgb(image1, pred["warp0"])
    image_0to1 = sample_rgb(image0, pred["warp1"])

    white0, white1 = torch.ones_like(certainty0).to(device), torch.ones_like(
        certainty1
    ).to(device)
    visible_0to0 = certainty0 * image_0to0 + (1 - certainty0) * white0
    visible_1to1 = certainty1 * image_1to1 + (1 - certainty1) * white1
    viz2d.plot_images(
        [visible_0to0, visible_1to1], titles=["visible area in image", None]
    )

    plt.savefig("warp_init.png")
    visible_1to0 = certainty0 * image_1to0 + (1 - certainty0) * white0
    visible_0to1 = certainty1 * image_0to1 + (1 - certainty1) * white1
    viz2d.plot_images(
        [visible_1to0, visible_0to1],
        titles=["projected onto other image", None],
    )

    plt.savefig("warp.png")
