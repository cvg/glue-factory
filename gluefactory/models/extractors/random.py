"""Random (non-grid) keypoint extractor.

Samples keypoints at genuinely random continuous positions, optionally biased
toward a region (depth mask, covisible mask, or valid image content), as a
sibling to grid_extractor.py's regular-grid sampler.
"""

import torch

from ...utils.misc import (
    content_bounds,
    erode_mask,
    sample_random_keypoints,
    sample_valid_keypoints,
)
from ..base_model import BaseModel


class RandomExtractor(BaseModel):
    default_conf = {
        "bias_to": None,  # "depth" | "covisible" | "image" | None (unbiased)
        "avoid_borders": None,  # None (disabled) or an int: mask erosion radius in pixels
        "max_num_keypoints": None,  # required — no fallback for a pure generator
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        assert (
            conf.max_num_keypoints is not None
        ), "RandomExtractor requires max_num_keypoints to be set."
        assert (
            conf.avoid_borders is None or conf.avoid_borders >= 0
        ), f"avoid_borders must be None or a non-negative int, got {conf.avoid_borders!r}."

    def _forward(self, data):
        b, c, h, w = data["image"].shape
        device, dtype = data["image"].device, data["image"].dtype
        n = self.conf.max_num_keypoints
        bias_to = self.conf.bias_to
        margin = self.conf.avoid_borders or 0

        if bias_to == "image":
            assert "transform" in data, "bias_to='image' requires data['transform']"
            xy_min, xy_max = content_bounds(
                data["transform"], data["original_image_size"], device, dtype
            )
            if margin:
                xy_min, xy_max = xy_min + margin, xy_max - margin
            assert torch.all(xy_max > xy_min), (
                f"RandomExtractor: avoid_borders margin {margin}px leaves an "
                "empty/inverted valid region for at least one batch item."
            )
            bbox = torch.cat([xy_min, xy_max], dim=-1)
            keypoints = sample_random_keypoints(n, None, None, device, bbox=bbox)
        else:
            if bias_to == "depth":
                assert "depth" in data, "bias_to='depth' requires data['depth']"
                mask = data["depth"] > 0
            elif bias_to == "covisible":
                assert "covisible_mask" in data, (
                    "bias_to='covisible' requires data['covisible_mask'], "
                    "populated by TwoViewPipeline for 2-view batches when this "
                    "extractor's own config declares bias_to='covisible'."
                )
                mask = data["covisible_mask"]
                stride = data.get("covisible_mask_stride", 4)
                if stride > 1:
                    # Upsample (nearest) to full resolution: sample_valid_keypoints
                    # already jitters each cell within its stride x stride block,
                    # but its distinct-position selection (topk) operates on
                    # whatever grid it's given — upsampling first lets it pick
                    # among all stride^2 real sub-pixel positions per cell instead
                    # of being capped at one distinct pick per coarse cell.
                    mask = mask.repeat_interleave(stride, dim=-2).repeat_interleave(
                        stride, dim=-1
                    )
            elif bias_to is None:
                mask = torch.ones(b, h, w, dtype=torch.bool, device=device)
            else:
                raise ValueError(f"Unknown bias_to: {bias_to}")

            if margin:
                mask = erode_mask(mask, margin)
            keypoints = sample_valid_keypoints(mask, n, dtype=dtype)

        return {"keypoints": keypoints}

    def loss(self, pred, data):
        raise NotImplementedError


if __name__ == "__main__":
    """Visualize RandomExtractor keypoints on a sample image."""
    import argparse

    import matplotlib.pyplot as plt

    from ...utils import preprocess
    from ...visualization import viz2d
    from .grid_extractor import GridExtractor

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="assets/sacre_coeur1.jpg")
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--avoid_borders", type=int, default=10)
    args = parser.parse_args()

    image_loader = preprocess.ImagePreprocessor({"resize": 512})
    data = image_loader.load_image(args.image)
    data = {k: v[None] if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    image = data["image"]
    _, _, h, w = image.shape

    # Fake a "depth" mask (a circle) so bias_to="depth" has something to work
    # with even though this demo image has no real depth.
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cy, cx, r = h * 0.55, w * 0.35, min(h, w) * 0.3
    circle_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r**2
    data["depth"] = circle_mask.float()[None]

    random_unbiased = RandomExtractor({"max_num_keypoints": args.max_num_keypoints})
    random_depth = RandomExtractor(
        {
            "max_num_keypoints": args.max_num_keypoints // 2,
            "bias_to": "depth",
            "avoid_borders": args.avoid_borders,
        }
    )
    grid = GridExtractor({"cell_size": 24, "bias_to": None})

    pred_unbiased = random_unbiased(data)
    pred_depth = random_depth(data)
    pred_grid = grid(data)

    viz2d.plot_images(
        [image[0], image[0], image[0]],
        titles=[
            "RandomExtractor (unbiased)",
            "RandomExtractor (bias_to=depth)",
            "GridExtractor",
        ],
    )
    axes = plt.gcf().axes
    axes[1].imshow(circle_mask.numpy(), alpha=0.3, cmap="Reds")
    viz2d.plot_keypoints(
        [
            pred_unbiased["keypoints"][0],
            pred_depth["keypoints"][0],
            pred_grid["keypoints"][0],
        ],
        colors="lime",
        ps=6,
    )
    viz2d.save_plot("random_extractor_keypoints.png")
    print("Saved to random_extractor_keypoints.png")
    plt.show()
