"""Wrapper around the DaD model for keypoint detection.

Paper: DaD: Distilled Reinforcement Learning for Diverse Keypoint Detection
Authors: Johan Edstedt, Georg Bökman, Mårten Wadenbäck, Michael Felsberg.
Arxiv: https://arxiv.org/abs/2503.07347
Code: https://github.com/Parskatt/dad

License: MIT

Main differences to the original code:
- Unified API from gluefactory.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as transforms
from PIL import Image

from gluefactory.models.base_model import BaseModel


def check_not_i16(im):
    if im.mode == "I;16":
        raise NotImplementedError("Can't handle 16 bit images")


def get_best_device(verbose=False):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if verbose:
        print(f"Fastest device found is: {device}")
    return device


def get_grid(B, H, W, device=get_best_device()):
    x1_n = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
    return x1_n


def extract_patches_from_inds(x: torch.Tensor, inds: torch.Tensor, patch_size: int):
    B, H, W = x.shape
    B, N = inds.shape
    unfolder = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2, stride=1)
    unfolded_x: torch.Tensor = unfolder(x[:, None])  # B x K_H * K_W x H * W
    patches = torch.gather(
        unfolded_x,
        dim=2,
        index=inds[:, None, :].expand(B, patch_size**2, N),
    )  # B x K_H * K_W x N
    return patches


def sample_keypoints(
    keypoint_probs: torch.Tensor,
    num_samples=8192,
    device=get_best_device(),
    use_nms=True,
    nms_size=1,
    sample_topk=True,
    increase_coverage=True,
    remove_borders=False,
    return_probs=False,
    coverage_pow=1 / 2,
    coverage_size=51,
    subpixel=False,
    scoremap=None,  # required for subpixel
    subpixel_temp=0.5,
):
    B, H, W = keypoint_probs.shape
    if increase_coverage:
        weights = (
            -(torch.linspace(-2, 2, steps=coverage_size, device=device) ** 2)
        ).exp()[None, None]
        # 10000 is just some number for maybe numerical stability, who knows. :), result is invariant anyway
        local_density_x = F.conv2d(
            (keypoint_probs[:, None] + 1e-6) * 10000,
            weights[..., None, :],
            padding=(0, coverage_size // 2),
        )
        local_density = F.conv2d(
            local_density_x, weights[..., None], padding=(coverage_size // 2, 0)
        )[:, 0]
        keypoint_probs = keypoint_probs * (local_density + 1e-8) ** (-coverage_pow)
    grid = get_grid(B, H, W, device=device).reshape(B, H * W, 2)
    if use_nms:
        keypoint_probs = keypoint_probs * (
            keypoint_probs
            == F.max_pool2d(keypoint_probs, nms_size, stride=1, padding=nms_size // 2)
        )
    if remove_borders:
        frame = torch.zeros_like(keypoint_probs)
        # we hardcode 4px, could do it nicer, but whatever
        frame[..., 4:-4, 4:-4] = 1
        keypoint_probs = keypoint_probs * frame
    if sample_topk:
        inds = torch.topk(keypoint_probs.reshape(B, H * W), k=num_samples).indices
    else:
        inds = torch.multinomial(
            keypoint_probs.reshape(B, H * W), num_samples=num_samples, replacement=False
        )
    kps = torch.gather(grid, dim=1, index=inds[..., None].expand(B, num_samples, 2))
    if subpixel:
        print("Subpixel refinement is enabled, this will be slow and memory intensive.")
        offsets = get_grid(B, nms_size, nms_size).reshape(
            B, nms_size**2, 2
        )  # B x K_H x K_W x 2
        offsets[..., 0] = offsets[..., 0] * nms_size / W
        offsets[..., 1] = offsets[..., 1] * nms_size / H
        keypoint_patch_scores = extract_patches_from_inds(scoremap, inds, nms_size)
        keypoint_patch_probs = (keypoint_patch_scores / subpixel_temp).softmax(
            dim=1
        )  # B x K_H * K_W x N
        keypoint_offsets = torch.einsum("bkn, bkd ->bnd", keypoint_patch_probs, offsets)
        kps = kps + keypoint_offsets
    if return_probs:
        return kps, torch.gather(keypoint_probs.reshape(B, H * W), dim=1, index=inds)
    return kps


class Detector(ABC, nn.Module):
    @property
    @abstractmethod
    def topleft(self) -> float:
        pass

    @abstractmethod
    def load_image(im_path: Union[str, Path]) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def detect(
        self, batch: dict[str, torch.Tensor], *, num_keypoints, return_dense_probs=False
    ) -> dict[str, torch.Tensor]:
        pass

    @torch.inference_mode
    def detect_from_path(
        self,
        im_path: Union[str, Path],
        *,
        num_keypoints: int,
        return_dense_probs: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self.detect(
            self.load_image(im_path),
            num_keypoints=num_keypoints,
            return_dense_probs=return_dense_probs,
        )

    def to_pixel_coords(
        self, normalized_coords: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        if normalized_coords.shape[-1] != 2:
            raise ValueError(
                f"Expected shape (..., 2), but got {normalized_coords.shape}"
            )
        pixel_coords = torch.stack(
            (
                w * (normalized_coords[..., 0] + 1) / 2,
                h * (normalized_coords[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
        return pixel_coords

    def to_normalized_coords(
        self, pixel_coords: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        if pixel_coords.shape[-1] != 2:
            raise ValueError(f"Expected shape (..., 2), but got {pixel_coords.shape}")
        normalized_coords = torch.stack(
            (
                2 * (pixel_coords[..., 0]) / w - 1,
                2 * (pixel_coords[..., 1]) / h - 1,
            ),
            axis=-1,
        )
        return normalized_coords


class DeDoDeDetector(Detector):
    def __init__(
        self,
        *args,
        encoder: nn.Module,
        decoder: nn.Module,
        resize: int,
        nms_size: int,
        subpixel: bool,
        subpixel_temp: float,
        keep_aspect_ratio: bool,
        remove_borders: bool,
        increase_coverage: bool,
        coverage_pow: float,
        coverage_size: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.encoder = encoder
        self.decoder = decoder
        self.remove_borders = remove_borders
        self.resize = resize
        self.increase_coverage = increase_coverage
        self.coverage_pow = coverage_pow
        self.coverage_size = coverage_size
        self.nms_size = nms_size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.subpixel = subpixel
        self.subpixel_temp = subpixel_temp

    @property
    def topleft(self):
        return 0.5

    def forward_impl(
        self,
        images,
    ):
        features, sizes = self.encoder(images)
        logits = 0
        context = None
        scales = ["8", "4", "2", "1"]
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_logits, context = self.decoder(
                feature_map, context=context, scale=scale
            )
            logits = (
                logits + delta_logits.float()
            )  # ensure float (need bf16 doesnt have f.interpolate)
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                logits = F.interpolate(
                    logits, size=size, mode="bicubic", align_corners=False
                )
                context = F.interpolate(
                    context.float(), size=size, mode="bilinear", align_corners=False
                )
        return logits.float()

    def forward(self, batch) -> dict[str, torch.Tensor]:
        # wraps internal forward impl to handle
        # different types of batches etc.
        if "im_A" in batch:
            images = torch.cat((batch["im_A"], batch["im_B"]))
        else:
            images = batch["image"]
        scoremap = self.forward_impl(images)
        return {"scoremap": scoremap}

    @torch.inference_mode()
    def detect(
        self, batch, *, num_keypoints, return_dense_probs=False
    ) -> dict[str, torch.Tensor]:
        self.train(False)
        scoremap = self.forward(batch)["scoremap"]
        B, K, H, W = scoremap.shape
        dense_probs = (
            scoremap.reshape(B, K * H * W)
            .softmax(dim=-1)
            .reshape(B, K, H * W)
            .sum(dim=1)
        )
        dense_probs = dense_probs.reshape(B, H, W)
        keypoints, confidence = sample_keypoints(
            dense_probs,
            use_nms=True,
            nms_size=self.nms_size,
            sample_topk=True,
            num_samples=num_keypoints,
            return_probs=True,
            increase_coverage=self.increase_coverage,
            remove_borders=self.remove_borders,
            coverage_pow=self.coverage_pow,
            coverage_size=self.coverage_size,
            subpixel=self.subpixel,
            subpixel_temp=self.subpixel_temp,
            scoremap=scoremap.reshape(B, H, W),
        )
        result = {"keypoints": keypoints, "keypoint_probs": confidence}
        if return_dense_probs:
            result["dense_probs"] = dense_probs
            result["scoremap"] = scoremap
        return result

    def load_image(self, im_path, device=get_best_device()) -> dict[str, torch.Tensor]:
        pil_im = Image.open(im_path)
        check_not_i16(pil_im)
        pil_im = pil_im.convert("RGB")
        if self.keep_aspect_ratio:
            W, H = pil_im.size
            scale = self.resize / max(W, H)
            W = int((scale * W) // 8 * 8)
            H = int((scale * H) // 8 * 8)
        else:
            H, W = self.resize, self.resize
        pil_im = pil_im.resize((W, H))
        standard_im = np.array(pil_im) / 255.0
        return {
            "image": self.normalizer(torch.from_numpy(standard_im).permute(2, 0, 1))
            .float()
            .to(device)[None]
        }


class Decoder(nn.Module):
    def __init__(
        self, layers, *args, super_resolution=False, num_prototypes=1, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.scales = self.layers.keys()
        self.super_resolution = super_resolution
        self.num_prototypes = num_prototypes

    def forward(self, features, context=None, scale=None):
        if context is not None:
            features = torch.cat((features, context), dim=1)
        stuff = self.layers[scale](features)
        logits, context = (
            stuff[:, : self.num_prototypes],
            stuff[:, self.num_prototypes :],
        )
        return logits, context


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=True,
        kernel_size=5,
        hidden_blocks=5,
        amp=True,
        residual=False,
        amp_dtype=torch.float16,
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim,
            hidden_dim,
            dw=False,
            kernel_size=1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.residual = residual

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=True,
        kernel_size=5,
        bias=True,
        norm_type=nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = (
            norm_type(out_dim)
            if norm_type is nn.BatchNorm2d
            else norm_type(num_channels=out_dim)
        )
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, feats):
        b, c, hs, ws = feats.shape
        with torch.autocast(
            device_type=feats.device.type, enabled=self.amp, dtype=self.amp_dtype
        ):
            x0 = self.block1(feats)
            x = self.hidden_blocks(x0)
            if self.residual:
                x = (x + x0) / 1.4
            x = self.out_conv(x)
            return x


class VGG19(nn.Module):
    def __init__(self, amp=False, amp_dtype=torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn().features[:40])
        # Maxpool layers: 6, 13, 26, 39
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast(
            device_type=x.device.type, enabled=self.amp, dtype=self.amp_dtype
        ):
            feats = []
            sizes = []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes


class VGG(nn.Module):
    def __init__(self, size="19", amp=False, amp_dtype=torch.float16) -> None:
        super().__init__()
        if size == "11":
            self.layers = nn.ModuleList(tvm.vgg11_bn().features[:22])
        elif size == "13":
            self.layers = nn.ModuleList(tvm.vgg13_bn().features[:28])
        elif size == "19":
            self.layers = nn.ModuleList(tvm.vgg19_bn().features[:40])
        # Maxpool layers: 6, 13, 26, 39
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast(
            device_type=x.device.type, enabled=self.amp, dtype=self.amp_dtype
        ):
            feats = []
            sizes = []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes


def dedode_detector_S():
    residual = True
    hidden_blocks = 3
    amp_dtype = torch.float16
    amp = True
    NUM_PROTOTYPES = 1
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG(size="11", amp=amp, amp_dtype=amp_dtype)
    decoder = Decoder(conv_refiner)
    return encoder, decoder


def dedode_detector_B():
    residual = True
    hidden_blocks = 5
    amp_dtype = torch.float16
    amp = True
    NUM_PROTOTYPES = 1
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG19(amp=amp, amp_dtype=amp_dtype)
    decoder = Decoder(conv_refiner)
    return encoder, decoder


def dedode_detector_L():
    NUM_PROTOTYPES = 1
    residual = True
    hidden_blocks = 8
    amp_dtype = (
        torch.float16
    )  # torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                128,
                64 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 64,
                64,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG19(amp=amp, amp_dtype=amp_dtype)
    decoder = Decoder(conv_refiner)
    return encoder, decoder


class DaD(DeDoDeDetector):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *args,
        resize=1024,
        nms_size=3,
        remove_borders=False,
        increase_coverage=False,
        coverage_pow=None,
        coverage_size=None,
        subpixel=False,  # False for our evals
        subpixel_temp=0.5,
        keep_aspect_ratio=True,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            encoder=encoder,
            decoder=decoder,
            resize=resize,
            nms_size=nms_size,
            remove_borders=remove_borders,
            increase_coverage=increase_coverage,
            coverage_pow=coverage_pow,
            coverage_size=coverage_size,
            subpixel=subpixel,
            keep_aspect_ratio=keep_aspect_ratio,
            subpixel_temp=subpixel_temp,
            **kwargs,
        )


class DeDoDev2(DeDoDeDetector):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *args,
        resize=784,
        nms_size=3,
        remove_borders=False,
        increase_coverage=True,
        coverage_pow=0.5,
        coverage_size=51,
        subpixel=False,
        subpixel_temp=None,
        keep_aspect_ratio=False,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            encoder=encoder,
            decoder=decoder,
            resize=resize,
            nms_size=nms_size,
            remove_borders=remove_borders,
            increase_coverage=increase_coverage,
            coverage_pow=coverage_pow,
            coverage_size=coverage_size,
            subpixel=subpixel,
            keep_aspect_ratio=keep_aspect_ratio,
            subpixel_temp=subpixel_temp,
            **kwargs,
        )


def load_DaD(
    resize=1024,
    nms_size=3,
    remove_borders=True,
    increase_coverage=False,
    coverage_pow=None,
    coverage_size=None,
    subpixel=True,
    subpixel_temp=0.5,
    keep_aspect_ratio=True,
    pretrained=True,
    weights_path=None,
) -> DaD:
    if weights_path is None:
        weights_path = (
            "https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth"
        )
    device = get_best_device()
    encoder, decoder = dedode_detector_S()
    model = DaD(
        encoder,
        decoder,
        resize=resize,
        nms_size=nms_size,
        remove_borders=remove_borders,
        increase_coverage=increase_coverage,
        coverage_pow=coverage_pow,
        coverage_size=coverage_size,
        subpixel=subpixel,
        subpixel_temp=subpixel_temp,
        keep_aspect_ratio=keep_aspect_ratio,
    ).to(device)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url(
            weights_path, weights_only=False, map_location=device
        )
        model.load_state_dict(weights)
    return model


def load_DaDLight(
    resize=1024,
    nms_size=3,
    remove_borders=True,
    increase_coverage=False,
    coverage_pow=None,
    coverage_size=None,
    subpixel=True,
    subpixel_temp=0.5,
    keep_aspect_ratio=True,
    pretrained=True,
    weights_path=None,
) -> DaD:
    if weights_path is None:
        weights_path = (
            "https://github.com/Parskatt/dad/releases/download/v0.1.0/dad_light.pth"
        )
    return load_DaD(
        resize=resize,
        nms_size=nms_size,
        remove_borders=remove_borders,
        increase_coverage=increase_coverage,
        coverage_pow=coverage_pow,
        coverage_size=coverage_size,
        subpixel=subpixel,
        subpixel_temp=subpixel_temp,
        keep_aspect_ratio=keep_aspect_ratio,
        pretrained=pretrained,
        weights_path=weights_path,
    )


def load_DaDDark(
    resize=1024,
    nms_size=3,
    remove_borders=True,
    increase_coverage=False,
    coverage_pow=None,
    coverage_size=None,
    subpixel=True,
    subpixel_temp=0.5,
    keep_aspect_ratio=True,
    pretrained=True,
    weights_path=None,
) -> DaD:
    if weights_path is None:
        weights_path = (
            "https://github.com/Parskatt/dad/releases/download/v0.1.0/dad_dark.pth"
        )
    return load_DaD(
        resize=resize,
        nms_size=nms_size,
        remove_borders=remove_borders,
        increase_coverage=increase_coverage,
        coverage_pow=coverage_pow,
        coverage_size=coverage_size,
        subpixel=subpixel,
        subpixel_temp=subpixel_temp,
        keep_aspect_ratio=keep_aspect_ratio,
        pretrained=pretrained,
        weights_path=weights_path,
    )


def load_dedode_v2() -> DeDoDev2:
    device = get_best_device()
    weights = torch.hub.load_state_dict_from_url(
        "https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth",
        map_location=device,
    )

    encoder, decoder = dedode_detector_L()
    model = DeDoDev2(encoder, decoder).to(device)
    model.load_state_dict(weights)
    return model


class DaD_Extractor(BaseModel):
    default_conf = {
        "name": "dad",
        "resize": 1024,
        "nms_size": 3,
        "remove_borders": True,
        "subpixel": True,
        "subpixel_temp": 0.5,
        "keep_aspect_ratio": False,
        "max_num_keypoints": 512,
    }

    def _init(self, conf):
        self.model = load_DaD(
            resize=conf.resize,
            nms_size=conf.nms_size,
            remove_borders=conf.remove_borders,
            subpixel=conf.subpixel,
            subpixel_temp=conf.subpixel_temp,
            keep_aspect_ratio=conf.keep_aspect_ratio,
        )
        self.set_initialized(to=True)
        self.conf = conf

    def _forward(self, data):
        input_data = deepcopy(data)
        if input_data["image"].shape[1] == 1:
            input_data["image"] = input_data["image"].repeat(1, 3, 1, 1)
        # ImageNet normalization
        input_data["image"] = self.model.normalizer(input_data["image"])
        detections = self.model.detect(
            input_data,
            num_keypoints=self.conf.max_num_keypoints,
            return_dense_probs=True,
        )
        keypoints = self.model.to_pixel_coords(
            detections["keypoints"],
            input_data["image"].shape[2],
            input_data["image"].shape[3],
        )
        out = {
            "keypoints": keypoints,
            "keypoint_scores": detections["keypoint_probs"],
            "dense_probs": detections["dense_probs"].unsqueeze(1),
        }
        return out

    def loss(self, pred, data):
        raise NotImplementedError("DaD does not support loss calculation")
