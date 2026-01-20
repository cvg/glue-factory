import collections.abc as collections
from pathlib import Path
from typing import Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from ..geometry import homography
from ..geometry import transforms as gtr
from . import misc


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


class ImagePreprocessor:
    default_conf = {
        "resize": None,  # target edge length, None for no resizing
        "edge_divisible_by": None,
        "side": "long",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
        "square_pad": False,
        "add_padding_mask": False,
        "crop_if_short_side": False,
        "crop_mode": "center",
        "pad_value": 0.0,
        "center_pad": False,
        "homography": {
            "p": 0.0,
            "difficulty": 0.7,
            "max_angle": 0.0,
        },  # homography augmentation config
    }

    def __init__(self, conf) -> None:
        super().__init__()
        default_conf = OmegaConf.create(self.default_conf)
        OmegaConf.set_struct(default_conf, True)
        OmegaConf.set_struct(default_conf.homography, False)
        self.conf = OmegaConf.merge(default_conf, conf)

    def __call__(self, img: torch.Tensor, interpolation: Optional[str] = None) -> dict:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        size = h, w
        if self.conf.resize is not None or self.conf.edge_divisible_by is not None:
            if interpolation is None:
                interpolation = self.conf.interpolation
            size = self.get_new_image_size(h, w)
            img = kornia.geometry.transform.resize(
                img,
                size,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
                interpolation=interpolation,
            )
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        r_t_img = np.diag([scale[0], scale[1], 1])

        # Apply random homography (optional)
        if isinstance(self.conf.resize, int):
            target_hw = (self.conf.resize, self.conf.resize)
        else:
            target_hw = self.conf.resize

        img, w_t_r = self.apply_homography(
            img,
            target_hw=target_hw,
            mode=interpolation,
        )
        r_t_img = w_t_r @ r_t_img
        data = {
            "scales": scale,
            "image_size": np.array(img.shape[-2:][::-1]),  # w, h
            "transform": r_t_img,
            "original_image_size": np.array([w, h]),
        }
        if self.conf.square_pad and (
            self.conf.side == "long" or not self.conf.crop_if_short_side
        ):
            padding_data = square_pad(
                img,
                return_mask=self.conf.add_padding_mask,
                fill_value=self.conf.get("pad_value", 0.0),
                center=self.conf.get("center_pad", False),
            )
            data["image"] = padding_data["image"]
            data["corners_hw"] = padding_data["corners_hw"].numpy()
            if "valid" in padding_data:
                data["padding_mask"] = padding_data["valid"]
            data["transform"] = padding_data["transform"] @ data["transform"]
        elif (
            self.conf.side == "short"
            and self.conf.crop_if_short_side
            and self.conf.square_pad
        ):
            crop_data = square_crop(img, mode=self.conf.crop_mode)
            data["image"] = crop_data["image"]
            data["image_size"] = np.array(crop_data["image"].shape[-2:][::-1])  # w, h
            data["corners_hw"] = crop_data["corners_hw"].numpy()
            data["transform"] = crop_data["transform"] @ data["transform"]
        else:
            data["image"] = img
        return data

    def interpolate(
        self,
        img: torch.Tensor,
        norm_t_img: np.ndarray,
        target_hw: Tuple[int, int],
        mode: str = "bilinear",
    ) -> dict:
        """Interpolate an image with a given transform to a target size."""
        if self.conf.resize is None and self.conf.edge_divisible_by is None:
            assert not self.conf.square_pad
            return img
        return kornia.geometry.transform.warp_perspective(
            img[None],
            torch.as_tensor(norm_t_img, device=img.device, dtype=torch.float32)[None],
            dsize=target_hw,
            mode=mode,
            align_corners=False,
        )[0]

    def load_image(self, image_path: Path) -> dict:
        return self(load_image(image_path))

    def get_new_image_size(
        self,
        h: int,
        w: int,
    ) -> Tuple[int, int]:
        side = self.conf.side
        if isinstance(self.conf.resize, collections.Iterable):
            assert len(self.conf.resize) == 2
            return tuple(self.conf.resize)
        elif isinstance(self.conf.resize, int):
            side_size = self.conf.resize
            aspect_ratio = w / h
            if side not in ("short", "long", "vert", "horz"):
                raise ValueError(
                    f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'"  # noqa: E501
                )
            if side == "vert":
                size = side_size, int(side_size * aspect_ratio)
            elif side == "horz":
                size = int(side_size / aspect_ratio), side_size
            elif (side == "short") ^ (aspect_ratio < 1.0):
                size = side_size, int(side_size * aspect_ratio)
            else:
                size = int(side_size / aspect_ratio), side_size
        else:
            assert self.conf.resize is None
            size = (h, w)

        if self.conf.edge_divisible_by is not None:
            df = self.conf.edge_divisible_by
            size = list(map(lambda x: int(x // df * df), size))
        return size

    def apply_homography(
        self, img: torch.Tensor, target_hw: Tuple[int, int], **kwargs
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Sample a random homography matrix and apply it to the image."""
        conf = OmegaConf.to_container(self.conf.get("homography", {}))
        prob = conf.pop("p", 0.0)
        if prob > 0.0:
            warp_H_i, _, _, _ = homography.sample_homography_corners(
                img.shape[-2:][::-1], target_hw, **conf
            )
            warped_img = kornia.geometry.warp_perspective(
                img[None],
                torch.as_tensor(warp_H_i, device=img.device, dtype=torch.float32)[None],
                dsize=target_hw[::-1],
                **kwargs,
            )[0]
            return warped_img, warp_H_i
        else:
            return img, np.eye(3)


def square_pad(
    image: torch.Tensor,
    center: bool = False,
    return_mask: bool = False,
    fill_value: float | None = 0.0,
) -> dict[str, torch.Tensor]:
    """zero pad images to size x size"""
    h, w = image.shape[-2:]
    ox, oy = 0, 0
    hw = max(h, w)
    padded = torch.zeros(
        *image.shape[:-2], hw, hw, device=image.device, dtype=image.dtype
    )
    if center:
        ox, oy = (hw - w) // 2, (hw - h) // 2
        padded[..., oy : oy + h, ox : ox + w] = image
    else:
        ox, oy = 0, 0
        padded[..., :h, :w] = image
        if fill_value is None:
            if h < hw:
                padded[..., h:, :w] = image[..., -1:, :]
            elif w < hw:
                padded[..., :h, w:] = image[..., :, -1:]

    pad_t_img = torch.eye(3, device=image.device)
    pad_t_img[:2, 2] = torch.tensor([ox, oy], device=image.device)
    ret = {
        "image": padded,
        "corners_hw": torch.as_tensor(
            [[oy, ox], [oy + h, ox + w]], dtype=int, device=image.device
        ),
        "transform": pad_t_img,
    }
    if return_mask:
        valid = torch.full_like(padded, dtype=torch.bool, fill_value=0)
        if center:
            valid[..., oy : oy + h, ox : ox + w] = True
        else:
            valid[..., :h, :w] = True
        ret["valid"] = valid
    return ret


def square_crop(
    image: torch.Tensor,
    mode: str = "center",
) -> dict[str, torch.Tensor]:
    """Crop the image to a square shape."""
    h, w = image.shape[-2:]
    hw = min(h, w)
    if mode == "random":
        oy = np.random.randint(0, h - hw + 1)
        ox = np.random.randint(0, w - hw + 1)
        image = image[..., oy : oy + hw, ox : ox + hw]
    elif mode == "center":
        oy = (h - hw) // 2
        ox = (w - hw) // 2
        image = image[..., oy : oy + hw, ox : ox + hw]
    elif mode == "top":
        image = image[..., :hw, :hw]
        ox, oy = 0, 0
    else:
        raise ValueError(f"Unknown crop mode {mode}")
    corners_hw = torch.as_tensor(
        [[oy, ox], [oy + hw, ox + hw]], dtype=int, device=image.device
    )
    crop_t_img = torch.eye(3, device=image.device)
    crop_t_img[:2, 2] = torch.tensor([-ox, -oy], device=image.device)
    ret = {"image": image, "corners_hw": corners_hw, "transform": crop_t_img}
    return ret


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.as_tensor(image / 255.0, dtype=torch.float)


def load_image(path: Path, grayscale=False) -> torch.Tensor:
    image = read_image(path, grayscale=grayscale)
    return numpy_image_to_torch(image)


def resize(image, size, fn=None, interp="linear", df=None):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        w_new, h_new = get_divisible_wh(w_new, h_new, df)
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def crop(image, size, random=True, other=None, K=None, return_bbox=False):
    """Random or deterministic crop of an image, adjust depth and intrinsics."""
    h, w = image.shape[:2]
    h_new, w_new = (size, size) if isinstance(size, int) else size
    top = np.random.randint(0, h - h_new + 1) if random else 0
    left = np.random.randint(0, w - w_new + 1) if random else 0
    image = image[top : top + h_new, left : left + w_new]
    ret = [image]
    if other is not None:
        ret += [other[top : top + h_new, left : left + w_new]]
    if K is not None:
        K[0, 2] -= left
        K[1, 2] -= top
        ret += [K]
    if return_bbox:
        ret += [(top, top + h_new, left, left + w_new)]
    return ret


def zero_pad(size, *images):
    """zero pad images to size x size"""
    ret = []
    for image in images:
        if image is None:
            ret.append(None)
            continue
        h, w = image.shape[:2]
        padded = np.zeros((size, size) + image.shape[2:], dtype=image.dtype)
        padded[:h, :w] = image
        ret.append(padded)
    return ret


class ImageNetNormalizer(nn.Module):
    """Module to normalize images with ImageNet statistics."""

    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.transpose(-3, -1) - self.mean) / self.std).transpose(-3, -1)
