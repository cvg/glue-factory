"""
Convenience classes for an SE3 pose and a pinhole Camera with lens distortion.
Based on PyTorch tensors: differentiable, batched, with GPU support.
"""

import dataclasses
import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)

import h5py
import numpy as np
import plotly.graph_objects as go

Self: TypeAlias = Any

try:
    import pycolmap
except ImportError:
    pycolmap = None
    print(
        "pycolmap not found, COLMAP support is disabled. "
        "Install it with `pip install pycolmap`."
    )

import torch
import torch.nn.functional as tnf

from ..utils import misc, tensor, tools
from ..visualization import viz3d
from . import transforms as gtr

logger = logging.getLogger(__name__)


class Pose(tensor.TensorWrapper):
    def __init__(self, data_: torch.Tensor):
        self.data_ = data_
        super().__post_init__()

    @classmethod
    def identity(cls, device=None, dtype=None):
        R = torch.eye(3, device=device, dtype=dtype)
        t = torch.zeros(3, device=device, dtype=dtype)
        return cls.from_Rt(R, t)

    @classmethod
    @tensor.autocast
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor):
        """Pose from a rotation matrix and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
        """
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)

    def to_Rt(self, numpy: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert pose to rotation matrix and translation vector."""
        R = self.R.clone()
        t = self.t.clone()
        if numpy:
            R = R.detach().cpu().numpy()
            t = t.detach().cpu().numpy()
        return R, t

    @classmethod
    @tensor.autocast
    def from_aa(cls, aa: torch.Tensor, t: torch.Tensor):
        """Pose from an axis-angle rotation vector and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            aa: axis-angle rotation vector with shape (..., 3).
            t: translation vector with shape (..., 3).
        """
        assert aa.shape[-1] == 3
        assert t.shape[-1] == 3
        assert aa.shape[:-1] == t.shape[:-1]
        return cls.from_Rt(gtr.so3exp_map(aa), t)

    @classmethod
    def from_4x4mat(cls, T: torch.Tensor):
        """Pose from an SE(3) transformation matrix.
        Args:
            T: transformation matrix with shape (..., 4, 4).
        """
        assert T.shape[-2:] == (4, 4)
        return cls.from_projection_matrix(T[..., :3, :4])

    @classmethod
    def from_projection_matrix(cls, P: torch.Tensor):
        """Pose from a projection matrix.
        Args:
            P: projection matrix with shape (..., 3, 4).
        """
        assert P.shape[-2:] == (3, 4)
        R, t = P[..., :3, :3], P[..., :3, 3]
        return cls.from_Rt(R, t)

    @classmethod
    def from_colmap(cls, image: NamedTuple):
        """Pose from a COLMAP Image."""
        return cls.from_Rt(image.qvec2rotmat(), image.tvec)

    @classmethod
    def from_pycolmap(cls, image):
        """Pose from a COLMAP Image."""
        assert pycolmap is not None, "pycolmap is not installed."
        w_t_c = image.cam_from_world()
        return cls.from_Rt(
            torch.from_numpy(w_t_c.rotation.matrix()),
            torch.from_numpy(w_t_c.translation),
        )

    @property
    def R(self) -> torch.Tensor:
        """Underlying rotation matrix with shape (..., 3, 3)."""
        rvec = self.data_[..., :9]
        return rvec.reshape(rvec.shape[:-1] + (3, 3))

    @property
    def t(self) -> torch.Tensor:
        """Underlying translation vector with shape (..., 3)."""
        return self.data_[..., -3:]

    @property
    def P(self) -> torch.Tensor:
        """Projection matrix with shape (..., 3, 4)."""
        return torch.cat([self.R, self.t[..., :, None]], dim=-1)

    @property
    def E(self) -> torch.Tensor:
        """Convert poses to essential matrices."""
        return gtr.skew_symmetric(self.t) @ self.R

    def inv(self) -> "Pose":
        """Invert an SE(3) pose."""
        R = self.R.transpose(-1, -2)
        t = -(R @ self.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    def compose(self, other: "Pose") -> "Pose":
        """Chain two SE(3) poses: T_B2C.compose(T_A2B) -> T_A2C."""
        R = self.R @ other.R
        t = self.t + (self.R @ other.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    @tensor.autocast
    @tensor.autovmap
    def transform(self, p3d: torch.Tensor) -> torch.Tensor:
        """Transform a set of 3D points (or 4D hom.).
        Args:
            p3d: 3D points, numpy array or PyTorch tensor with shape (..., 3).
        """
        assert p3d.shape[-1] in (3, 4), p3d.shape
        sc = 1.0 if p3d.shape[-1] == 3 else p3d[..., -1:]
        return self.R @ p3d[..., :3] + self.t * sc

    def __mul__(self, p3D: torch.Tensor) -> torch.Tensor:
        """Transform a set of 3D points: T_A2B * p3D_A -> p3D_B."""
        return self.transform(p3D)

    def __matmul__(
        self, other: Union["Pose", torch.Tensor]
    ) -> Union["Pose", torch.Tensor]:
        """Transform a set of 3D points: T_A2B * p3D_A -> p3D_B.
        or chain two SE(3) poses: T_B2C @ T_A2B -> T_A2C."""
        if isinstance(other, self.__class__):
            return self.compose(other)
        else:
            return self.transform(other)

    @tensor.autocast
    def J_transform(self, p3d_out: torch.Tensor):
        # [[1,0,0,0,-pz,py],
        #  [0,1,0,pz,0,-px],
        #  [0,0,1,-py,px,0]]
        J_t = torch.diag_embed(torch.ones_like(p3d_out))
        J_rot = -gtr.skew_symmetric(p3d_out)
        J = torch.cat([J_t, J_rot], dim=-1)
        return J  # N x 3 x 6

    def numpy(self) -> Tuple[np.ndarray]:
        return self.R.numpy(), self.t.numpy()

    def magnitude(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Magnitude of the SE(3) transformation.
        Returns:
            dr: rotation anngle in degrees.
            dt: translation distance in meters.
        """
        trace = torch.diagonal(self.R, dim1=-1, dim2=-2).sum(-1)
        cos = torch.clamp((trace - 1) / 2, -1, 1)
        dr = torch.acos(cos).abs() / math.pi * 180
        dt = torch.norm(self.t, dim=-1)
        return dr, dt

    def norm(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.magnitude()

    def angular_drdt(
        self, other: "Pose", stack: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        dr, _ = (self @ other.inv()).magnitude()
        i_p_j = tnf.normalize(self.t, dim=-1)
        i_p_j_gt = tnf.normalize(other.t, dim=-1)
        dt = torch.einsum("...d,...d->...", i_p_j, i_p_j_gt)
        dt = dt.clip(-1, 1).arccos().rad2deg()
        dt = torch.minimum(dt, 180 - dt)
        # dr = torch.minimum(dr, 180 - dr)
        if stack:
            return torch.stack([dr, dt], -1)
        return dr, dt

    def angular_error(self, other: "Pose", agg="max") -> torch.Tensor:
        return {
            "max": lambda x: x.max(-1).values,
            "mean": lambda x: x.mean(-1),
            "dr": lambda x: x[..., 0],
            "dt": lambda x: x[..., 1],
        }[agg](self.angular_drdt(other, stack=True))

    def _opening_angle(self, return_cos: bool = False) -> float:
        v0 = torch.zeros_like(self.t)
        v0[..., -1] = 1.0
        v1 = self.R[..., 2, :]
        v01 = self.inv().t
        v01 = v01 / v01.norm()
        n = v0.cross(v01, dim=-1)
        n = n / n.norm()
        cos = n.dot(v1)
        return cos if return_cos else (90 - torch.acos(cos) * 180.0 / 3.1415).abs()

    def opening_angle(self, return_cos: bool = False) -> float:
        return 0.5 * (
            self._opening_angle(return_cos=return_cos)
            + self.inv()._opening_angle(return_cos=return_cos)
        )

    @classmethod
    def exp(cls, delta: torch.Tensor):
        dr, dt = delta.split(3, dim=-1)
        return cls.from_aa(dr, dt)

    def manifold(self, delta: torch.Tensor | None = None) -> torch.Tensor:
        if delta is None:
            delta = torch.zeros_like(self.data_[..., :6])
        return delta

    def update(self, delta: torch.Tensor | Self, inplace: bool = False) -> "Pose":
        delta = delta.to(self.dtype).to(self.device)
        if not isinstance(delta, self.__class__):
            delta = Pose.exp(delta)
        # Inverted!
        updated_pose = self @ delta
        if inplace:
            self.data_ = updated_pose.data_
            return self
        else:
            return updated_pose

    def __repr__(self):
        return f"Pose: {self.shape} {self.dtype} {self.device}"


class Camera(tensor.TensorWrapper, tensor_only=False, nocast=True):
    eps: float = 1e-4
    share_df: bool = True  # share fx, fy updates (global): has no impact outside BA

    def __init__(self, data_: torch.Tensor, share_df: bool = False, eps: float = 1e-4):
        assert data_.shape[-1] in {6, 8, 10}
        self.data_ = data_
        self.share_df = share_df
        self.eps = eps
        super().__post_init__()

    @classmethod
    def from_colmap(cls, camera: Union[Dict, NamedTuple]):
        """Camera from a COLMAP Camera tuple or dictionary.
        We use the corner-convetion from COLMAP (center of top left pixel is (0.5, 0.5))
        """
        if isinstance(camera, tuple):
            camera = camera._asdict()

        model = camera["model"]
        params = camera["params"]

        if model in ["OPENCV", "PINHOLE", "RADIAL"]:
            (fx, fy, cx, cy), params = np.split(params, [4])
        elif model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            (f, cx, cy), params = np.split(params, [3])
            fx = fy = f
            if model == "SIMPLE_RADIAL":
                params = np.r_[params, 0.0]
        else:
            raise NotImplementedError(model)

        data_ = np.r_[camera["width"], camera["height"], fx, fy, cx, cy, params]
        return cls(torch.from_numpy(data_))

    @classmethod
    def from_pycolmap(cls, camera):
        assert pycolmap is not None, "pycolmap is not installed."
        return cls.from_colmap(
            {
                "model": camera.model.name,
                "width": camera.width,
                "height": camera.height,
                "params": camera.params,
            }
        )

    @classmethod
    def data_from_K(cls, K: torch.Tensor):
        cx, cy = K[..., 0, 2], K[..., 1, 2]
        fx, fy = K[..., 0, 0], K[..., 1, 1]
        data = torch.stack([2 * cx, 2 * cy, fx, fy, cx, cy], -1)
        return data

    @classmethod
    @tensor.autocast
    def from_calibration_matrix(cls, K: torch.Tensor):
        return cls(cls.data_from_K(K))

    @classmethod
    def from_image(cls, img: torch.Tensor):
        h, w = img.shape[-2:]
        cx, cy = w / 2, h / 2
        fx = fy = 0.5 * max(w, h)
        data = torch.tensor([w, h, fx, fy, cx, cy], device=img.device, dtype=img.dtype)
        return cls(data)

    @tensor.autocast
    def calibration_matrix(self):
        K = torch.zeros(
            *self.data_.shape[:-1],
            3,
            3,
            device=self.data_.device,
            dtype=self.data_.dtype,
        )
        K[..., 0, 2] = self.data_[..., 4]
        K[..., 1, 2] = self.data_[..., 5]
        K[..., 0, 0] = self.data_[..., 2]
        K[..., 1, 1] = self.data_[..., 3]
        K[..., 2, 2] = 1.0
        return K

    @property
    def K(self):
        return self.calibration_matrix()

    @property
    def size(self) -> torch.Tensor:
        """Size (width height) of the images, with shape (..., 2)."""
        return self.data_[..., :2]

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy) with shape (..., 2)."""
        return self.data_[..., 2:4]

    def fov(self) -> torch.Tensor:
        """Vertical field of view in radians."""
        return gtr.focal2fov(self.f, self.size)

    @property
    def c(self) -> torch.Tensor:
        """Principal points (cx, cy) with shape (..., 2)."""
        return self.data_[..., 4:6]

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., {0, 2, 4})."""
        return self.data_[..., 6:]

    @tensor.autocast
    def scale(self, scales: torch.Tensor):
        """Update the camera parameters after resizing an image."""
        s = scales
        data = torch.cat([self.size * s, self.f * s, self.c * s, self.dist], -1)
        return self.__class__(data)

    @tensor.autocast
    def compose_image_transform(
        self, new_t_img: torch.Tensor, inplace: bool = False
    ) -> "Camera":
        """Update the camera parameters after an image space transformation.
        Args:
            new_t_img: 3x3 image transformation matrix.
            inplace: whether to update the current camera or return a new one.
        """
        K = self.calibration_matrix()
        new_K = new_t_img.to(K) @ K
        cls = self.__class__
        newdata_ = cls.data_from_K(new_K)

        alldata_ = torch.cat([newdata_, self.dist], -1)
        if inplace:
            self.data_ = alldata_
            return self
        return cls(alldata_)

    def empty_image(self, rgb: bool = False) -> torch.Tensor:  # H X W or 3 X H X W
        """Create an empty image with the camera size."""
        assert self.data_.ndim == 1, "Only for single cameras."
        w, h = self.size.long().tolist()
        dims = (3, h, w) if rgb else (h, w)
        return torch.zeros(dims, dtype=self.data_.dtype, device=self.data_.device)

    def crop(self, left_top: Tuple[float], size: Tuple[int]):
        """Update the camera parameters after cropping an image."""
        left_top = self.data_.new_tensor(left_top)
        size = self.data_.new_tensor(size)
        data = torch.cat([size, self.f, self.c - left_top, self.dist], -1)
        return self.__class__(data)

    @tensor.autocast
    @tensor.autovmap
    def in_image(self, p2d: torch.Tensor):
        """Check if 2D points are within the image boundaries."""
        assert p2d.shape[-1] == 2
        # assert p2d.shape[:-2] == self.shape  # allow broadcasting
        size = self.size
        valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
        return valid

    @tensor.autovmap
    def denormalize_pixels(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return (p2d + 1) / 2 * self.size

    @tensor.autovmap
    def normalize_pixels(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert pixel coordinates into normalized 2D coordinates."""
        return p2d / self.size * 2 - 1

    @tensor.autocast
    @tensor.autovmap
    def project(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Project 3D points into the camera plane and check for visibility."""
        z = p3d[..., -1:]
        valid = z > self.eps
        z = z.clamp(min=self.eps)
        return p3d[..., :-1] / z, valid[..., 0]

    @tensor.autocast
    @tensor.autovmap
    def distort(self, pts: torch.Tensor) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates
        and check for validity of the distortion model.
        """
        assert pts.shape[-1] == 2
        # assert pts.shape[:-2] == self.shape  # allow broadcasting
        return gtr.distort_points(pts, self.dist)

    @tensor.autocast
    @tensor.autovmap
    def denormalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return p2d * self.f + self.c

    @tensor.autocast
    @tensor.autovmap
    def normalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return (p2d - self.c) / self.f

    @tensor.autocast
    @tensor.autovmap
    def cam2image(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Transform 3D points into 2D pixel coordinates."""
        p2d, visible = self.project(p3d)
        p2d, mask = self.distort(p2d)
        p2d = self.denormalize(p2d)
        valid = visible & mask & self.in_image(p2d)
        return p2d, valid

    def image2cam(self, p2d: torch.Tensor, homogeneous: bool = True) -> torch.Tensor:
        """Convert 2D pixel corrdinates to 3D points with z=1"""
        assert self.data_.shape
        p2d = self.normalize(p2d)
        # iterative undistortion
        if homogeneous:
            return gtr.to_homogeneous(p2d)
        else:
            return p2d

    @property
    def params(self) -> torch.Tensor:
        if self.share_df:
            return self.data_[..., 2:3]
        else:
            return self.data_[..., 2:4]

    @params.setter
    def params(self, value: torch.Tensor) -> None:
        # Broadcast in the case of share_df
        self.data_[..., 2:4] = value

    def manifold(self, delta: torch.Tensor | None = None) -> torch.Tensor:
        if delta is None:
            return torch.zeros_like(self.params)
        return delta

    def update(self, delta: torch.Tensor | Self, inplace=False) -> "Camera":
        delta = delta.to(self.dtype).to(self.device)
        if isinstance(delta, self.__class__):
            delta = delta.params

        if inplace:
            self.params += delta
            return self
        else:
            cam_copy = self.clone()
            cam_copy.params += delta
            return cam_copy

    def to_cameradict(self, camera_model: Optional[str] = None) -> List[Dict]:
        data = self.data_.clone()
        if data.dim() == 1:
            data = data.unsqueeze(0)
        assert data.dim() == 2
        b, d = data.shape
        if camera_model is None:
            camera_model = {6: "PINHOLE", 8: "RADIAL", 10: "OPENCV"}[d]
        cameras = []
        for i in range(b):
            if camera_model.startswith("SIMPLE_"):
                params = [x.item() for x in data[i, 3 : min(d, 7)]]
            else:
                params = [x.item() for x in data[i, 2:]]
            cameras.append(
                {
                    "model": camera_model,
                    "width": int(data[i, 0].item()),
                    "height": int(data[i, 1].item()),
                    "params": params,
                }
            )
        return cameras if self.data_.dim() == 2 else cameras[0]

    def __repr__(self):
        return f"Camera {self.shape} {self.dtype} {self.device}"


class PerspectiveCamera(Camera):
    # data: w,h, k00, k01, k02, k10, k11, k12, k20, k21, k22, params

    @classmethod
    def from_colmap(cls, camera: Union[Dict, NamedTuple]):
        if isinstance(camera, tuple):
            camera = camera._asdict()

        model = camera["model"]
        params = camera["params"]

        if model in ["OPENCV", "PINHOLE", "RADIAL"]:
            (fx, fy, cx, cy), params = np.split(params, [4])
        elif model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            (f, cx, cy), params = np.split(params, [3])
            fx = fy = f
            if model == "SIMPLE_RADIAL":
                params = np.r_[params, 0.0]
        else:
            raise NotImplementedError(model)

        data_ = np.r_[
            camera["width"],
            camera["height"],
            fx,
            0.0,
            cx,
            0.0,
            fy,
            cy,
            0.0,
            0.0,
            1.0,
            params,
        ]
        return cls(torch.from_numpy(data_))

    @classmethod
    def data_from_K(cls, K: torch.Tensor):
        cx, cy = K[..., 0, 2], K[..., 1, 2]
        data = torch.concat([2 * cx[..., None], 2 * cy[..., None], K.flatten(-2)], -1)
        return data

    @classmethod
    def from_image(cls, img):
        h, w = img.shape[-2:]
        cx, cy = w / 2, h / 2
        fx = fy = 0.5 * max(w, h)
        data = torch.tensor(
            [w, h, fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
            device=img.device,
            dtype=img.dtype,
        )
        return cls(data)

    @tensor.autocast
    def calibration_matrix(self):
        return self.data_[..., 2:11].reshape(self.data_.shape[:-1] + (3, 3))

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy) with shape (..., 2)."""
        return self.data_[..., (2, 6)]

    @property
    def c(self) -> torch.Tensor:
        """Principal points (cx, cy) with shape (..., 2)."""
        return self.data_[..., (4, 7)]

    @tensor.autocast
    def scale(self, scales):
        size = self.size * scales
        K = self.calibration_matrix() @ torch.diag_embed(gtr.to_homogeneous(scales))
        data = torch.cat([size, K.flatten(-2), self.dist], -1)
        return self.__class__(data)

    @tensor.autocast
    @tensor.autovmap
    def project(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        return super().project(p3d)

    @tensor.autocast
    @tensor.autovmap
    def denormalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return gtr.from_homogeneous(
            gtr.to_homogeneous(p2d) @ self.calibration_matrix().transpose(-1, -2)
        )

    @tensor.autocast
    @tensor.autovmap
    def normalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return gtr.from_homogeneous(
            gtr.to_homogeneous(p2d)
            @ self.calibration_matrix().inverse().transpose(-1, -2)
        )

    @property
    def params(self) -> torch.Tensor:
        return self.data_[..., 2:]

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., {0, 2, 4})."""
        return self.data_[..., 11:]

    def to_cameradict(self, camera_model=None):
        raise NotImplementedError("PerspectiveCamera does not support to_cameradict.")

    def __repr__(self):
        return f"PerspectiveCamera {self.shape} {self.dtype} {self.device}"


@dataclasses.dataclass
class Reconstruction:
    w_t_c: Pose  # batched
    cameras: Camera  # batched
    camera_idx: torch.Tensor  # int
    image_names: list[str]
    registered: torch.Tensor  # bool

    @classmethod
    def from_colmap(cls, colmap_model: Path):
        if isinstance(colmap_model, Path):
            colmap_model = pycolmap.Reconstruction(colmap_model)
        reg_image_ids = sorted(colmap_model.reg_image_ids())
        reg_images = [
            colmap_model.images[reg_image_id] for reg_image_id in reg_image_ids
        ]
        camera_ids, cameras = list(
            zip(
                *[
                    (cam_id, Camera.from_pycolmap(cam))
                    for cam_id, cam in colmap_model.cameras.items()
                ]
            )
        )
        cameras = torch.stack(cameras)
        camera_idx = [camera_ids.index(img.camera_id) for img in reg_images]
        c_t_w = torch.stack([Pose.from_pycolmap(img) for img in reg_images])
        image_names = [img.name for img in reg_images]

        return cls(
            w_t_c=c_t_w.inv().float(),
            cameras=cameras.float(),
            camera_idx=torch.Tensor(camera_idx).long(),
            image_names=image_names,
            registered=torch.ones(len(reg_image_ids), dtype=bool),
        )

    def get_camera(self, image_id: int | Sequence[int]) -> Camera:
        return self.cameras[self.camera_idx[image_id]]

    def to(self, device: torch.device | str) -> "Reconstruction":
        for attr in dataclasses.fields(self):
            if attr.name == "registered":
                continue
            setattr(
                self, attr.name, misc.batch_to_device(getattr(self, attr.name), device)
            )
        return self

    def cuda(self) -> "Reconstruction":
        """Move the reconstruction to the GPU."""
        return self.to("cuda")

    def cpu(self) -> "Reconstruction":
        """Move the reconstruction to the CPU."""
        return self.to("cpu")

    def clone(self) -> "Reconstruction":
        return deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"Reconstruction: {self.num_images} images, "
            f"{self.num_reg_images} registered, "
            f"cameras {self.cameras.shape}, "
            f"poses {self.w_t_c.shape}"
            f" {self.w_t_c.dtype} {self.w_t_c.device}"
        )

    @property
    def reg_image_ids(self) -> list[int]:
        return torch.where(self.registered)[0].tolist()

    @property
    def image_ids(self) -> list[int]:
        return torch.arange(self.num_images).tolist()

    @property
    def non_reg_image_ids(self) -> list[int]:
        return torch.where(~self.registered)[0].tolist()

    @property
    def num_images(self) -> int:
        return self.w_t_c.shape[0]

    @property
    def num_reg_images(self) -> int:
        return self.registered.sum().item()

    @property
    def device(self):
        return self.w_t_c.device

    def compare_poses_to(
        self,
        gt: "Reconstruction",
        thresholds: tuple[int, ...] = (1, 3, 5, 10, 20),  # degrees.
        image_ids: Sequence[int] | None = None,  # Where to compare poses.
        which: str = "max",  # dr, dt, mean
    ) -> tuple[torch.Tensor, Sequence[float]]:
        image_ids = image_ids if image_ids is not None else self.image_ids

        # Get image ids in the ground truth reconstruction
        gt_name_to_id = {n: i for i, n in enumerate(gt.image_names)}
        gt_image_ids = [gt_name_to_id[self.image_names[i]] for i in image_ids]

        # Get requested poses
        w_t_c = self.w_t_c[image_ids]
        w_tgt_c = gt.w_t_c[gt_image_ids]

        # Compute relative poses
        cj_t_ci = w_t_c[None].inv() @ w_t_c[:, None]
        is_registered = self.registered[image_ids]
        cj_tgt_ci = w_tgt_c[None].inv() @ w_tgt_c[:, None]

        # Compute angular errors between all pairs of poses
        errors = cj_t_ci.angular_error(cj_tgt_ci, agg=which)  # M x M

        # Set invalid pairs to max error (180 degrees)
        valid = is_registered[:, None] & is_registered[None, :]
        errors = torch.where(valid.to(errors.device), errors, 180.0)

        # Remove diagonal elements (self-self)
        errors_without_diag = errors[~torch.eye(errors.shape[0], dtype=bool)]
        return (
            errors,
            tools.AUCMetric(thresholds, errors_without_diag.cpu().numpy()).compute(),
        )

    def calibration_error_to(
        self,
        gt: "Reconstruction",
        image_ids: Sequence[int] | None = None,  # Where to compare cameras.
        which: Sequence[str] = ["mean"],  # max, mean, median
        fov_thresholds: tuple[float, ...] = (1, 5),  # degrees
    ) -> dict[str, torch.Tensor]:
        image_ids = image_ids if image_ids is not None else self.image_ids

        # Get image ids in the ground truth reconstruction
        gt_name_to_id = {n: i for i, n in enumerate(gt.image_names)}
        gt_image_ids = [gt_name_to_id[self.image_names[i]] for i in image_ids]

        # Get requested cameras
        cameras = self.get_camera(image_ids)
        cameras_gt = gt.get_camera(gt_image_ids)

        # Compute focal length errors
        abs_error = (cameras_gt.f - cameras.f).abs()
        rel_error = abs_error / cameras_gt.f
        fov_error = (cameras_gt.fov() - cameras.fov()).abs().mean(-1) * 180.0 / torch.pi
        do_compare = self.registered[image_ids].bool().to(abs_error.device)
        errors = {
            "f_abs": torch.where(do_compare, abs_error.max(-1).values, 5000),
            "f_rel": torch.where(
                do_compare,
                (rel_error.max(-1).values * 100),
                100.0,
            ),
            "fov": torch.where(do_compare, fov_error, 180.0),
        }

        metrics = {}
        for k, v in errors.items():
            for agg in which:
                metrics[f"{k}_{agg}"] = {
                    "max": torch.max,
                    "mean": torch.mean,
                    "median": torch.median,
                }[agg](v).item()

        aucs = tools.AUCMetric(fov_thresholds, errors["fov"].cpu().numpy()).compute()
        for th, auc in zip(fov_thresholds, aucs):
            metrics[f"fov_AUC@{th}°"] = auc
        return errors, metrics

    @classmethod
    def empty_like(cls, ref_sfm):
        w_t_c = torch.stack([Pose.identity(p.device, p.dtype) for p in ref_sfm.w_t_c])
        registered = torch.zeros_like(ref_sfm.registered).bool()
        return cls(
            w_t_c=w_t_c,
            image_names=ref_sfm.image_names,
            cameras=ref_sfm.cameras.clone(),
            camera_idx=ref_sfm.camera_idx.clone(),
            registered=registered,
        )

    # ------------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------------

    def to_h5(self, output_file: Path):
        with h5py.File(str(output_file), "w") as h5f:
            h5f.createdata_set("image_names", data=self.image_names)
            h5f.createdata_set(
                "cameras", data=self.cameras.data_.detach().cpu().numpy()
            )
            for image_id, image_name in enumerate(self.image_names):
                grp = h5f.create_group(image_name)
                grp.createdata_set(
                    "w_t_c", data=self.w_t_c[image_id].data_.detach().cpu().numpy()
                )
                grp.attrs["camera_idx"] = (
                    self.camera_idx[image_id].detach().cpu().numpy()
                )
                grp.attrs["registered"] = (
                    self.registered[image_id].detach().cpu().numpy()
                )

    @classmethod
    def from_h5(cls, input_file: Path):
        with h5py.File(str(input_file), "r") as h5f:
            image_names = h5f["image_names"].__array__().astype(str).tolist()
            cameras = Camera(torch.from_numpy(h5f["cameras"].__array__()))
            pose, camera_idx, registered = [], [], []
            for image_id, image_name in enumerate(image_names):
                grp = h5f[image_name]
                pose.append(grp["w_t_c"].__array__())
                camera_idx.append(grp.attrs["camera_idx"])
                registered.append(grp.attrs["registered"])

        return cls(
            w_t_c=Pose(torch.from_numpy(np.array(pose))).float(),
            cameras=cameras.float(),
            camera_idx=torch.Tensor(camera_idx).int(),
            registered=torch.Tensor(registered).long(),
            image_names=image_names,
        )

    def plot3d(
        self,
        fig: go.Figure | None = None,
        image_ids: Sequence[int] | None = None,
        height: int = 600,
        projection: str = "orthographic",
        cs: int = 10.0,
    ) -> go.Figure:
        """Plot the reconstruction in a matplotlib 3D axis."""
        if fig is None:
            fig = viz3d.init_figure(height=height, projection=projection)
        if image_ids is None:
            image_ids = self.image_ids
        viz3d.plot_cameras(
            fig,
            self.w_t_c[image_ids],
            self.get_camera(image_ids),
            name=self.image_names,
            scale=cs,
            legendgroup="cameras",
        )
        return fig


if __name__ == "__main__":

    # import torch._C._functorch as cft  # type: ignore

    # def fn0(x):
    #     print("HIIIIII", x, type(x))

    #     print(cft.is_batchedtensor(x))
    #     print(cft.is_legacy_batchedtensor(x))
    #     # x.transform
    #     return x

    # print(torch.vmap(fn0)(Pose(torch.randn(5, 12))))
    print(Pose(torch.rand(5, 12)))

    print(isinstance(Pose(torch.rand(5, 12)), tensor.TensorWrapper))

    def fn(x, y):
        p = Pose(x)
        print(p.R.shape, p.shape, p.device)
        return p, p.transform(y)

    print(torch.vmap(fn)(torch.randn(2, 12), torch.rand(2, 3, 3)))

    print(Camera(torch.rand(5, 6)))
