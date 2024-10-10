import cv2
import numpy as np
import torch
import kornia


def read_image(path, grayscale=False):
    """Read an image from path as RGB or grayscale"""
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][:: -1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array(
            [[fy, 0.0, cy], [0.0, fx, w - cx], [0.0, 0.0, 1.0]], dtype=K.dtype
        )
    elif rot == 2:
        return np.array(
            [[fx, 0.0, w - cx], [0.0, fy, h - cy], [0.0, 0.0, 1.0]],
            dtype=K.dtype,
        )
    else:  # if rot == 3:
        return np.array(
            [[fy, 0.0, h - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]], dtype=K.dtype
        )


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array(
            [
                [np.cos(r), -np.sin(r), 0.0, 0.0],
                [np.sin(r), np.cos(r), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    """Scale intrinsics after resizing the corresponding image."""
    scales = np.diag(np.concatenate([scales, [1.0]]))
    return np.dot(scales.astype(K.dtype, copy=False), K)


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


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


def resize_img_kornia(img: torch.Tensor, size: int) -> torch.Tensor:
    """
    This resize function has similar functionality to ImagePreprocessor resize.
    Here we resize and keep aspect ratio by scaling long side to 'size' abd scale the other part accoringly.

    Args:
        img (torch.Tensor): image to resize
        size (int): shape to resize to

    Returns:
        torch.Tensor: reshaped image
    """
    resized = kornia.geometry.transform.resize(
            img,
            size,
            side="long",
            antialias=True,
            align_corners=None,
            interpolation='bilinear',
        )
    return resized


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


def sample_homography(
    shape,
    perspective=True,
    scaling=True,
    rotation=True,
    translation=True,
    n_scales=5,
    n_angles=25,
    scaling_amplitude=0.1,
    perspective_amplitude_x=0.1,
    perspective_amplitude_y=0.1,
    patch_ratio=0.5,
    max_angle=1.57,
    allow_artifacts=False,
    translation_overflow=0.0,
):
    """Computes the homography transformation from a random patch in the
        original image to a warped projection with the same image size. The
        original patch, which is initialized with a simple half-size centered
        crop, is iteratively projected, scaled, rotated and translated.

    Args:
        shape: A tuple specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        An np.array of shape `[3, 3]` corresponding to the homography.
    """
    # Convert shape to ndarry
    if not isinstance(shape, np.ndarray):
        shape = np.array(shape)

    # Corners of the output patch
    margin = (1 - patch_ratio) / 2
    pts1 = margin + np.array(
        [[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]]
    )
    # Corners of the intput image
    pts2 = pts1.copy()

    # Random perspective and affine perturbations
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)

        # normal distribution with mean=0, std=perspective_amplitude_y/2
        perspective_displacement = np.random.normal(
            0.0, perspective_amplitude_y / 2, [1]
        )
        h_displacement_left = np.random.normal(0.0, perspective_amplitude_x / 2, [1])
        h_displacement_right = np.random.normal(0.0, perspective_amplitude_x / 2, [1])
        pts2 += np.stack(
            [
                np.concatenate([h_displacement_left, perspective_displacement], 0),
                np.concatenate([h_displacement_left, -perspective_displacement], 0),
                np.concatenate([h_displacement_right, perspective_displacement], 0),
                np.concatenate([h_displacement_right, -perspective_displacement], 0),
            ]
        )

    # Random scaling: sample several scales, check collision with borders,
    # randomly pick a valid one
    if scaling:
        scales = np.concatenate(
            [[1.0], np.random.normal(1, scaling_amplitude / 2, [n_scales])], 0
        )
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[None, ...] * scales[..., None, None] + center
        # all scales are valid except scale=1
        if allow_artifacts:
            valid = np.arange(n_scales)
        else:
            valid = np.where(np.all((scaled >= 0.0) & (scaled < 1.0), (1, 2)))[0]
        idx = valid[np.random.uniform(0.0, valid.shape[0], ()).astype(np.int32)]
        pts2 = scaled[idx]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += (
            np.stack(
                [
                    np.random.uniform(-t_min[0], t_max[0], ()),
                    np.random.uniform(-t_min[1], t_max[1], ()),
                ]
            )
        )[None, ...]

    # Random rotation: sample several rotations, check collision with borders,
    # randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        # in case no rotation is valid
        angles = np.concatenate([[0.0], angles], axis=0)
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(
            np.stack(
                [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)],
                axis=1,
            ),
            [-1, 2, 2],
        )
        rotated = (
            np.matmul(
                np.tile((pts2 - center)[None, ...], [n_angles + 1, 1, 1]), rot_mat
            )
            + center
        )
        if allow_artifacts:
            valid = np.array(range(n_angles))  # all angles are valid, except angle=0
        else:
            valid = np.where(np.all((rotated >= 0.0) & (rotated < 1.0), axis=(1, 2)))[0]
        idx = valid[np.random.uniform(0.0, valid.shape[0], ()).astype(np.int32)]
        pts2 = rotated[idx]

    # Rescale to actual size
    shape = shape[::-1].astype(np.float32)  # different convention [y, x]
    pts1 *= shape[None, ...]
    pts2 *= shape[None, ...]

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(
        np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0)
    )
    homo_vec, _, _, _ = np.linalg.lstsq(a_mat, p_mat, rcond=None)

    # Compose the homography vector back to matrix
    homo_mat = np.concatenate(
        [
            homo_vec[0:3, 0][None, ...],
            homo_vec[3:6, 0][None, ...],
            np.concatenate((homo_vec[6], homo_vec[7], [1]), axis=0)[None, ...],
        ],
        axis=0,
    )

    return homo_mat


def warp_points(points, H):
    """Warp 2D points by an homography H."""
    n_points = points.shape[0]
    reproj_points = points.copy()[:, [1, 0]]
    reproj_points = np.concatenate([reproj_points, np.ones((n_points, 1))], axis=1)
    reproj_points = H.dot(reproj_points.transpose()).transpose()
    reproj_points = reproj_points[:, :2] / reproj_points[:, 2:]
    reproj_points = reproj_points[:, [1, 0]]
    return reproj_points


def warp_lines(lines, H):
    """Warp lines of the shape [N, 2, 2] by an homography H."""
    return warp_points(lines.reshape(-1, 2), H).reshape(-1, 2, 2)
