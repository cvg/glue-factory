import cv2
import numpy as np
import pycolmap
import torch
from omegaconf import OmegaConf
from scipy.spatial import KDTree

from ..base_model import BaseModel
from ..utils.misc import pad_to_length

EPS = 1e-6


def sift_to_rootsift(x):
    x = x / (np.linalg.norm(x, ord=1, axis=-1, keepdims=True) + EPS)
    x = np.sqrt(x.clip(min=EPS))
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)
    return x


# from OpenGlue
def nms_keypoints(kpts: np.ndarray, responses: np.ndarray, radius: float) -> np.ndarray:
    # TODO: add approximate tree
    kd_tree = KDTree(kpts)

    sorted_idx = np.argsort(-responses)
    kpts_to_keep_idx = []
    removed_idx = set()

    for idx in sorted_idx:
        # skip point if it was already removed
        if idx in removed_idx:
            continue

        kpts_to_keep_idx.append(idx)
        point = kpts[idx]
        neighbors = kd_tree.query_ball_point(point, r=radius)
        # Variable `neighbors` contains the `point` itself
        removed_idx.update(neighbors)

    mask = np.zeros((kpts.shape[0],), dtype=bool)
    mask[kpts_to_keep_idx] = True
    return mask


def detect_kpts_opencv(
    features: cv2.Feature2D, image: np.ndarray, describe: bool = True
) -> np.ndarray:
    """
    Detect keypoints using OpenCV Detector.
    Optionally, perform NMS and filter top-response keypoints.
    Optionally, perform description.
    Args:
        features: OpenCV based keypoints detector and descriptor
        image: Grayscale image of uint8 data type
        describe: flag indicating whether to simultaneously compute descriptors
    Returns:
        kpts: 1D array of detected cv2.KeyPoint
    """
    if describe:
        kpts, descriptors = features.detectAndCompute(image, None)
    else:
        kpts = features.detect(image, None)
    kpts = np.array(kpts)

    responses = np.array([k.response for k in kpts], dtype=np.float32)

    # select all
    top_score_idx = ...
    pts = np.array([k.pt for k in kpts], dtype=np.float32)
    scales = np.array([k.size for k in kpts], dtype=np.float32)
    angles = np.array([k.angle for k in kpts], dtype=np.float32)
    spts = np.concatenate([pts, scales[..., None], angles[..., None]], -1)

    if describe:
        return spts[top_score_idx], responses[top_score_idx], descriptors[top_score_idx]
    else:
        return spts[top_score_idx], responses[top_score_idx]


class SIFT(BaseModel):
    default_conf = {
        "has_detector": True,
        "has_descriptor": True,
        "descriptor_dim": 128,
        "pycolmap_options": {
            "first_octave": 0,
            "peak_threshold": 0.005,
            "edge_threshold": 10,
        },
        "rootsift": True,
        "nms_radius": None,
        "max_num_keypoints": -1,
        "max_num_keypoints_val": None,
        "force_num_keypoints": False,
        "randomize_keypoints_training": False,
        "detector": "pycolmap",  # ['pycolmap', 'pycolmap_cpu', 'pycolmap_cuda', 'cv2']
        "detection_threshold": None,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        self.sift = None  # lazy loading

    @torch.no_grad()
    def extract_features(self, image):
        image_np = image.cpu().numpy()[0]
        assert image.shape[0] == 1
        assert image_np.min() >= -EPS and image_np.max() <= 1 + EPS

        detector = str(self.conf.detector)

        if self.sift is None and detector.startswith("pycolmap"):
            options = OmegaConf.to_container(self.conf.pycolmap_options)
            device = (
                "auto" if detector == "pycolmap" else detector.replace("pycolmap_", "")
            )
            if self.conf.rootsift == "rootsift":
                options["normalization"] = pycolmap.Normalization.L1_ROOT
            else:
                options["normalization"] = pycolmap.Normalization.L2
            if self.conf.detection_threshold is not None:
                options["peak_threshold"] = self.conf.detection_threshold
            options["max_num_features"] = self.conf.max_num_keypoints
            self.sift = pycolmap.Sift(options=options, device=device)
        elif self.sift is None and self.conf.detector == "cv2":
            self.sift = cv2.SIFT_create(contrastThreshold=self.conf.detection_threshold)

        if detector.startswith("pycolmap"):
            keypoints, scores, descriptors = self.sift.extract(image_np)
        elif detector == "cv2":
            # TODO: Check if opencv keypoints are already in corner convention
            keypoints, scores, descriptors = detect_kpts_opencv(
                self.sift, (image_np * 255.0).astype(np.uint8)
            )

        if self.conf.nms_radius is not None:
            mask = nms_keypoints(keypoints[:, :2], scores, self.conf.nms_radius)
            keypoints = keypoints[mask]
            scores = scores[mask]
            descriptors = descriptors[mask]

        scales = keypoints[:, 2]
        oris = np.rad2deg(keypoints[:, 3])

        if self.conf.has_descriptor:
            # We still renormalize because COLMAP does not normalize well,
            # maybe due to numerical errors
            if self.conf.rootsift:
                descriptors = sift_to_rootsift(descriptors)
            descriptors = torch.from_numpy(descriptors)
        keypoints = torch.from_numpy(keypoints[:, :2])  # keep only x, y
        scales = torch.from_numpy(scales)
        oris = torch.from_numpy(oris)
        scores = torch.from_numpy(scores)

        # Keep the k keypoints with highest score
        max_kps = self.conf.max_num_keypoints

        # for val we allow different
        if not self.training and self.conf.max_num_keypoints_val is not None:
            max_kps = self.conf.max_num_keypoints_val

        if max_kps is not None and max_kps > 0:
            if self.conf.randomize_keypoints_training and self.training:
                # instead of selecting top-k, sample k by score weights
                raise NotImplementedError
            elif max_kps < scores.shape[0]:
                # TODO: check that the scores from PyCOLMAP are 100% correct,
                # follow https://github.com/mihaidusmanu/pycolmap/issues/8
                indices = torch.topk(scores, max_kps).indices
                keypoints = keypoints[indices]
                scales = scales[indices]
                oris = oris[indices]
                scores = scores[indices]
                if self.conf.has_descriptor:
                    descriptors = descriptors[indices]

        if self.conf.force_num_keypoints:
            keypoints = pad_to_length(
                keypoints,
                max_kps,
                -2,
                mode="random_c",
                bounds=(0, min(image.shape[1:])),
            )
            scores = pad_to_length(scores, max_kps, -1, mode="zeros")
            scales = pad_to_length(scales, max_kps, -1, mode="zeros")
            oris = pad_to_length(oris, max_kps, -1, mode="zeros")
            if self.conf.has_descriptor:
                descriptors = pad_to_length(descriptors, max_kps, -2, mode="zeros")

        pred = {
            "keypoints": keypoints,
            "scales": scales,
            "oris": oris,
            "keypoint_scores": scores,
        }

        if self.conf.has_descriptor:
            pred["descriptors"] = descriptors
        return pred

    @torch.no_grad()
    def _forward(self, data):
        pred = {
            "keypoints": [],
            "scales": [],
            "oris": [],
            "keypoint_scores": [],
            "descriptors": [],
        }

        image = data["image"]
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True).cpu()

        for k in range(image.shape[0]):
            img = image[k]
            if "image_size" in data.keys():
                # avoid extracting points in padded areas
                w, h = data["image_size"][k]
                img = img[:, :h, :w]
            p = self.extract_features(img)
            for k, v in p.items():
                pred[k].append(v)

        if (image.shape[0] == 1) or self.conf.force_num_keypoints:
            pred = {k: torch.stack(pred[k], 0) for k in pred.keys()}

        pred = {k: pred[k].to(device=data["image"].device) for k in pred.keys()}

        pred["oris"] = torch.deg2rad(pred["oris"])
        return pred

    def loss(self, pred, data):
        raise NotImplementedError
