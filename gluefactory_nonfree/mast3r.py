"""Wrapper for the MASt3R foundation model. Adapted from MP-SfM."""

import os
import sys

import numpy as np
import torch
import torchvision.transforms as tvf

from gluefactory import settings
from gluefactory.models.base_model import BaseModel

mast3r_root_dir = settings.THIRD_PARTY_PATH / "mast3r"
sys.path.append(str(mast3r_root_dir))  # noqa: E402
dust3r_root_dir = settings.THIRD_PARTY_PATH / "mast3r/dust3r"
sys.path.append(str(dust3r_root_dir))  # noqa: E402
curope_root_dir = settings.THIRD_PARTY_PATH / "mast3r/dust3r/croco/models/curope"
sys.path.append(str(curope_root_dir))  # noqa: E402


from mast3r.fast_nn import fast_reciprocal_NNs  # noqa: E402
from mast3r.model import AsymmetricMASt3R, load_model  # noqa: E402


def symmetric_inference(model, img1, img2):
    shape1 = torch.tensor(img1.shape[-2:])[None].to(img1.device, non_blocking=True)
    shape2 = torch.tensor(img2.shape[-2:])[None].to(img2.device, non_blocking=True)

    # compute encoder only once
    feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
        return res1, res2

    # decoder 1-2
    res11, res21 = decoder(feat1, feat2, pos1, pos2, shape1, shape2)
    # decoder 2-1
    res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1)

    return (res11, res21, res22, res12)


def merge_corres(idx1, idx2, shape1=None, shape2=None, ret_xy=True, ret_index=False):
    assert idx1.dtype == idx2.dtype == np.int32

    # unique and sort along idx1
    corres = np.unique(np.c_[idx2, idx1].view(np.int64), return_index=ret_index)

    if ret_index:
        corres, indices = corres
    xy2, xy1 = corres[:, None].view(np.int32).T

    if ret_xy:
        assert shape1 and shape2
        xy1 = np.unravel_index(xy1, shape1)
        xy2 = np.unravel_index(xy2, shape2)
        if ret_xy != "y_x":
            xy1 = xy1[0].base[:, ::-1]
            xy2 = xy2[0].base[:, ::-1]
    if ret_index:
        return xy1, xy2, indices
    return xy1, xy2


def extract_correspondences(feats, qonfs, subsample=8, ptmap_key="pred_desc"):
    feat11, feat21, feat22, feat12 = feats
    qonf11, qonf21, qonf22, qonf12 = qonfs
    assert feat11.shape[:2] == feat12.shape[:2] == qonf11.shape == qonf12.shape
    assert feat21.shape[:2] == feat22.shape[:2] == qonf21.shape == qonf22.shape

    opt = (
        dict(device="cpu", workers=32)
        if "3d" in ptmap_key
        else dict(device=feat11.device, dist="dot", block_size=2**13)
    )

    # matching the two pairs
    idx1 = []
    idx2 = []
    qonf1 = []
    qonf2 = []
    for A, B, QA, QB in [
        (feat11, feat21, qonf11.cpu(), qonf21.cpu()),
        (feat12, feat22, qonf12.cpu(), qonf22.cpu()),
    ]:
        nn1to2 = fast_reciprocal_NNs(
            A, B, subsample_or_initxy1=subsample, ret_xy=False, **opt
        )
        nn2to1 = fast_reciprocal_NNs(
            B, A, subsample_or_initxy1=subsample, ret_xy=False, **opt
        )

        idx1.append(np.r_[nn1to2[0], nn2to1[1]])
        idx2.append(np.r_[nn1to2[1], nn2to1[0]])
        qonf1.append(QA.ravel()[idx1[-1]])
        qonf2.append(QB.ravel()[idx2[-1]])

    # merge corres from opposite pairs
    H1, W1 = feat11.shape[:2]
    H2, W2 = feat22.shape[:2]
    cat = np.concatenate

    xy1, xy2, idx = merge_corres(
        cat(idx1), cat(idx2), (H1, W1), (H2, W2), ret_xy=True, ret_index=True
    )
    corres = (xy1.copy(), xy2.copy(), np.sqrt(cat(qonf1)[idx] * cat(qonf2)[idx]))
    return corres


def map_keypoints_to_original_after_crop(keypoints_crop, cx, cy, halfw, halfh):
    keypoints_scaled = keypoints_crop + np.array([cx - halfw, cy - halfh])
    return keypoints_scaled


# Silence messages from model loading
class AsymmetricMASt3R(AsymmetricMASt3R):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            # added verbose
            return load_model(
                pretrained_model_name_or_path, device="cpu", verbose=False
            )
        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kw)


class Mast3rMatcher(BaseModel):
    default_conf = {
        "model_name": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        "long_edge_size": 512,
        "window": 8,
        "nms_radius": 6,
        "NN_scores_thresh": 0.85,
    }
    required_keys = ["view0", "view1"]

    # Initialize the line matcher
    def _init(self, conf):
        self.net = AsymmetricMASt3R.from_pretrained(conf.model_name)
        self.set_initialized()

    def process_image(self, image, square_ok=False):
        H, W = image.shape[-2:]
        cx, cy = W // 2, H // 2
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = int((3 * halfw / 4) // 8 * 8)
        image = image[..., cy - halfh : cy + halfh, cx - halfw : cx + halfw]
        H2, W2 = image.shape[-2:]
        ImgNorm = tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = ImgNorm(image)
        return image, (cx, cy, halfw, halfh, None, None, H2, W2, H, W)

    def _forward(self, data):
        im0, var0 = self.process_image(data["view0"]["image"])
        im1, var1 = self.process_image(data["view1"]["image"])

        res = symmetric_inference(self.net, im0, im1)
        X11, _, X22, _ = [r["pts3d"][0].cpu().numpy() for r in res]
        C11, _, C22, _ = [r["conf"][0].cpu().numpy() for r in res]
        descs = [r["desc"][0] for r in res]
        qonfs = [r["desc_conf"][0] for r in res]
        pred = {}
        # extracting 2v corres
        corres = extract_correspondences(descs, qonfs, subsample=self.conf.window)
        dkps0, dkps1, scores0 = corres

        dkps0 = map_keypoints_to_original_after_crop(dkps0, *var0[:-6])
        dkps1 = map_keypoints_to_original_after_crop(dkps1, *var1[:-6])

        def extract_rescale_crop(v):
            cropx_a, cropx_b = v[0] - v[2], v[0] + v[2]
            cropy_a, cropy_b = v[1] - v[3], v[1] + v[3]
            return (cropx_a, cropx_b, cropy_a, cropy_b)

        rescale_crop = [extract_rescale_crop(v) for v in [var0, var1]]
        (cropx1a, cropx1b, cropy1a, cropy1b), (cropx2a, cropx2b, cropy2a, cropy2b) = (
            rescale_crop
        )

        for i, (H, W, slicex, slicey, X, C) in enumerate(
            [
                (
                    *var0[-2:],
                    slice(cropx1a, cropx1b),
                    slice(cropy1a, cropy1b),
                    X11,
                    C11,
                ),
                (
                    *var1[-2:],
                    slice(cropx2a, cropx2b),
                    slice(cropy2a, cropy2b),
                    X22,
                    C22,
                ),
            ]
        ):
            pred[f"depth{i}"] = np.zeros((H, W))
            pred[f"variance{i}"] = np.ones((H, W)) * 1e6
            pred[f"valid{i}"] = np.zeros((H, W), dtype=bool)
            pred[f"depth{i}"][slicey, slicex] = X[..., -1]
            pred[f"variance{i}"][slicey, slicex] = (1 / C) ** 2
            pred[f"valid{i}"][slicey, slicex] = True

        pred = {
            "matches0": torch.arange(0, scores0.shape[-1]),
            "matches1": torch.arange(0, scores0.shape[-1]),
            "matching_scores0": scores0,
            "matching_scores1": scores0,
            "keypoints0": dkps0,
            "keypoints1": dkps1,
            "keypoint_scores0": scores0,
            "keypoint_scores1": scores0,
        }

        pred = {k: torch.as_tensor(v, device=im0.device)[None] for k, v in pred.items()}
        return pred

    def loss(self, data, pred):
        # No loss for inference
        raise NotImplementedError("MASt3R does not support training, only inference.")
