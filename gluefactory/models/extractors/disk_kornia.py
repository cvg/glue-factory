import kornia
import torch

from ..base_model import BaseModel
from ..utils.misc import pad_and_stack


class DISK(BaseModel):
    default_conf = {
        "weights": "depth",
        "dense_outputs": False,
        "max_num_keypoints": None,
        "desc_dim": 128,
        "nms_window_size": 5,
        "detection_threshold": 0.0,
        "force_num_keypoints": False,
        "pad_if_not_divisible": True,
        "chunk": 4,  # for reduced VRAM in training
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        self.model = kornia.feature.DISK.from_pretrained(conf.weights)
        self.set_initialized()

    def _get_dense_outputs(self, images):
        B = images.shape[0]
        if self.conf.pad_if_not_divisible:
            h, w = images.shape[2:]
            pd_h = 16 - h % 16 if h % 16 > 0 else 0
            pd_w = 16 - w % 16 if w % 16 > 0 else 0
            images = torch.nn.functional.pad(images, (0, pd_w, 0, pd_h), value=0.0)

        heatmaps, descriptors = self.model.heatmap_and_dense_descriptors(images)
        if self.conf.pad_if_not_divisible:
            heatmaps = heatmaps[..., :h, :w]
            descriptors = descriptors[..., :h, :w]

        keypoints = kornia.feature.disk.detector.heatmap_to_keypoints(
            heatmaps,
            n=self.conf.max_num_keypoints,
            window_size=self.conf.nms_window_size,
            score_threshold=self.conf.detection_threshold,
        )

        features = []
        for i in range(B):
            features.append(keypoints[i].merge_with_descriptors(descriptors[i]))

        return features, descriptors

    def _forward(self, data):
        image = data["image"]

        keypoints, scores, descriptors = [], [], []
        if self.conf.dense_outputs:
            dense_descriptors = []
        chunk = self.conf.chunk
        for i in range(0, image.shape[0], chunk):
            if self.conf.dense_outputs:
                features, d_descriptors = self._get_dense_outputs(
                    image[: min(image.shape[0], i + chunk)]
                )
                dense_descriptors.append(d_descriptors)
            else:
                features = self.model(
                    image[: min(image.shape[0], i + chunk)],
                    n=self.conf.max_num_keypoints,
                    window_size=self.conf.nms_window_size,
                    score_threshold=self.conf.detection_threshold,
                    pad_if_not_divisible=self.conf.pad_if_not_divisible,
                )
            keypoints += [f.keypoints for f in features]
            scores += [f.detection_scores for f in features]
            descriptors += [f.descriptors for f in features]
            del features

        if self.conf.force_num_keypoints:
            # pad to target_length
            target_length = self.conf.max_num_keypoints
            keypoints = pad_and_stack(
                keypoints,
                target_length,
                -2,
                mode="random_c",
                bounds=(
                    0,
                    data.get("image_size", torch.tensor(image.shape[-2:])).min().item(),
                ),
            )
            scores = pad_and_stack(scores, target_length, -1, mode="zeros")
            descriptors = pad_and_stack(descriptors, target_length, -2, mode="zeros")
        else:
            keypoints = torch.stack(keypoints, 0)
            scores = torch.stack(scores, 0)
            descriptors = torch.stack(descriptors, 0)

        pred = {
            "keypoints": keypoints.to(image) + 0.5,
            "keypoint_scores": scores.to(image),
            "descriptors": descriptors.to(image),
        }
        if self.conf.dense_outputs:
            pred["dense_descriptors"] = torch.cat(dense_descriptors, 0)
        return pred

    def loss(self, pred, data):
        raise NotImplementedError
