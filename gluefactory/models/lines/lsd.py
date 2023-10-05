import numpy as np
import torch
from joblib import Parallel, delayed
from pytlsd import lsd

from ..base_model import BaseModel


class LSD(BaseModel):
    default_conf = {
        "min_length": 15,
        "max_num_lines": None,
        "force_num_lines": False,
        "n_jobs": 4,
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        if self.conf.force_num_lines:
            assert (
                self.conf.max_num_lines is not None
            ), "Missing max_num_lines parameter"

    def detect_lines(self, img):
        # Run LSD
        segs = lsd(img)

        # Filter out keylines that do not meet the minimum length criteria
        lengths = np.linalg.norm(segs[:, 2:4] - segs[:, 0:2], axis=1)
        to_keep = lengths >= self.conf.min_length
        segs, lengths = segs[to_keep], lengths[to_keep]

        # Keep the best lines
        scores = segs[:, -1] * np.sqrt(lengths)
        segs = segs[:, :4].reshape(-1, 2, 2)
        indices = np.argsort(-scores)
        if self.conf.max_num_lines is not None:
            indices = indices[: self.conf.max_num_lines]
            segs = segs[indices]
            scores = scores[indices]

        # Pad if necessary
        n = len(segs)
        valid_mask = np.ones(n, dtype=bool)
        if self.conf.force_num_lines:
            pad = self.conf.max_num_lines - n
            segs = np.concatenate(
                [segs, np.zeros((pad, 2, 2), dtype=np.float32)], axis=0
            )
            scores = np.concatenate([scores, np.zeros(pad, dtype=np.float32)], axis=0)
            valid_mask = np.concatenate([valid_mask, np.zeros(pad, dtype=bool)], axis=0)

        return segs, scores, valid_mask

    def _forward(self, data):
        # Convert to the right data format
        image = data["image"]
        if image.shape[1] == 3:
            # Convert to grayscale
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        device = image.device
        b_size = len(image)
        image = np.uint8(image.squeeze(1).cpu().numpy() * 255)

        # LSD detection in parallel
        if b_size == 1:
            lines, line_scores, valid_lines = self.detect_lines(image[0])
            lines = [lines]
            line_scores = [line_scores]
            valid_lines = [valid_lines]
        else:
            lines, line_scores, valid_lines = zip(
                *Parallel(n_jobs=self.conf.n_jobs)(
                    delayed(self.detect_lines)(img) for img in image
                )
            )

        # Batch if possible
        if b_size == 1 or self.conf.force_num_lines:
            lines = torch.tensor(lines, dtype=torch.float, device=device)
            line_scores = torch.tensor(line_scores, dtype=torch.float, device=device)
            valid_lines = torch.tensor(valid_lines, dtype=torch.bool, device=device)

        return {"lines": lines, "line_scores": line_scores, "valid_lines": valid_lines}

    def loss(self, pred, data):
        raise NotImplementedError
