"""
Reads superpoint heatmap stored in hdf5 files for all files in a given folder (including subfolders)
and extracts keypoints. Savers them in a numpy file.
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm import tqdm

from gluefactory.models.extractors.aliked import DKD


class KPExtractor:
    @staticmethod
    def simple_nms(scores, radius):
        """Perform non maximum suppression on the heatmap using max-pooling.
        This method does not suppress contiguous points that have the same score.
        Args:
            scores: the score heatmap of size `(B, H, W)`.
            size: an interger scalar, the radius of the NMS window.
        """

        def max_pool(x):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=radius * 2 + 1, stride=1, padding=radius
            )

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    @staticmethod
    def remove_borders(keypoints, b, h, w):
        mask_h = (keypoints[1] >= b) & (keypoints[1] < (h - b))
        mask_w = (keypoints[2] >= b) & (keypoints[2] < (w - b))
        mask = mask_h & mask_w
        return (keypoints[0][mask], keypoints[1][mask], keypoints[2][mask])

    @staticmethod
    def soft_argmax_refinement(keypoints, scores, radius: int):
        width = 2 * radius + 1
        sum_ = torch.nn.functional.avg_pool2d(
            scores[:, None], width, 1, radius, divisor_override=1
        )
        sum_ = torch.clamp(sum_, min=1e-6)
        ar = torch.arange(-radius, radius + 1).to(scores)
        kernel_x = ar[None].expand(width, -1)[None, None]
        dx = torch.nn.functional.conv2d(scores[:, None], kernel_x, padding=radius)
        dy = torch.nn.functional.conv2d(
            scores[:, None], kernel_x.transpose(2, 3), padding=radius
        )
        dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
        refined_keypoints = []
        for i, kpts in enumerate(keypoints):
            delta = dydx[i][tuple(kpts.t())]
            refined_keypoints.append(kpts.float() + delta)
        return refined_keypoints

    def __init__(self, config):
        self.threshold_value = config.get('threshold_value', 0.045)
        self.max_num_keypoints = config.get('max_keypoints', 5000)
        self.nms_radius = config.get('nms_radius', 4)
        self.remove_borders_conf = config.get('remove_borders', 4)
        self.refinement_radius = config.get('refinement_radius', 0)
        self.dkd_extractor = DKD(radius=self.nms_radius, top_k=0, scores_th=self.threshold_value, n_limit=self.max_num_keypoints)

    def sort_keypoints(self, keypoints, scores):
        """
        Returns a sorted list of all keypoints.
        """
        sorted_indices = torch.argsort(scores, dim=0, descending=True, stable=True)
        sorted_keypoints = keypoints[sorted_indices]
        sorted_scores = scores[sorted_indices]
        return sorted_keypoints, sorted_scores

    def extract_keypoints(self, sp_heatmap, b_size=1):
        """Extract keypoints as implemented in Superpoint"""
        h, w = sp_heatmap.shape[1] // 8, sp_heatmap.shape[2] // 8

        scores = self.simple_nms(sp_heatmap, self.nms_radius)

        # Extract keypoints
        best_kp = torch.where(scores > self.threshold_value)

        # Discard keypoints near the image borders
        best_kp = self.remove_borders(best_kp, self.remove_borders_conf, h * 8, w * 8)
        scores = scores[best_kp]

        # Separate into batches
        keypoints = [
            torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i]
            for i in range(b_size)
        ]
        scores = [scores[best_kp[0] == i] for i in range(b_size)]

        # Keep the k keypoints with highest score
        keypoints = keypoints[0]
        scores = scores[0]
        keypoints, scores = self.sort_keypoints(keypoints, scores)

        if self.refinement_radius > 0:
            keypoints = self.soft_argmax_refinement(
                keypoints, sp_heatmap, self.refinement_radius
            )
        assert scores.shape[0] > 5
        return keypoints, scores

    def extract_keypoints_dkd(self, sp_heatmap, device="cpu"):
        sp_heatmap = sp_heatmap.to(device)
        keypoints, _, scores = self.dkd_extractor(sp_heatmap.unsqueeze(0), sub_pixel=True)
        # sort
        _, h, w = sp_heatmap.shape
        wh = torch.tensor([w, h], device=sp_heatmap.device)

        rescaled_kp = wh * (torch.stack(keypoints) + 1.0) / 2.0
        keypoints, scores = rescaled_kp.squeeze(0), scores[0]
        keypoints, scores = self.sort_keypoints(keypoints, scores)
        return keypoints, scores

def read_datasets_from_h5(keys: list, file) -> dict:
    """
    Read datasets from h5 file. Expects np arrays in h5py files.
    """
    data = {}
    for key in keys:
        data[key] = torch.from_numpy(
            np.nan_to_num(file[key].__array__())
        )  # nan_to_num needed because of weird sp gt format
    return data

def extract_keypoints(hdf5_path, extractor, debug: bool = False, device="cpu"):
    with h5py.File(hdf5_path, "r") as hmap_file:
        sp_gt_heatmap = read_datasets_from_h5(
            ["superpoint_heatmap"], hmap_file  # loaded tensor has shape 2 x N
        )["superpoint_heatmap"].unsqueeze(0)
    keypoints, scores = extractor.extract_keypoints_dkd(sp_gt_heatmap, device)
    if debug:
        print("Heatmap_shape:", sp_gt_heatmap.shape)
        print("Scores_shape:", scores.shape)
        print("Keypoints shape:", keypoints.shape)
        print("Max/Min y: ", keypoints[:, 0].max(), keypoints[:, 0].min())
        print("Max/Min x: ", keypoints[:, 1].max(), keypoints[:, 1].min())
    # concat keypoints and scores
    keypoints = keypoints.cpu()
    scores = scores.cpu()
    to_store = np.concatenate((keypoints, scores.unsqueeze(1)), axis=1)
    if debug:
        print("to_store.shape:", to_store.shape)
    kp_store_path = hdf5_path.parent / f"{hdf5_path.stem}.npy"
    np.save(kp_store_path, to_store)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("path", type=str)
    arg_parser.add_argument("--debug", action="store_true")
    arg_parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    arguments = arg_parser.parse_args()

    input_path = Path(arguments.path)
    assert Path(arguments.path).is_dir()

    print("Start Extracting Keypoints from ", arguments.path)

    heatmap_files = input_path.glob("**/*.hdf5")
    extractor = KPExtractor({})

    for f in tqdm(heatmap_files):
        extract_keypoints(f, extractor, bool(arguments.debug), arguments.device)

    print("Finish Extracting Keypoints!")