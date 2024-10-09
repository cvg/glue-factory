"""
Extract lines given intermediate representations of an image in the form of keypoints, distance field and angle field.
Usage:
    python -m gluefactory.models.lines.pold2_extractor --conf gluefactory/configs/pold2_line_extractor_test.yaml --show
"""

import argparse
import glob
import logging
import os
import pickle
import time
from functools import cmp_to_key

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from gluefactory.models.base_model import BaseModel
from gluefactory.models.lines.line_refinement import merge_lines_torch
from gluefactory.models.lines.pold2_mlp import POLD2_MLP
from gluefactory.settings import DATA_PATH, root

logger = logging.getLogger(__name__)


class LineExtractor(BaseModel):

    default_conf = {
        "num_sample": 8,
        "max_point_size": 1500,
        "min_line_length": 20,
        "max_line_length": 200,
        "distance_map": {
            "threshold": 0.5,
            "avg_filter_size": 13,
            "avg_filter_padding": 6,
            "avg_filter_stride": 1,
            "inlier_ratio": 1.0,
            "max_accepted_mean_value": 0.8,
        },
        "mlp_conf": POLD2_MLP.default_conf,
        "nms": True,
        "device": None,
        "debug": False,
    }

    def _init(self, conf: dict):

        self.num_sample = conf.num_sample
        self.num_sample_strong = conf.mlp_conf.num_line_samples
        self.max_point_size = conf.max_point_size

        if conf.device is not None:
            self.device = conf.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Precompute coeffs
        coeffs = torch.arange(0, 1, 1 / self.num_sample).to(self.device).view(-1, 1)
        coeffs_second = (
            (1 - torch.arange(0, 1, 1 / self.num_sample)).to(self.device).view(-1, 1)
        )

        coeffs_strong = (
            torch.arange(0, 1, 1 / self.num_sample_strong).to(self.device).view(-1, 1)
        )
        coeffs_strong_second = (
            (1 - torch.arange(0, 1, 1 / self.num_sample_strong))
            .to(self.device)
            .view(-1, 1)
        )

        # Precompute indices
        indices = torch.combinations(torch.arange(0, self.max_point_size), r=2).numpy()

        # Sort list
        # Key corresponds to the max index in the pair
        indices = sorted(
            indices,
            key=cmp_to_key(lambda e1, e2: max(e1[0], e1[1]) - max(e2[0], e2[1])),
        )
        indices = torch.tensor(np.array(indices)).to(self.device)

        self.coeffs = coeffs
        self.coeffs_second = coeffs_second
        self.coeffs_strong = coeffs_strong
        self.coeffs_strong_second = coeffs_strong_second
        self.indices = indices

        # Import mlp
        self.model = POLD2_MLP(conf.mlp_conf)
        self.model.to(self.device)
        self.model.eval()

    # Distance map processing
    def process_distance_map(self, distance_map):
        df_conf = self.conf.distance_map
        distance_map = distance_map < df_conf.threshold

        average_filter = nn.AvgPool2d(
            kernel_size=df_conf.avg_filter_size,
            stride=df_conf.avg_filter_stride,
            padding=df_conf.avg_filter_padding,
        )

        distance_map_smooth = average_filter(distance_map[None, :, :].float())
        output = (distance_map & (distance_map_smooth < 1))[0]

        return output

    def get_coordinates(self, points, indices, coeffs, coeffs_second):
        first_point_position = points[indices[:, 0]].view(-1)
        first_point_position = coeffs * first_point_position

        second_point_position = points[indices[:, 1]].view(-1)
        second_point_position = coeffs_second * second_point_position

        # Compute the points along the line
        # If we have 3 points for example
        # ==> p1 = 1.0 * first + 0.0 * second
        # ==> p2 = 0.5 * first + 0.5 * second
        # ==> p3 = 0.0 * first + 1.0 * second
        return (first_point_position + second_point_position).view(-1, 2).int()

    def sample_map(self, points, map):
        return map[points[:, 1], points[:, 0]]

    def filter_small_lines(self, points, indices_image):
        """
        Filter out lines that are too short.
        """
        min_line_length = self.conf.min_line_length
        lines = points[indices_image]
        diff = lines[:, 0] - lines[:, 1]
        diff = torch.sum(diff ** 2, dim=1)

        return indices_image[diff > (min_line_length**2)]

    def filter_large_lines(self, points, indices_image):
        """
        Filter out lines that are too short.
        """
        max_line_length = self.conf.max_line_length
        lines = points[indices_image]
        diff = lines[:, 0] - lines[:, 1]
        diff = torch.sum(diff ** 2, dim=1)

        return indices_image[diff < (max_line_length**2)]

    # Distance map filtering
    def filter_with_distance_field(
        self, points: torch.Tensor, binary_distance_map: torch.Tensor, distance_map: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Heuristic filtering line candidates given each as 2 line endpoints, by sampling equidistant points
        inbetween line endpoints. It then checks:
            1. Is the mean DF value of samples for a line below a certain thresh
            2. How many df values of samples for a line are below a threshold (told by binary_distance_map at these points)

        If both conditions are True for a line candidate it is considered a valid line by this filter.
        
        Args:
            points (torch.Tensor): all keypoints
            binary_distance_map (torch.Tensor): binary map, telling whether df values at pos are smaller than a threshold
            distance_map (torch.Tensor): distance field as in DeepLSD, normalized to [0, 1]
            indices (torch.Tensor): indices of line endpoints
        Return:
            Indices of
        """
        inlier_ratio = self.conf.distance_map.inlier_ratio
        max_accepted_mean_value = self.conf.distance_map.max_accepted_mean_value

        # Get coordinates
        points_coordinates = self.get_coordinates(
            points, indices, self.coeffs, self.coeffs_second
        )

        # Sample points
        binary_df_sampled = self.sample_map(
            points_coordinates, binary_distance_map
        ).view(self.num_sample, -1)
        df_sampled = self.sample_map(points_coordinates, distance_map).view(
            self.num_sample, -1
        )

        # We reduce the values along the line
        detected_line_indices = (
            torch.sum(binary_df_sampled, dim=0) >= self.num_sample * inlier_ratio
        )
        detected_line_float = torch.mean(df_sampled.float(), dim=0) <= max_accepted_mean_value

        if self.conf.debug:
            print(f"Decision pos lines binary df: {torch.sum(detected_line_indices)}")
            print(f"Decision pos lines df: {torch.sum(detected_line_float)}")
            print(f"Decision overall num lines: {torch.sum(detected_line_indices & detected_line_float)}")
        # Finally we can filter the indices
        return indices[detected_line_indices & detected_line_float]


    # MLP filter
    def mlp_filter(self, points: torch.Tensor, indices_image: torch.Tensor, distance_map: torch.Tensor, angle_map: torch.Tensor) -> torch.Tensor:
        """
        Uses a small Fully Connected NN (MLP) to predict the probabilities of all given line candidates to really be a line!
        This probability is then used to filter out line candidates whose predicted probability is below a certain threshhold.

        Args:
            points (torch.Tensor): all keypoints, Shape: (#keypoints , 2)
            indices_image (torch.Tensor): Each row contains indices of 2 keypoints (indexing keypoint list "points"). Thus each row = 1 line candidate. Shape: (#candidate_lines , 2)
            distance_map (torch.Tensor): distance field as in DeepLSD
            angle_map (torch.Tensor): angle field as in DeepLSD

        Returns:
            torch.Tensor: _description_
        """
        use_df = self.conf.mlp_conf.has_distance_field
        use_af = self.conf.mlp_conf.has_angle_field

        # Sample coordinates (sample line points for each pair of kp representing a line candidate)
        points_coordinates = torch.zeros(0, 2).int().to(self.device)

        num_bands = self.conf.mlp_conf.num_bands
        band_width = self.conf.mlp_conf.band_width
        band_ids = np.arange(0, num_bands) - num_bands // 2
        band_ids *= band_width
        for i in band_ids:
            band_pts = points + i
            band_pts = band_pts.int()

            cur_coords = self.get_coordinates(
                band_pts, indices_image, self.coeffs_strong, self.coeffs_strong_second
            )
            cur_coords[:, 0] = torch.clamp(cur_coords[:, 0], 0, distance_map.shape[1] - 1)
            cur_coords[:, 1] = torch.clamp(cur_coords[:, 1], 0, distance_map.shape[0] - 1)
            points_coordinates = torch.cat((points_coordinates, cur_coords), dim=0)
        # Sample points
        if use_df:
            df_vals = self.sample_map(points_coordinates, distance_map).view(self.num_sample_strong*num_bands, -1)
            
        if use_af:
            af_vals = self.sample_map(points_coordinates, angle_map).view(self.num_sample_strong*num_bands, -1)
            
        # Prepare input for MLP
        if use_df and use_af:
            inp_vals = torch.cat((df_vals, af_vals), dim=0)
        elif use_df:
            inp_vals = df_vals
        elif use_af:
            inp_vals = af_vals

        # Return estimated probabilities
        predictions = self.model(
            {
                "input": torch.swapaxes(inp_vals, 0, 1),
            }
        )["line_probs"]
        mlp_output = (predictions > self.conf.mlp_conf.pred_threshold).reshape(-1)

        # Filter lines based on MLP output
        return indices_image[mlp_output]

    # Post processing step
    def two_stage_filter(
        self, points, binary_distance_map, distance_map, angle_map, indices_image
    ):

        # Filter out small lines
        indices_image = self.filter_small_lines(points, indices_image)
        indices_image = self.filter_large_lines(points, indices_image)

        # First pass - weak filter - Handcrafted heuristic
        indices_image = self.filter_with_distance_field(
            points, binary_distance_map, distance_map, indices_image
        )
        if self.conf.debug:
            print(f"Num lines after 1st stage: {indices_image.shape[0]}")

        # Second pass - strong filer - MLP filter
        indices_image = self.mlp_filter(points, indices_image, distance_map, angle_map)

        if self.conf.debug:
            print(f"Num lines after MLP stage: {indices_image.shape[0]}")
        return indices_image

    def _forward(self, data: dict) -> torch.Tensor:
        points = data["points"]
        distance_map = data["distance_map"]
        angle_map = data["angle_map"]
        descriptors = data["descriptors"]

        # Convert angle map (direction vector) to angle (radians from 0 to pi) but only if loading ground truth!
        if angle_map.ndim > 2:
            angle_map = torch.atan2(angle_map[1], angle_map[0]) % torch.pi

        # Get indices
        if len(points) > self.max_point_size:
            logger.warning(
                f"WARNING: We have more than {self.max_point_size} points in this image ({len(points)}), keeping only {self.max_point_size} firsts"
            )
            points = points[: self.max_point_size]

        # Precompute indices
        number_pairs = int(len(points) * (len(points) - 1) / 2)
        indices_image = self.indices[:number_pairs]

        # normalize to [0, 1] by dividing by max value TODO: strict positive Z-Score normalization ?
        df_max = distance_map.max()
        distance_map = distance_map.float()
        distance_map /= df_max

        # Process distance map
        binary_distance_map = self.process_distance_map(distance_map)

        # Apply two stage filter
        filtered_idx = self.two_stage_filter(points, binary_distance_map, distance_map, angle_map, indices_image)

        # Apply NMS
        if self.conf.nms:
            # Get line end points
            lines = points[filtered_idx]

            # Append indices to lines - (N,2,3)
            # Lines are now in the form (x1, y1, idx1), (x2, y2, idx2), where idx1 and idx2 are the indices of the keypoints
            # and (x1, y1) and (x2, y2) are the coordinates of the keypoints
            lines = torch.cat([lines.reshape(-1,2), filtered_idx.reshape(-1,1)], axis=-1).reshape(-1, 2, 3)
            filtered_idx = merge_lines_torch(torch.tensor(lines), return_indices=True).int()

        # Prepare output
        lines = points[filtered_idx]
        line_descriptors = torch.stack(
            [descriptors[filtered_idx[:, 0]], descriptors[filtered_idx[:, 1]]], dim=1
        )

        return {
            "lines": lines,
            "line_descriptors": line_descriptors,
            "line_endpoint_indices": filtered_idx,
        }

    def loss(self, pred, data):
        raise NotImplementedError


def show_points(image, points):
    for point in points:
        cv2.circle(image, (point[0], point[1]), 4, (191, 69, 17), -1)

    return image


def show_lines(image, lines):
    for pair_line in lines:
        cv2.line(image, pair_line[0], pair_line[1], (255, 255, 0), 3)

    return image


def test_extractor(extractor, folder_path, device, show=False):
    image = torch.from_numpy(np.array(Image.open(f"{folder_path}/base_image.jpg"))).to(
        device
    )
    distance_map = torch.from_numpy(np.array(Image.open(f"{folder_path}/df.jpg"))).to(
        device
    )
    angle_map = torch.from_numpy(np.array(Image.open(f"{folder_path}/angle.jpg"))).to(
        device
    )

    # Prepare distance map
    distance_map = distance_map.float() / 255
    with open(f"{folder_path}/values.pkl", "rb") as f:
        values = pickle.load(f)
        distance_map = distance_map * values["max_df"]

    # Normalize angle map
    angle_map = angle_map.float() / 255 * np.pi
    angle_map = torch.cat(
        (torch.cos(angle_map).unsqueeze(2), torch.sin(angle_map).unsqueeze(2)), dim=2
    ).permute(2, 0, 1)

    # Load keypoints
    points_np = np.load(f"{folder_path}/keypoints.npy", allow_pickle=True)
    points = torch.from_numpy(points_np).to(device).int()

    # Start counter
    start_time = time.perf_counter()

    data = {
        "points": points,
        "distance_map": distance_map,
        "angle_map": angle_map,
        "descriptors": torch.zeros(len(points), 128).to(device)
    }
    # Post processing step
    indices_image = extractor(data)["line_endpoint_indices"]

    if device == "cuda":
        torch.cuda.synchronize()

    # End counter
    end_time = time.perf_counter()

    image = image.cpu().detach().numpy()
    points = points.cpu().numpy()
    indices_image = indices_image.cpu().numpy()

    # Samples lines from indices
    lines = points[indices_image]

    # Show lines and points
    o_img = show_lines(image, lines[:, :, :2])
    o_img = show_points(o_img, points)

    folder_path = folder_path.split("/")[-1]

    # logger.info(f"Elapsed time for {folder_path} is : {end_time - start_time}")

    plt.imshow(o_img)
    if show:
        plt.show()

    plt.title("Before NMS")
    plt.axis("off")
    plt.savefig(
        f"tmp/{folder_path}_orig.jpg", dpi=300, bbox_inches="tight", pad_inches=0
    )
    plt.close()

    # Merge lines
    # Append indices to lines - (N,2,3)
    lines = np.concatenate(
        [lines.reshape(-1, 2), indices_image.reshape(-1, 1)], axis=-1
    ).reshape(-1, 2, 3)
    indices_nms = (
        merge_lines_torch(torch.tensor(lines), return_indices=True).int().cpu().numpy()
    )
    n_lines = points[indices_nms]
    n_img = show_lines(image, n_lines)
    n_img = show_points(n_img, points)

    plt.imshow(n_img)
    plt.title("After NMS")
    plt.axis("off")
    plt.savefig(
        f"tmp/{folder_path}_nms.jpg", dpi=300, bbox_inches="tight", pad_inches=0
    )
    plt.close()

    # Merge lines without indices
    onms_lines = (
        merge_lines_torch(torch.tensor(lines[:, :, :2]), return_indices=False)
        .int()
        .cpu()
        .numpy()
    )
    print(f"Number of lines after orig NMS: {len(onms_lines)}")

    onms_img = show_lines(image, onms_lines)
    onms_img = show_points(onms_img, points)

    plt.imshow(onms_img)
    plt.title("After orig NMS")
    plt.axis("off")
    plt.savefig(
        f"tmp/{folder_path}_orig_nms.jpg", dpi=300, bbox_inches="tight", pad_inches=0
    )
    plt.close()


if __name__ == "__main__":
    from ... import logger

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--conf", type=str, default=None)
    argParser.add_argument("--show", action="store_true")
    args = argParser.parse_args()

    extractor_conf = (
        OmegaConf.load(args.conf)
        if args.conf is not None
        else LineExtractor.default_conf
    )
    extractor = LineExtractor(extractor_conf)

    if os.path.exists("tmp"):
        os.system("rm -r tmp")
    os.makedirs("tmp", exist_ok=True)

    for val in tqdm(
        glob.glob(
            str(DATA_PATH / "revisitop1m_POLD2/**/base_image.jpg"), recursive=True
        )
    ):
        test_extractor(
            extractor, os.path.split(val)[0], extractor_conf.device, args.show
        )
