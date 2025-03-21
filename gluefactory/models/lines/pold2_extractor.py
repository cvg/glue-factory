"""
Extract lines given intermediate representations of an image in the form of keypoints, distance field and angle field.
Usage:
    python -m gluefactory.models.lines.pold2_extractor --conf gluefactory/configs/pold2_line_extractor_test.yaml --show
"""

import logging
import os
from functools import cmp_to_key

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from gluefactory.models.base_model import BaseModel
from gluefactory.models.lines.line_refinement import merge_lines_torch
from gluefactory.settings import DATA_PATH

logger = logging.getLogger(__name__)


class LineExtractor(BaseModel):

    default_conf = {
        "samples": [24],  # Number of samples to take along the line
        "max_point_size": 1500,  # Maximum number of points to consider
        "min_line_length": 6,  # Minimum line length
        "max_line_length": None,  # Maximum line length
        "max_lines": 2000,  # Maximum number of lines to consider
        "distance_map": {
            "max_value": 5,  # Maximum value to which the distance map is capped [Line Neighbourhood in Extractor Config]
            "threshold": 0.45,  # Threshold for generating binary distance map
            "smooth_threshold": 0.8,  # Threshold for smoothened distance map
            "avg_filter_size": 13,  # Size of the average filter
            "avg_filter_padding": 6,  # Padding of the average filter
            "avg_filter_stride": 1,  # Stride of the average filter
            "inlier_ratio": 0.7,  # Ratio of inliers
            "max_accepted_mean_value": 0.5,  # Maximum accepted DF mean value along the line
        },
        "brute_force_df": {
            "use": True,  # Use brute force sampling for distance field in the second stage
            "image_size": 800,  # Image size for which the coefficients are generated
            "binary_threshold": 0.3,  # Threshold for binary distance map
            "inlier_ratio": 0.8,  # Ratio of inliers
            "max_accepted_mean_value": 0.3,  # Maximum accepted DF mean value along the line
        },
        "nms": True,
        "device": None,
        "coeff_dir": "line_extraction_coeffs",
        "debug": False,
        "debug_dir": "tmp",
    }

    def _init(self, conf: OmegaConf):
        self.max_point_size = conf.max_point_size

        if conf.device is not None:
            self.device = conf.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Precompute coeffs
        self.samples = self.conf.samples
        self.coeffs_list = [
            torch.arange(0, 1, 1 / sample).to(self.device).view(-1, 1)
            for sample in self.samples
        ]
        self.coeffs_second_list = [
            (1 - torch.arange(0, 1, 1 / sample)).to(self.device).view(-1, 1)
            for sample in self.samples
        ]

        # Precompute indices
        indices = torch.combinations(torch.arange(0, self.max_point_size), r=2).numpy()

        # Sort list
        # Key corresponds to the max index in the pair
        indices = sorted(
            indices,
            key=cmp_to_key(lambda e1, e2: max(e1[0], e1[1]) - max(e2[0], e2[1])),
        )
        indices = torch.tensor(np.array(indices)).to(self.device)
        self.indices = indices

        # Brute force sampling for distance field
        self.brute_force_df = True
        image_size = conf.brute_force_df.image_size
        max_line_length = np.sqrt(2 * image_size**2).astype(int)

        self.bf_num_samples_df = max_line_length
        logger.info(
            f"Brute force sampling for distance field with {max_line_length} samples"
        )

        coeff_file_path = os.path.join(
            DATA_PATH, conf.coeff_dir, f"coeffs_df_{image_size}.npy"
        )
        coeff_second_file_path = os.path.join(
            DATA_PATH, conf.coeff_dir, f"coeffs_df_second_{image_size}.npy"
        )
        logger.info(
            f"Loading weights from {coeff_file_path} and {coeff_second_file_path}"
        )

        if os.path.exists(coeff_file_path) and os.path.exists(
            coeff_second_file_path
        ):
            self.all_coeffs_df = torch.from_numpy(np.load(coeff_file_path)).to(
                self.device
            )
            self.all_coeffs_df_second = torch.from_numpy(
                np.load(coeff_second_file_path)
            ).to(self.device)
        else:
            self.all_coeffs_df, self.all_coeffs_df_second = (
                self.precompute_brute_force_coeffs(
                    max_line_length,
                    conf.coeff_dir,
                    coeff_file_path,
                    coeff_second_file_path,
                )
            )


    def precompute_brute_force_coeffs(
        self, max_line_length, coeff_dir, coeff_file_path, coeff_second_file_path
    ):
        # Generate coefficients
        all_coeffs = torch.zeros((0, max_line_length)).to(self.device)
        all_coeffs_second = torch.zeros((0, max_line_length)).to(self.device)
        for i in range(1, max_line_length + 1):
            c1 = torch.arange(0, 1, 1 / i).to(self.device).view(-1, 1)[:i]
            c2 = (1 - torch.arange(0, 1, 1 / i)).to(self.device).view(-1, 1)[:i]

            # Pad with 0s to match the maximum line length from both sides
            pad = max_line_length - i
            left_pad = torch.zeros(pad // 2, 1).to(self.device)
            right_pad = torch.zeros(pad - pad // 2, 1).to(self.device)

            # print(f"i: {i}")
            # print(f"left_pad: {left_pad.shape}, c1: {c1.shape}, right_pad: {right_pad.shape}")
            # print(f"all_coeffs: {all_coeffs.shape}, cat: {torch.cat((left_pad, c1, right_pad), dim=0).view(1,-1).shape}")

            all_coeffs = torch.cat(
                (
                    all_coeffs,
                    torch.cat((left_pad, c1, right_pad), dim=0).view(1, -1),
                ),
                dim=0,
            )
            all_coeffs_second = torch.cat(
                (
                    all_coeffs_second,
                    torch.cat((left_pad, c2, right_pad), dim=0).view(1, -1),
                ),
                dim=0,
            )

        # Save coefficients
        if not os.path.exists(os.path.join(DATA_PATH, coeff_dir)):
            os.makedirs(os.path.join(DATA_PATH, coeff_dir))
        np.save(coeff_file_path, all_coeffs.cpu().numpy())
        np.save(coeff_second_file_path, all_coeffs_second.cpu().numpy())

        return all_coeffs, all_coeffs_second

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
        output = (distance_map & (distance_map_smooth < df_conf.smooth_threshold))[0]

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

    def get_coordinates_brute_force(self, points, indices, coeffs, coeffs_second):
        """
        Get the coordinates of the points along the line.

        Args:
            points (torch.Tensor): All keypoints, Shape: (N_pts, 2)
            indices (torch.Tensor): Indices of the candiate line endpoints, Shape: (N_lines, 2)
            coeffs (torch.Tensor): Coefficients for the first point, Shape: (N_lines, max_line_length, 1)
        """

        first_point_position = points[indices[:, 0]][
            :, None, :
        ]  # Shape: (num_lines, 1, 2)
        first_point_position = (
            coeffs * first_point_position
        )  # Shape: (num_lines, num_samples, 2)

        second_point_position = points[indices[:, 1]][
            :, None, :
        ]  # Shape: (num_lines, 1, 2)
        second_point_position = (
            coeffs_second * second_point_position
        )  # Shape: (num_lines, num_samples, 2)

        # Compute the points along the line
        # If we have 3 points for example
        # ==> p1 = 1.0 * first + 0.0 * second
        # ==> p2 = 0.5 * first + 0.5 * second
        # ==> p3 = 0.0 * first + 1.0 * second
        coords = (
            (first_point_position + second_point_position).permute(1, 0, 2).int()
        )  # Shape: (num_samples, num_lines, 2)
        return coords.reshape(-1, 2)

    def sample_map(self, points, map):
        return map[points[:, 1], points[:, 0]]

    def filter_small_lines(self, points, indices_image):
        """
        Filter out lines that are shorter than a threshold.
        """
        min_line_length = self.conf.min_line_length
        if min_line_length is None:
            return indices_image
        lines = points[indices_image]
        diff = lines[:, 0] - lines[:, 1]
        diff = torch.sum(diff**2, dim=1)

        return indices_image[diff > (min_line_length**2)]

    def filter_large_lines(self, points, indices_image):
        """
        Filter out lines that are longer than a threshold.
        """
        max_line_length = self.conf.max_line_length
        if max_line_length is None:
            return indices_image
        lines = points[indices_image]
        diff = lines[:, 0] - lines[:, 1]
        diff = torch.sum(diff**2, dim=1)

        return indices_image[diff < (max_line_length**2)]

    # Distance map filtering
    def filter_with_distance_field(
        self,
        points: torch.Tensor,
        binary_distance_map: torch.Tensor,
        distance_map: torch.Tensor,
        indices: torch.Tensor,
        sample_idx: int,
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
            sample_idx (int): index of the samples to use, i.e. how many samples to take along the line
        Return:
            Indices of valid line candidates
        """

        num_sample = self.samples[sample_idx]
        inlier_ratio = self.conf.distance_map.inlier_ratio
        max_accepted_mean_value = self.conf.distance_map.max_accepted_mean_value

        # Get coordinates
        points_coordinates = self.get_coordinates(
            points,
            indices,
            self.coeffs_list[sample_idx],
            self.coeffs_second_list[sample_idx],
        )

        # Sample points
        binary_df_sampled = self.sample_map(
            points_coordinates, binary_distance_map
        ).view(num_sample, -1)
        df_sampled = self.sample_map(points_coordinates, distance_map).view(
            num_sample, -1
        )

        # We reduce the values along the line
        detected_line_indices = (
            torch.sum(binary_df_sampled, dim=0) >= num_sample * inlier_ratio
        )
        detected_line_float = (
            torch.mean(df_sampled.float(), dim=0) <= max_accepted_mean_value
        )

        if self.conf.debug:
            print(f"=============== Distance Filter =================")
            print(f"Decision pos lines binary df: {torch.sum(detected_line_indices)}")
            print(f"Decision pos lines df: {torch.sum(detected_line_float)}")
            print(
                f"Decision overall num lines: {torch.sum(detected_line_indices & detected_line_float)}"
            )
            print(f"===============================================")

        # Finally we can filter the indices
        return indices[detected_line_indices & detected_line_float]

    def brute_force_filter_with_distance_field(
        self,
        points: torch.Tensor,
        binary_distance_map: torch.Tensor,
        distance_map: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:

        # Prepare the coefficients for brute force sampling

        # get distances between candidate end points
        distances = torch.norm(
            (points[indices[:, 0]] - points[indices[:, 1]]).float(), dim=1
        )
        distances = distances.int() - 1

        coeffs_df = self.all_coeffs_df[distances][
            :, :, None
        ]  # Shape: (num_lines, max_line_length, 1)
        coeffs_df_second = self.all_coeffs_df_second[distances][
            :, :, None
        ]  # Shape: (num_lines, max_line_length, 1)

        # Padded coeffs will sample the DF
        # Hardcode DF[0,0]
        distance_map[0, 0] = -1

        # Sample coordinates (sample line points for each pair of kp representing a line candidate)
        points_coordinates = self.get_coordinates_brute_force(
            points, indices, coeffs_df, coeffs_df_second
        ).reshape(self.bf_num_samples_df, len(indices), 2)

        # Shape: (num_lines, bf_num_samples_df, 2)
        points_coordinates = points_coordinates.reshape(
            -1, 2
        )  # Shape: (num_lines * bf_num_samples_df, 2)

        binary_distance_map = (
            distance_map < self.conf.brute_force_df.binary_threshold
        ).float()

        # Sample points
        binary_df_sampled = self.sample_map(
            points_coordinates, binary_distance_map
        ).view(self.bf_num_samples_df, -1)
        df_sampled = self.sample_map(points_coordinates, distance_map).view(
            self.bf_num_samples_df, -1
        )

        # We reduce the values along the line
        num_valid = torch.sum(df_sampled >= 0, dim=0)
        binary_df_sampled[df_sampled < 0] = 0
        df_sampled[df_sampled < 0] = 0

        detected_line_indices = (
            torch.sum(binary_df_sampled, dim=0)
            >= num_valid * self.conf.brute_force_df.inlier_ratio
        )

        if self.conf.debug:
            print(f"=============== Distance Filter BRUTE FORCE =================")
            print(f"Decision pos lines binary df: {torch.sum(detected_line_indices)}")
            print(f"=============================================================")

        # Finally we can filter the indices
        return indices[detected_line_indices]

    # Post processing step
    def multi_stage_filter(
        self, points, binary_distance_map, distance_map, indices_image
    ):

        # Filter out small lines
        indices_image = self.filter_small_lines(points, indices_image)
        indices_image = self.filter_large_lines(points, indices_image)

        for i in range(len(self.samples)):

            # Weak filter - Handcrafted heuristic on distance field
            indices_image = self.filter_with_distance_field(
                points, binary_distance_map, distance_map, indices_image, i
            )
            if self.conf.debug:
                print(
                    f"Num lines after stage {i} [Distance Filtering]: {indices_image.shape[0]}"
                )

        # Strong filter - brute force sampling for distance field
        indices_image = self.brute_force_filter_with_distance_field(
            points, binary_distance_map, distance_map, indices_image
        )
        if self.conf.debug:
            print(f"Num lines after brute force DF stage: {indices_image.shape[0]}")        

        return indices_image

    def _forward(self, data: dict) -> torch.Tensor:
        points = data["points"]
        distance_map = data["distance_map"]
        descriptors = data["descriptors"]

        # Get indices
        if len(points) > self.max_point_size:
            logger.warning(
                f"WARNING: We have more than {self.max_point_size} points in this image ({len(points)}), keeping only {self.max_point_size} firsts"
            )
            points = points[: self.max_point_size]

        # Precompute indices
        number_pairs = int(len(points) * (len(points) - 1) / 2)
        indices_image = self.indices[:number_pairs]

        # normalize DF to [0, 1] by dividing by max value TODO: strict positive Z-Score normalization ?
        df_max = self.conf.distance_map.max_value
        distance_map = distance_map.float()
        distance_map /= df_max

        # Process distance map
        binary_distance_map = self.process_distance_map(distance_map)


        # Apply two stage filter
        filtered_idx = self.multi_stage_filter(
            points, binary_distance_map, distance_map, indices_image
        )

        # Apply NMS
        if self.conf.nms:
            # Get line end points
            lines = points[filtered_idx]

            # Append indices to lines - (N,2,3)
            # Lines are now in the form (x1, y1, idx1), (x2, y2, idx2), where idx1 and idx2 are the indices of the keypoints
            # and (x1, y1) and (x2, y2) are the coordinates of the keypoints
            lines = torch.cat(
                [lines.reshape(-1, 2), filtered_idx.reshape(-1, 1)], axis=-1
            ).reshape(-1, 2, 3)
            filtered_idx = merge_lines_torch(lines, return_indices=True).int()

            if self.conf.debug:
                print(f"Number of lines after NMS: {len(filtered_idx)}")

        # Prepare output
        if len(filtered_idx) == 0:
            lines = torch.zeros(0, 2, 2).to(self.device)
            line_descriptors = torch.zeros(0, 2, 128).to(self.device)
        else:
            lines = points[filtered_idx]
            line_descriptors = torch.stack(
                [descriptors[filtered_idx[:, 0]], descriptors[filtered_idx[:, 1]]],
                dim=1,
            )

        # sort lines by length and select the top k (max_lines)
        line_lengths = torch.sum((lines[:, 0] - lines[:, 1]) ** 2, dim=1)
        _, sorted_idx = torch.sort(line_lengths, descending=True)
        sorted_idx = sorted_idx[: self.conf.max_lines]

        lines = lines[sorted_idx]
        line_descriptors = line_descriptors[sorted_idx]
        filtered_idx = filtered_idx[sorted_idx]

        return {
            "lines": lines,
            "line_descriptors": line_descriptors,
            "line_endpoint_indices": filtered_idx,
        }

    def loss(self, pred, data):
        raise NotImplementedError
