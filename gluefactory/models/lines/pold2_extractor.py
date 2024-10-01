"""
Extract lines given intermediate representations of an image in the form of keypoints, distance field and angle field.
Usage:
    python -m gluefactory.models.lines.pold2_extractor --conf gluefactory/configs/pold2_line_extractor_test.yaml --show
"""

import time
import pickle
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import os
import glob
import argparse
from functools import cmp_to_key
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.settings import root
from gluefactory.models.lines.pold2_mlp import POLD2_MLP
from gluefactory.models.base_model import BaseModel
from gluefactory.settings import DATA_PATH
from gluefactory.models.lines.line_refinement import merge_lines_torch

logger = logging.getLogger(__name__)

class LineExtractor(BaseModel):

    default_conf = {
        "num_sample": 8,
        "max_point_size": 1500,

        "distance_map": {
            "threshold": 0.5,
            "avg_filter_size": 13,
            "avg_filter_padding": 6,
            "avg_filter_stride": 1,
            "max_value": 2,
            "inlier_ratio": 1.0,
            "mean_value_ratio": 0.8
        },

        "mlp_conf": POLD2_MLP.default_conf,
        "device": None
    }

    def _init(self, conf):

        self.num_sample = conf.num_sample
        self.num_sample_strong = conf.mlp_conf.num_line_samples
        self.max_point_size = conf.max_point_size

        if conf.device is not None:
            self.device = conf.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Precompute coeffs
        coeffs = torch.arange(0, 1, 1 / self.num_sample).to(self.device).view(-1, 1)
        coeffs_second = (1 - torch.arange(0, 1, 1 / self.num_sample)).to(self.device).view(-1, 1)

        coeffs_strong = torch.arange(0, 1, 1 / self.num_sample_strong).to(self.device).view(-1, 1)
        coeffs_strong_second = (1 - torch.arange(0, 1, 1 / self.num_sample_strong)).to(self.device).view(-1, 1)

        # Precompute indices
        indices = torch.combinations(torch.arange(0, self.max_point_size), r=2).numpy()

        # Sort list
        # Key corresponds to the max index in the pair
        indices = sorted(indices, key=cmp_to_key(lambda e1, e2: max(e1[0], e1[1]) - max(e2[0], e2[1])))
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
        distance_map = (distance_map < df_conf.threshold)

        average_filter = nn.AvgPool2d(
            kernel_size=df_conf.avg_filter_size,
            stride=df_conf.avg_filter_stride,
            padding=df_conf.avg_filter_padding
        )

        distance_map_smooth = average_filter(distance_map[None, :, :].float())
        output = (distance_map & (distance_map_smooth < 1))[0]

        return output

    def get_coordinates(self, points, indices, coeffs, coeffs_second):
        first_point_position = points[indices[:, 0]].view(-1)
        first_point_position = (coeffs * first_point_position)

        second_point_position = points[indices[:, 1]].view(-1)
        second_point_position = (coeffs_second * second_point_position)

        # Compute the points along the line
        # If we have 3 points for example
        # ==> p1 = 1.0 * first + 0.0 * second
        # ==> p2 = 0.5 * first + 0.5 * second
        # ==> p3 = 0.0 * first + 1.0 * second
        return (first_point_position + second_point_position).view(-1, 2).int()

    def sample_map(self, points, map):
        return map[points[:, 1], points[:, 0]]

    # Distance map filtering
    def filter_with_distance_field(self, points, binary_distance_map, distance_map, indices):
        inlier_ratio = self.conf.distance_map.inlier_ratio
        mean_value_ratio = self.conf.distance_map.mean_value_ratio

        # Get coordinates
        points_coordinates = self.get_coordinates(points, indices, self.coeffs, self.coeffs_second)

        # Sample points
        binary_df_sampled = self.sample_map(points_coordinates, binary_distance_map).view(self.num_sample, -1)
        df_sampled = self.sample_map(points_coordinates, distance_map).view(self.num_sample, -1)

        # We reduce the values along the line
        detected_line_indices = (torch.sum(binary_df_sampled, dim=0) >= self.num_sample * inlier_ratio)
        detected_line_float = (torch.mean(df_sampled.float(), dim=0) <= mean_value_ratio)

        # Finally we can filter the indices
        return indices[detected_line_indices & detected_line_float]

    # MLP filter
    def mlp_filter(self, points, indices_image, distance_map, angle_map):
        use_df = self.conf.mlp_conf.has_distance_field
        use_af = self.conf.mlp_conf.has_angle_field

        # Sample coordinates
        points_coordinates = self.get_coordinates(points, indices_image, self.coeffs_strong, self.coeffs_strong_second)

        # Sample points
        if use_df:
            df_vals = self.sample_map(points_coordinates, distance_map).view(self.num_sample_strong, -1)
        if use_af:
            af_vals = self.sample_map(points_coordinates, angle_map).view(self.num_sample_strong, -1)

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
    def three_stage_filter(self, points, binary_distance_map, distance_map, angle_map, indices_image):

        # First pass - weak filter - Handcrafted heuristic
        indices_image = self.filter_with_distance_field(
            points, binary_distance_map, distance_map, indices_image
        )

        # Second pass - strong filer - MLP filter
        indices_image = self.mlp_filter(points, indices_image, distance_map, angle_map)

        return indices_image

    # TODO: Add merging of lines
    def _forward(self, data):
        points = data['points']
        distance_map = data['distance_map']
        angle_map = data['angle_map']

        # Convert angle map (direction vector) to angle (radians from 0 to pi)
        angle_map = torch.atan2(angle_map[1], angle_map[0]) % torch.pi

        # Get indices
        if len(points) > self.max_point_size:
            logger.warning(
                f'WARNING: We have more than {self.max_point_size} points in this image ({len(points)}), keeping only {self.max_point_size} firsts')
            points = points[:self.max_point_size]
        
        # Precompute indices
        number_pairs = int(len(points) * (len(points) - 1) / 2)
        indices_image = self.indices[:number_pairs]
        
        df_max = self.conf.distance_map.max_value
        distance_map[distance_map > df_max] = df_max

        distance_map = distance_map.float()
        distance_map /= df_max

        # Process distance map
        binary_distance_map = self.process_distance_map(distance_map)

        # Apply two stage filter
        return self.three_stage_filter(points, binary_distance_map, distance_map, angle_map, indices_image)

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
    image = torch.from_numpy(
        np.array(Image.open(f'{folder_path}/base_image.jpg'))).to(device)
    distance_map = torch.from_numpy(
        np.array(Image.open(f'{folder_path}/df.jpg'))).to(device)
    angle_map = torch.from_numpy(
        np.array(Image.open(f'{folder_path}/angle.jpg'))).to(device)

    # Prepare distance map
    distance_map = distance_map.float() / 255
    with open(f'{folder_path}/values.pkl', 'rb') as f:
        values = pickle.load(f)
        distance_map = distance_map * values['max_df']

    # Normalize angle map
    angle_map = angle_map.float() / 255 * np.pi
    angle_map = torch.cat((torch.cos(angle_map).unsqueeze(2), torch.sin(angle_map).unsqueeze(2)), dim=2).permute(2, 0, 1)

    # Load keypoints
    points_np = np.load(f'{folder_path}/keypoints.npy', allow_pickle=True)
    points = torch.from_numpy(points_np).to(device).int()

    # Start counter
    start_time = time.perf_counter()

    data = {
        "points": points,
        "distance_map": distance_map,
        "angle_map": angle_map
    }
    # Post processing step
    indices_image = extractor(data)

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

    folder_path = folder_path.split('/')[-1]

    # logger.info(f"Elapsed time for {folder_path} is : {end_time - start_time}")

    plt.imshow(o_img)
    if show:
        plt.show()

    plt.title("Before NMS")
    plt.axis('off')
    plt.savefig(f'tmp/{folder_path}_orig.jpg', dpi=300,bbox_inches='tight', pad_inches=0)
    plt.close()

    # Merge lines
    # Append indices to lines - (N,2,3)
    lines = np.concatenate([lines.reshape(-1,2), indices_image.reshape(-1,1)], axis=-1).reshape(-1, 2, 3)
    indices_nms = merge_lines_torch(torch.tensor(lines), return_indices=True).int().cpu().numpy()
    n_lines = points[indices_nms]
    n_img = show_lines(image, n_lines)
    n_img = show_points(n_img, points)

    plt.imshow(n_img)
    plt.title("After NMS")
    plt.axis('off')
    plt.savefig(f'tmp/{folder_path}_nms.jpg', dpi=300,bbox_inches='tight', pad_inches=0)
    plt.close()

    # Merge lines without indices
    onms_lines = merge_lines_torch(torch.tensor(lines[:,:,:2]), return_indices=False).int().cpu().numpy()
    print(f"Number of lines after orig NMS: {len(onms_lines)}")

    onms_img = show_lines(image, onms_lines)
    onms_img = show_points(onms_img, points)

    plt.imshow(onms_img)
    plt.title("After orig NMS")
    plt.axis('off')
    plt.savefig(f'tmp/{folder_path}_orig_nms.jpg', dpi=300,bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':
    from ... import logger

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--conf", type=str, default=None)
    argParser.add_argument("--show", action="store_true")
    args = argParser.parse_args()

    extractor_conf = OmegaConf.load(args.conf) if args.conf is not None else LineExtractor.default_conf
    extractor = LineExtractor(extractor_conf)

    if os.path.exists("tmp"):
        os.system("rm -r tmp")
    os.makedirs("tmp", exist_ok=True)

    for val in tqdm(glob.glob(str(DATA_PATH / "revisitop1m_POLD2/**/base_image.jpg"), recursive=True)):
        test_extractor(extractor, os.path.split(val)[0], extractor_conf.device, args.show)