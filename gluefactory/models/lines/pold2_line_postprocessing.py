import argparse
import glob
import os
import shutil
import time
from functools import cmp_to_key

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from gluefactory.models.lines.line_refinement import merge_lines


class LineExtractor:
    def __init__(self, num_sample, num_sample_strong, device):

        self.num_sample = num_sample
        self.num_sample_strong = num_sample_strong
        self.device = device
        # Precompute coeffs
        coeffs = torch.arange(0, 1, 1 / num_sample).to(self.device).view(-1, 1)
        coeffs_second = (
            (1 - torch.arange(0, 1, 1 / num_sample)).to(self.device).view(-1, 1)
        )

        coeffs_strong = (
            torch.arange(0, 1, 1 / num_sample_strong).to(self.device).view(-1, 1)
        )
        coeffs_strong_second = (
            (1 - torch.arange(0, 1, 1 / num_sample_strong)).to(self.device).view(-1, 1)
        )

        # Precompute indices
        MAX_POINT_SIZE = 1500
        indices = torch.combinations(torch.arange(0, MAX_POINT_SIZE), r=2).numpy()

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
        self.MAX_POINT_SIZE = MAX_POINT_SIZE

    # Distance map processing
    def process_distance_map(self, distance_map):
        distance_map = distance_map < 0.5

        average_filter = nn.AvgPool2d(13, stride=1, padding=6)
        distance_map_smooth = average_filter(distance_map[None, :, :].float())

        output = (distance_map & (distance_map_smooth < 1))[0]

        # plt.imsave(f"output/distance_{threshold}.jpeg", output.cpu().numpy())

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

    # Distance map filtering
    def filter_with_distance_field(
        self,
        points,
        distance_map,
        distance_map_float,
        indices,
        coeffs,
        coeffs_second,
        sample_points_num,
        ratio_inliner=1.0,
        mean_value_ratio=100,
    ):
        # Get coordinates
        points_coordinates = self.get_coordinates(
            points, indices, coeffs, coeffs_second
        )

        # Sample points
        sampled_values_along_the_line = self.sample_map(
            points_coordinates, distance_map
        ).view(sample_points_num, -1)
        sampled_values_along_the_line_float = self.sample_map(
            points_coordinates, distance_map_float
        ).view(sample_points_num, -1)

        # We reduce the values along the line
        detected_line_indices = (
            torch.sum(sampled_values_along_the_line, dim=0)
            >= sample_points_num * ratio_inliner
        )
        detected_line_float = (
            torch.mean(sampled_values_along_the_line_float.float(), dim=0)
            <= mean_value_ratio
        )

        # Finally we can filter the indices
        return indices[detected_line_indices & detected_line_float]

    # Angle map filtering
    def sample_values_angle_map(
        self, points, angle_map, indices, coeffs, coeffs_second, sample_points_num
    ):
        # Get coordinates
        points_coordinates = self.get_coordinates(
            points, indices, coeffs, coeffs_second
        )

        # Get nearest neighbours coordinates
        nn_points_coordinates = torch.round(points_coordinates)

        # Sample points - we get nearest neighbours implementation with nn_points_coordinates
        return angle_map[nn_points_coordinates[:, 1], nn_points_coordinates[:, 0]].view(
            sample_points_num, -1, 2
        )

    def filter_with_angle_field(
        self,
        points,
        angle_map,
        indices,
        coeffs,
        coeffs_second,
        sample_points_num,
        threshold=1.0,
    ):
        # Calculate the angle of each segment
        line_directions = points[indices[:, 1]] - points[indices[:, 0]]
        line_directions = line_directions / torch.norm(
            line_directions.float(), dim=-1, keepdim=True
        )

        # Sample the values in the angle map
        sampled_values_along_the_line = self.sample_values_angle_map(
            points, angle_map, indices, coeffs, coeffs_second, sample_points_num
        )
        sampled_values_along_the_line = sampled_values_along_the_line / torch.norm(
            sampled_values_along_the_line.float(), dim=-1, keepdim=True
        )

        # Apply dot product elementwise on the last dimension
        dotted = torch.sum(line_directions * sampled_values_along_the_line, dim=-1)
        abs_dotted = torch.abs(dotted)

        # Count number of inliners
        number_inliners = torch.sum(abs_dotted.float() > threshold, dim=0)

        # TODO: Make it parametrizable
        valid_indices = number_inliners > 0.5 * sample_points_num

        # Return values larger than ratio
        return indices[valid_indices]

    # Post processing step
    def three_stage_filter(
        self, points, distance_map, binary_distance_map, angle_map, indices_image
    ):

        # First pass - weak filter
        indices_image = self.filter_with_distance_field(
            points,
            binary_distance_map,
            distance_map,
            indices_image,
            self.coeffs,
            self.coeffs_second,
            self.num_sample,
            ratio_inliner=1.0,
            mean_value_ratio=0.8,
        )

        # Second pass - strong filter
        indices_image = self.filter_with_distance_field(
            points,
            binary_distance_map,
            distance_map,
            indices_image,
            self.coeffs_strong,
            self.coeffs_strong_second,
            self.num_sample_strong,
            ratio_inliner=0.98,
            mean_value_ratio=0.8,
        )

        # Third stage - discard using angle map
        # indices_image = self.filter_with_angle_field(
        #    points, angle_map, indices_image, self.coeffs,self.coeffs_second, self.num_sample, threshold=0.9)

        return indices_image

    def post_processing_step(self, points, distance_map, angle_map):
        # Get indices
        if len(points) > self.MAX_POINT_SIZE:
            print(
                f"WARNING: We have more than {self.MAX_POINT_SIZE} points in this image ({len(points)}), keeping only {self.MAX_POINT_SIZE} firsts"
            )
            points = points[: self.MAX_POINT_SIZE]

        # Precompute indices
        number_pairs = int(len(points) * (len(points) - 1) / 2)
        indices_image = self.indices[:number_pairs]

        # Normalize distance map
        distance_map = distance_map.float()
        distance_map /= torch.max(distance_map)

        # Process distance map
        binary_distance_map = self.process_distance_map(distance_map)

        # Apply two stage filter
        return self.three_stage_filter(
            points, distance_map, binary_distance_map, angle_map, indices_image
        )


def print_points(image, points):
    for point in points:
        cv2.circle(image, (point[0], point[1]), 3, (0, 0, 255), -1)

    return image


def print_lines(image, lines):
    for pair_line in lines:
        cv2.line(image, pair_line[0], pair_line[1], (255, 255, 0), 3)

    return image


def post_process_image(extractor, folder_path):
    image = torch.from_numpy(np.array(Image.open(f"{folder_path}/base_image.jpg"))).to(
        device
    )
    distance_map = torch.from_numpy(np.array(Image.open(f"{folder_path}/df.jpg"))).to(
        device
    )
    angle_map = torch.from_numpy(np.array(Image.open(f"{folder_path}/angle.jpg"))).to(
        device
    )

    angle_map = angle_map.float() / 255 * np.pi
    angle_map = torch.cat(
        (torch.cos(angle_map).unsqueeze(2), torch.sin(angle_map).unsqueeze(2)), dim=2
    )

    points_np = np.load(f"{folder_path}/keypoints.npy", allow_pickle=True)
    points = torch.from_numpy(points_np).to(device).int()

    # Start counter
    start_time = time.perf_counter()

    # Post processing step
    indices_image = extractor.post_processing_step(points, distance_map, angle_map)

    # End counter
    end_time = time.perf_counter()

    measure = False
    if measure:
        print(f"{1000*(end_time - start_time)}")
    else:
        plt.figure()

        image = image.cpu().detach().numpy()
        points = points.cpu().numpy()
        indices_image = indices_image.cpu().numpy()

        # Samples lines from indices
        lines = points[indices_image]

        # Apply non-max suppresion on lines - GPU implementation from deeplsd
        # https://github.com/cvg/DeepLSD/blob/52212738362711254f040c673276905c73b86ca5/deeplsd/geometry/line_utils.py#L431
        # TODO: Make it parametrizable
        lines = merge_lines(torch.tensor(lines)).int().cpu().numpy()

        # Print lines
        image = print_lines(image, lines)

        # Print points
        image = print_points(image, points)

        folder_path = folder_path.split("/")[-1]

        print(f"Elapsed time for {folder_path} is : {end_time - start_time}")

        plt.figure(frameon=False)
        plt.imshow(image)
        if args.show:
            plt.show()

        plt.axis("off")
        plt.savefig(
            f"output/{folder_path}.jpeg", dpi=300, bbox_inches="tight", pad_inches=0
        )


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-num", "--num_sample", help="number of sample points")
    argParser.add_argument(
        "-num_second",
        "--num_sample_strong",
        help="number of sample points for strong filter",
    )
    argParser.add_argument(
        "-s", "--show", help="flag to show animation", action="store_true"
    )
    argParser.add_argument("-f", "--f", help="use second method", action="store_true")
    argParser.add_argument("-d", "--device", help="Device")
    argParser.add_argument(
        "-r1",
        "--ratio1",
        help="Ratio to keep a line weak filter",
        default=1.0,
        type=float,
    )
    argParser.add_argument(
        "-r2",
        "--ratio2",
        help="Ratio to keep a line strong filter",
        default=1.0,
        type=float,
    )
    args = argParser.parse_args()

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device is not None:
        device = args.device

    # Get samples
    num_sample = 8
    if args.num_sample is not None:
        num_sample = int(args.num_sample)

    num_sample_strong = 300
    if args.num_sample is not None:
        num_sample_strong = int(args.num_sample_strong)

    extractor = LineExtractor(num_sample, num_sample_strong, device)

    shutil.rmtree("output")
    if not os.path.isdir("output"):
        os.mkdir("output")

    for val in glob.glob("**/base_image.jpg", recursive=True):
        post_process_image(extractor, os.path.split(val)[0])
