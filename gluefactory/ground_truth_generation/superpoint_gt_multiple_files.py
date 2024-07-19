"""
Run the homography adaptation with Superpoint for all images in the minidepth dataset.
Goal: create groundtruth with superpoint. Format: stores groundtruth for every image in a separate file.
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.geometry.homography import sample_homography_corners, warp_points
from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.settings import EVAL_PATH
from gluefactory.utils.image import numpy_image_to_torch

conf = {
    "patch_shape": [800, 800],
    "difficulty": 0.8,
    "translation": 1.0,
    "n_angles": 10,
    "max_angle": 60,
    "min_convexity": 0.05,
}

sp_conf = {
    "max_num_keypoints": None,
    "nms_radius": 4,
    "detection_threshold": 0.005,
    "remove_borders": 4,
    "descriptor_dim": 256,
    "channels": [64, 64, 128, 128, 256],
    "dense_outputs": None,
    "weights": None,  # local path of pretrained weights
}

homography_params = {
    "translation": True,
    "rotation": True,
    "scaling": True,
    "perspective": True,
    "scaling_amplitude": 0.2,
    "perspective_amplitude_x": 0.2,
    "perspective_amplitude_y": 0.2,
    "patch_ratio": 0.85,
    "max_angle": 1.57,
    "allow_artifacts": True,
}


def get_dataset_and_loader(num_workers):  # folder where dataset images are placed
    config = {
        "name": "minidepth",  # name of dataset class in gluefactory > datasets
        "grayscale": True,  # commented out things -> dataset must also have these keys but has not
        "preprocessing": {"resize": [800, 800]},
        "train_batch_size": 1,  # prefix must match split mode
        "num_workers": num_workers,
        "split": "train",  # if implemented by dataset class gives different splits
    }
    omega_conf = OmegaConf.create(config)
    dataset = get_dataset(omega_conf.name)(omega_conf)
    loader = dataset.get_data_loader(omega_conf.get("split", "train"))
    return loader


def sample_homography(img, conf: dict, size: list):
    data = {}
    H, _, coords, _ = sample_homography_corners(img.shape[:2][::-1], **conf)
    data["image"] = cv2.warpPerspective(img, H, tuple(size))
    data["H_"] = H.astype(np.float32)
    data["coords"] = coords.astype(np.float32)
    data["image_size"] = np.array(size, dtype=np.float32)
    return data


def ha_df(img, num=100):
    """Perform homography adaptation to regress line distance function maps.
    Args:
        img: a grayscale np image.
        num: number of homographies used during HA.
        border_margin: margin used to erode the boundaries of the mask.
        min_counts: any pixel which is not activated by more than min_count is BG.
    Returns:
        The aggregated distance function maps in pixels
        and the angle to the closest line.
    """
    h, w = img.shape[:2]

    aggregated_heatmap = np.zeros((w, h, num), dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SuperPoint(sp_conf).to(device)
    model.eval().to(device)

    with torch.no_grad():
        # iterate over num homographies
        for i in range(num):

            # warp image
            homography = sample_homography(img, conf, [w, h])

            # apply detector
            image_warped = homography["image"]
            pred = model({"image": numpy_image_to_torch(image_warped)[None].to(device)})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            # warped_heatmap = pred["heatmap"]
            keypoints = pred["keypoints"]
            scores = pred["keypoint_scores"]

            warped_back_kp = warp_points(keypoints, homography["H_"], inverse=True)
            warped_back_kp = np.floor(warped_back_kp).astype(int)

            for j in range(len(warped_back_kp)):
                x, y = warped_back_kp[j][0] + 1, warped_back_kp[j][1] + 1
                if x < w and y < h and x > 0 and y > 0 and scores[j] > 0:
                    aggregated_heatmap[y, x, i] = scores[j]

        mask = aggregated_heatmap > 0

        aggregated_heatmap_nan = aggregated_heatmap.copy()
        aggregated_heatmap_nan[~mask] = np.nan

        median_scores_non_zero = np.nanmedian(aggregated_heatmap_nan, axis=2)

    return median_scores_non_zero


def process_image(img_data, num_H, output_folder_path):
    img = img_data["image"]  # B x C x H x W
    img_npy = img.numpy()
    img_npy = img_npy[0, :, :, :]
    img_npy = np.transpose(img_npy, (1, 2, 0))  # H x W x C
    # Run homography adaptation

    # convert to superpoint format: img_npy to 0-255, uint8 and h * w
    sp_image = (img_npy[:, :, 0] * 255).astype(np.uint8)
    superpoint_heatmap = ha_df(sp_image, num=num_H)

    assert len(img_data["name"]) == 1  # Currently expect batch size one!
    # store gt in same structure as images of minidepth

    complete_out_folder = (output_folder_path / str(img_data["name"][0])).parent
    complete_out_folder.mkdir(parents=True, exist_ok=True)
    output_file_path = (
        complete_out_folder / f"{Path(img_data['name'][0]).name.split('.')[0]}.hdf5"
    )

    # Save the DF in a hdf5 file
    with h5py.File(output_file_path, "w") as f:
        f.create_dataset("superpoint_heatmap", data=superpoint_heatmap)


def export_ha(data_loader, output_folder_path, num_H, n_jobs):
    # Process each image in parallel
    Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(process_image)(img_data, num_H, output_folder_path)
        for img_data in tqdm(data_loader, total=len(data_loader))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_folder", type=str, help="Output folder.", default="superpoint_gt"
    )
    parser.add_argument(
        "--num_H", type=int, default=100, help="Number of homographies used during HA."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=2,
        help="Number of jobs (that perform HA) to run in parallel.",
    )
    parser.add_argument(
        "--n_jobs_dataloader",
        type=int,
        default=1,
        help="Number of jobs the dataloader uses to load images",
    )
    args = parser.parse_args()

    out_folder_path = EVAL_PATH / args.output_folder
    out_folder_path.mkdir(exist_ok=True, parents=True)

    print("OUTPUT PATH: ", out_folder_path)
    print("NUMBER OF HOMOGRAPHIES: ", args.num_H)
    print("N JOBS: ", args.n_jobs)
    print("N DATALOADER JOBS: ", args.n_jobs_dataloader)

    dataloader = get_dataset_and_loader(args.n_jobs_dataloader)
    export_ha(dataloader, out_folder_path, args.num_H, args.n_jobs)
    print("Done !")
