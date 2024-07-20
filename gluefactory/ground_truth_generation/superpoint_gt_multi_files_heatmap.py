"""
Run the homography adaptation with Superpoint for all images in the minidepth or oxford paris mini dataset.
Goal: create groundtruth with superpoint. Format: stores groundtruth for every image in a separate file.
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from joblib import Parallel, delayed
from kornia.geometry.transform import warp_perspective
from kornia.morphology import erosion
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.geometry.homography import sample_homography_corners
from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.settings import EVAL_PATH

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

H_params = {
    "difficulty": 0.8,
    "translation": 1.0,
    "max_angle": 60,
    "n_angles": 10,
    "min_convexity": 0.05,
}

ha = {
    "enable": False,
    "num_H": 100,
    "mini_bs": 3,
    "aggregation": "mean",
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


def get_dataset_and_loader(num_workers: int, dataset: str):  # folder where dataset images are placed
    print("Loading Dataset {}...".format(dataset))
    config = {
        "name": dataset,  # name of dataset class in gluefactory > datasets
        "grayscale": True,  # commented out things -> dataset must also have these keys but has not
        "preprocessing": {"resize": [800, 800]},
        "train_batch_size": 1,  # prefix must match split mode
        "val_batch_size": 1,  # prefix must match split mode
        "all_batch_size": 1,
        "num_workers": num_workers,
        "split": "all",  # if implemented by dataset class gives different splits
    }
    omega_conf = OmegaConf.create(config)
    dataset = get_dataset(omega_conf.name)(omega_conf)
    loader = dataset.get_data_loader(omega_conf.get("split", "all"))
    return loader


def sample_homography(img, conf: dict, size: list):
    data = {}
    H, _, coords, _ = sample_homography_corners(img.shape[:2][::-1], **conf)
    data["image"] = cv2.warpPerspective(img, H, tuple(size))
    data["H_"] = H.astype(np.float32)
    data["coords"] = coords.astype(np.float32)
    data["image_size"] = np.array(size, dtype=np.float32)
    return data


def ha_forward(img, num=100):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SuperPoint(sp_conf).to(device)
    model.eval().to(device)

    Hs = []
    for i in range(num):
        if i == 0:
            # Always include at least the identity
            Hs.append(torch.eye(3, dtype=torch.float, device=device))
        else:
            Hs.append(
                torch.tensor(
                    sample_homography_corners((w, h), patch_shape=(w, h), **H_params)[
                        0
                    ],
                    dtype=torch.float,
                    device=device,
                )
            )
    Hs = torch.stack(Hs, dim=0)

    bs = ha["mini_bs"]
    B = 1

    erosion_kernel = torch.tensor(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=torch.float,
    )

    erosion_kernel = erosion_kernel.to(device)

    sp_image_tensor = (
        torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    )
    n_mini_batch = int(np.ceil(num / bs))
    scores = torch.empty((B, 0, h, w), dtype=torch.float, device=device)
    counts = torch.empty((B, 0, h, w), dtype=torch.float, device=device)

    for i in range(n_mini_batch):
        H = Hs[i * bs : (i + 1) * bs]
        nh = len(H)
        H = H.repeat(B, 1, 1)
        H = H.to(device)

        a = torch.repeat_interleave(sp_image_tensor, nh, dim=0)
        warped_imgs = warp_perspective(a, H, (h, w), mode="bilinear")

        for j, img in enumerate(warped_imgs):
            with torch.no_grad():
                img1 = img / 255.0  # Normalize image
                img1 = img1.unsqueeze(0)  # Add batch dimension
                pred = model({"image": img1.to(device)})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

                warped_heatmap = pred["heatmap"]

                # convert to pytorch tensor
                score = torch.tensor(
                    warped_heatmap, dtype=torch.float32, device=device
                ).unsqueeze(0)

                # Compute valid pixels
                H_inv = torch.inverse(H[j])

                count = warp_perspective(
                    torch.ones_like(score).unsqueeze(1),
                    H[j].unsqueeze(0),
                    (h, w),
                    mode="nearest",
                )

                count = erosion(count, erosion_kernel)
                count = warp_perspective(
                    count, H_inv.unsqueeze(0), (h, w), mode="nearest"
                )[:, 0]
                score = warp_perspective(
                    score[:, None], H_inv.unsqueeze(0), (h, w), mode="bilinear"
                )[:, 0]

            scores = torch.cat([scores, score.reshape(B, 1, h, w)], dim=1)
            counts = torch.cat([counts, count.reshape(B, 1, h, w)], dim=1)
            scores[counts == 0] = 0
            score = scores.max(dim=1)[0]

            scoremap = score.squeeze(0)
    return scoremap


def process_image(img_data, num_H, output_folder_path):
    img = img_data["image"]  # B x C x H x W
    img_npy = img.numpy()
    img_npy = img_npy[0, :, :, :]
    img_npy = np.transpose(img_npy, (1, 2, 0))  # H x W x C
    # Run homography adaptation

    # convert to superpoint format: img_npy to 0-255, uint8 and h * w
    sp_image = (img_npy[:, :, 0] * 255).astype(np.uint8)
    superpoint_heatmap = ha_forward(sp_image, num=num_H)
    superpoint_heatmap = superpoint_heatmap.cpu()

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
    parser.add_argument("dataset", choices=["minidepth", "oxford_paris_mini"])
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

    print("DATASET: ", args.dataset)
    print("OUTPUT PATH: ", out_folder_path)
    print("NUMBER OF HOMOGRAPHIES: ", args.num_H)
    print("N JOBS: ", args.n_jobs)
    print("N DATALOADER JOBS: ", args.n_jobs_dataloader)

    dataloader = get_dataset_and_loader(args.n_jobs_dataloader, args.dataset)
    export_ha(dataloader, out_folder_path, args.num_H, args.n_jobs)
    print("Done !")
