"""
Run the homography adaptation with Superpoint for all images in the minidepth dataset.
Goal: create groundtruth with superpoint. Format: stores groundtruth for every image in a separate file.
"""

import argparse

import numpy as np
import torch
import time
from tqdm import tqdm

from omegaconf import OmegaConf

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device

model_configs = {
    "aliked": {
        "name": 'extractors.aliked',
        "max_num_keypoints": 1024,
        "model_name": "aliked-n16",
        "detection_threshold": -1,
        "force_num_keypoints": False,
        "pretrained": True,
        "nms_radius": 4
    },

    "sp": {
        "name": "extractors.superpoint_open",
        "max_num_keypoints": 1024,
        "nms_radius": 4,
        "detection_threshold": -1,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
        "dense_outputs": None,
        "weights": None,  # local path of pretrained weights
    },

    "deeplsd": {
        "name": "lines.deeplsd",
        "min_length": 25,
        "max_num_lines": None,
        "force_num_lines": False,
        "model_conf": {
            "detect_lines": True,
            "line_detection_params": {
                "merge": False,
                "grad_nfa": True,
                "filtering": "normal",
                "grad_thresh": 3.0,
            },
            "optimize": False,
        },
        "optimize": False,
    },

    "sold2": {
        "name": "lines.sold2",
        "min_length": 25,
        "max_num_lines": None,
        "force_num_lines": False,
        "model_conf": {
            "detect_lines": True,
            "line_detection_params": {
                "merge": False,
                "grad_nfa": True,
                "filtering": "normal",
                "grad_thresh": 3.0,
            },
        }
    },

    "jpldd": {
        "name": "extractors.jpldd.joint_point_line_extractor",
        "aliked_model_name": "aliked-n16",
        "max_num_keypoints": 1024,  # setting for training, for eval: -1
        "detection_threshold": -1,  # setting for training, for eval: 0.2
        "force_num_keypoints": False,
        "training": {  # training settings
            "do": False,
        },
        "line_detection": {
            "do": True,
        },
        "nms_radius": 4,
        "line_neighborhood": 5,  # used to normalize / denormalize line distance field
        "timeit": False,  # override timeit: False from BaseModel
        # "line_df_decoder_channels": 64, # uncomment it for models having the old number of channels
        # "line_af_decoder_channels": 64,
    }
}


def get_dataset_and_loader(num_workers, batch_size):  # folder where dataset images are placed
    config = {
        'name': 'minidepth',  # name of dataset class in gluefactory > datasets
        'grayscale': False,  # commented out things -> dataset must also have these keys but has not
        'preprocessing': {
            'resize': [800, 800]
        },
        'test_batch_size': batch_size,  # prefix must match split mode
        'num_workers': num_workers,
        'split': 'test'  # test is not shuffled, train is -> to get consistent results on same images, use test
    }
    omega_conf = OmegaConf.create(config)
    dataset = get_dataset(omega_conf.name)(omega_conf)
    loader = dataset.get_data_loader(omega_conf.get('split', 'test'))
    return loader


def sync_and_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time()
    return t


def run_measurement(dataloader, model, num_s, name, device, batch_size, do_jpldd_inner_timings=False):
    count = 0
    timings = []
    for img in tqdm(dataloader, total=num_s//batch_size,):
        img = batch_to_device(img, device, non_blocking=True)
        with torch.no_grad():
            start = sync_and_time()
            pred = model(img)
            end = sync_and_time()
            timings.append((end - start)/batch_size)
        count += batch_size
        if count >= num_s:
            break

    print(f"*** RESULTS FOR {name} ON {num_s} IMAGES WITH BATCH SIZE {batch_size} ***")
    print(f"\tMean: {round(np.mean(timings), 6)}")
    print(f"\tMedian: {round(np.median(timings), 6)}")
    print(f"\tMax: {round(np.max(timings), 6)}")
    print(f"\tMin: {round(np.min(timings), 6)}")
    print(f"\tStd: {round(np.std(timings), 6)}")
    if do_jpldd_inner_timings and name == "jpldd":
        print(f"INNER TIMINGS JPLDD")
        inner_timings = model.get_current_timings()
        for k, v in inner_timings.items():
            print(f"\t{k}: {round(v, 6)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, choices=['aliked', 'sp', 'deeplsd', 'jpldd'])
    parser.add_argument('--num_s', type=int, default=100, help='Number of timing samples.')
    parser.add_argument('--jpldd_inner_timings', action="store_true",
                        help='activates measurement of single parts of the jpldd pipeline')
    parser.add_argument('--n_jobs_dataloader', type=int, default=1,
                        help='Number of jobs the dataloader uses to load images')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size used for the model')
    args = parser.parse_args()

    print("NUMBER OF SAMPLES: ", args.num_s)
    print("MODEL TO TEST: ", args.config)
    print("N DATALOADER JOBS: ", args.n_jobs_dataloader)
    if args.config == 'jpldd':
        print("JPLDD-Inner timing activated: ", args.jpldd_inner_timings)

    dataloader = get_dataset_and_loader(args.n_jobs_dataloader,args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = model_configs[args.config]
    model_name = config["name"]
    if args.config == 'jpldd' and args.jpldd_inner_timings:
        config["timeit"] = True
    model = get_model(model_name)(config)

    model.eval().to(device)

    run_measurement(dataloader, model, args.num_s, args.config, device,args.batch_size,
                    do_jpldd_inner_timings=args.jpldd_inner_timings)
