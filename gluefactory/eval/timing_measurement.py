"""
Performs timing on a model and a dataset. Model and dataset must be specified in a conf file.
Attention: Make sure you configure batch_size and dataset split correctly.

- Ex run: python -m gluefactory.eval.timing_measurement --conf=gluefactory/configs/timing_conf.yaml --num_s=100 --device=cuda
"""

import argparse
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device


default_conf = {
    "model": {
        "name": ""
    },
    "dataset": {
        "name": ""
    }
}

def get_dataset_and_loader(
    dset_conf
):  
    dataset = get_dataset(dset_conf.name)(dset_conf)
    loader = dataset.get_data_loader(dset_conf.get("split", "test"))
    return loader


def sync_and_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time()
    return t


def run_measurement(
    dataloader, model, num_s, name, device, batch_size, jpl_inner_timings=False
):
    count = 0
    timings = []
    for img in tqdm(
        dataloader,
        total=num_s // batch_size,
    ):
        img = batch_to_device(img, device, non_blocking=True)
        with torch.no_grad():
            start = sync_and_time()
            pred = model(img)
            end = sync_and_time()
            timings.append((end - start) / batch_size)
        count += batch_size
        if count >= num_s:
            break

    print(f"*** RESULTS FOR {name} ON {num_s} IMAGES WITH BATCH SIZE {batch_size} ***")
    mean = round(np.mean(timings), 6)
    print(f"\tMean: {mean} --> ~{1/mean} FPS")
    median = round(np.median(timings), 6)
    print(f"\tMedian: {median} --> ~{1/median} FPS")
    max_t = round(np.max(timings), 6)
    min_t = round(np.min(timings), 6)
    std_t = round(np.std(timings), 6)
    print(f"\tMax: {max_t} --> ~{1/max_t} FPS")
    print(f"\tMin: {min_t} --> ~{1/min_t} FPS")
    print(f"\tStd: {std_t} --> ~{1/std_t} FPS")
    perc90 = round(np.percentile(timings, 90), 6)
    perc95 = round(np.percentile(timings, 95), 6)
    perc99 = round(np.percentile(timings, 99), 6)
    print(f"\t90th-percentile: {perc90} --> ~{1/perc90} FPS")
    print(f"\t95th-percentile: {perc95} --> ~{1/perc95} FPS")
    print(f"\t99th-percentile: {perc99} --> ~{1/perc99} FPS")
    
    if jpl_inner_timings:
        print(f"INNER TIMINGS JPLDD")
        inner_timings = model.get_current_timings()
        for k, v in inner_timings.items():
            print(f"\t{k}: {round(v, 6)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf", type=str, help="Name of config to run"
    )
    parser.add_argument(
        "--num_s", type=int, default=100, help="Number of timing samples."
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    args = parser.parse_args()

    default_conf = OmegaConf.create(default_conf)
    conf = OmegaConf.merge(default_conf, OmegaConf.load(args.conf))
    dset_conf = conf["data"]
    model_conf = conf["model"]
    model_is_jpl = "joint_point_line_extractor" in model_conf["name"]
    do_jpl_inner_timing = model_is_jpl and model_conf["timeit"]
    
    print("NUMBER OF SAMPLES: ", args.num_s)
    print("CONF TO TEST: ", args.conf)
    print("Dataset: ", dset_conf.name)
    print("--Split: ", dset_conf.split)
    print("Model: ", model_conf.name)
    
    if model_is_jpl: 
        print("JPLDD-Inner timing activated: ", do_jpl_inner_timing)

   
    # get data loader
    dataloader = get_dataset_and_loader(dset_conf)
    
    if args.device == "cuda":
        assert torch.cuda.is_available()
    elif args.device == "mps":
        assert torch.backends.mps.is_built()
        
    device = args.device
    print(f"Using Device: {device}")
    
    # get model
    model = get_model(model_conf.name)(model_conf)
    model.eval()
    model.to(device)

    run_measurement(
        dataloader=dataloader,
        model=model,
        num_s=args.num_s,
        name=model_conf.name,
        device=device,
        batch_size=dataloader.batch_size,
        jpl_inner_timings=do_jpl_inner_timing
    )
