"""Benchmark script for TwoViewPipeline inference latency.

Usage:
    python benchmark.py --checkpoint <experiment> image0.jpg image1.jpg
    python benchmark.py --conf <config> image0.jpg image1.jpg
    python benchmark.py --checkpoint <experiment> image0.jpg image1.jpg --resize 1024 --side long
    python benchmark.py --checkpoint <experiment> image0.jpg image1.jpg --repeat 200 --warmup 20
"""

import argparse
import logging
import time

import numpy as np
import torch

from gluefactory.eval.io import get_eval_parser, load_model, parse_eval_args
from gluefactory.utils import misc
from gluefactory.utils.preprocess import ImagePreprocessor, load_image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

torch.set_grad_enabled(False)


def measure(fn, device, warmup=10, repeat=100):
    """Measure latency of a callable with proper CUDA synchronization."""
    timings = np.zeros(repeat)
    use_cuda = device.type == "cuda"

    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for i in range(repeat):
        if use_cuda:
            starter.record()
            fn()
            ender.record()
            torch.cuda.synchronize()
            timings[i] = starter.elapsed_time(ender)
        else:
            start = time.perf_counter()
            fn()
            timings[i] = (time.perf_counter() - start) * 1e3

    stats = {
        "mean": timings.mean(),
        "std": timings.std(),
        "median": np.median(timings),
        "min": timings.min(),
        "max": timings.max(),
    }
    if use_cuda:
        stats["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return stats


def print_stats(name, stats):
    mem = (
        f"  mem: {stats['peak_memory_mb']:>7.1f} MB"
        if "peak_memory_mb" in stats
        else ""
    )
    print(
        f"  {name:<12} {stats['median']:>8.2f} ms  (mean {stats['mean']:.2f} ± {stats['std']:.2f}){mem}"
    )


def benchmark_pipeline(
    model, data, device, warmup=10, repeat=100, extract_parallel=False
):
    """Benchmark each stage of a TwoViewPipeline independently."""
    results = {}

    # --- Extraction ---
    if model.conf.extractor.name:
        if extract_parallel:
            vdata = misc.concat_tree(misc.iterelements(data, pattern="view{i}"))
            results["extractor"] = measure(
                lambda: model.extract_view(vdata), device, warmup, repeat
            )
        else:

            def run_extract():
                model.extract_view(data["view0"])
                model.extract_view(data["view1"])

            results["extractor"] = measure(run_extract, device, warmup, repeat)

    # Run extraction once to get intermediate predictions for downstream stages
    pred = {}
    if model.conf.extractor.name:
        for i in range(2):
            pred_i = model.extract_view(data[f"view{i}"])
            pred.update({f"{k}{i}": v for k, v in pred_i.items()})

    # --- Matching ---
    # Note: shallow-copy the input dict each call because some matchers (e.g.
    # pglue) permute feature tensors in-place on the dict during prepare().
    if model.conf.matcher.name:
        match_input = {**data, **pred}
        results["matcher"] = measure(
            lambda: model.matcher({**match_input}), device, warmup, repeat
        )
        match_pred = model.matcher({**match_input})
        pred.update(match_pred)

    # --- Filter ---
    if model.conf.filter.name:
        filter_input = {**data, **pred}
        results["filter"] = measure(
            lambda: model.filter({**filter_input}), device, warmup, repeat
        )
        filter_pred = model.filter({**filter_input})
        pred.update(filter_pred)

    # --- Solver / Refiner ---
    if model.conf.solver.name:
        solver_input = {**data, **pred}
        results["refiner"] = measure(
            lambda: model.solver({**solver_input}), device, warmup, repeat
        )

    # --- Full pipeline ---
    results["total"] = measure(lambda: model(data), device, warmup, repeat)

    return results


def get_parser():
    parser = get_eval_parser(
        argparse.ArgumentParser(
            description="Benchmark TwoViewPipeline inference latency"
        )
    )
    parser.add_argument(
        "images",
        nargs="*",
        default=["assets/sacre_coeur1.jpg", "assets/sacre_coeur2.jpg"],
        help="paths to two input images (default: assets/sacre_coeur{1,2}.jpg)",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu", "mps"], default="auto"
    )
    parser.add_argument(
        "--compile", action="store_true", help="compile model with torch.compile"
    )
    parser.add_argument(
        "--extract_parallel",
        action="store_true",
        help="extract both views in a single batched forward pass",
    )
    parser.add_argument(
        "--repeat", "-r", type=int, default=10, help="number of timed iterations"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="number of warmup iterations"
    )

    # Image preprocessing
    preproc = parser.add_argument_group("image preprocessing")
    preproc.add_argument(
        "--resize",
        type=int,
        default=768,
        help="resize target edge length (default: 768)",
    )
    preproc.add_argument(
        "--side", choices=["short", "long", "vert", "horz"], default="long"
    )
    preproc.add_argument(
        "--grayscale", action="store_true", help="load images as grayscale"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_intermixed_args()

    if len(args.images) != 2:
        parser.error("expected exactly 2 image paths")

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Parse config (re-uses eval CLI: --checkpoint, --conf, dotlist overrides)
    _, conf = parse_eval_args("benchmark", args, "configs/")
    model_conf = conf.get("model", conf)

    # Load model
    logger.info("Loading model...")
    model = load_model(model_conf, conf.get("checkpoint"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device)
    if device.type == "cuda":
        model_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    if args.compile:
        model = model.compile()
        logger.info("Model compiled with torch.compile")
    logger.info("Model loaded on %s", device)

    # Preprocess images (resize long edge, then center pad to square)
    preprocessor = ImagePreprocessor(
        {
            "resize": args.resize,
            "side": args.side,
            "square_pad": True,
            "center_pad": True,
        }
    )
    view0 = preprocessor(load_image(args.images[0], grayscale=args.grayscale))
    view1 = preprocessor(load_image(args.images[1], grayscale=args.grayscale))

    # Add batch dimension and move to device
    data = {
        "view0": misc.batch_to_device(
            {
                k: v[None] if isinstance(v, torch.Tensor) else v
                for k, v in view0.items()
            },
            device,
        ),
        "view1": misc.batch_to_device(
            {
                k: v[None] if isinstance(v, torch.Tensor) else v
                for k, v in view1.items()
            },
            device,
        ),
    }

    img_size0 = view0["image"].shape[-2:]
    img_size1 = view1["image"].shape[-2:]

    # Run benchmark
    logger.info(
        "Benchmarking: %d warmup + %d timed iterations", args.warmup, args.repeat
    )
    results = benchmark_pipeline(
        model,
        data,
        device,
        warmup=args.warmup,
        repeat=args.repeat,
        extract_parallel=args.extract_parallel,
    )

    # Print results
    print()
    print(f"Device:      {device}")
    print(f"Image 0:     {img_size0[1]}x{img_size0[0]}")
    print(f"Image 1:     {img_size1[1]}x{img_size1[0]}")
    print(f"Iterations:  {args.repeat}")
    print()
    for name, stats in results.items():
        print_stats(name, stats)
    print()
    total = results["total"]
    print(f"Throughput:  {1000 / total['mean']:>8.2f} pairs/s")
    if device.type == "cuda" and "peak_memory_mb" in total:
        gpu_total_mb = torch.cuda.get_device_properties(device).total_memory / 1024**2
        activation_mb = total["peak_memory_mb"] - model_memory_mb
        if activation_mb > 0:
            max_batch = int((gpu_total_mb - model_memory_mb) / activation_mb)
            batched_throughput = max_batch * 1000 / total["mean"]
            print()
            print(f"GPU memory:  {gpu_total_mb:>7.0f} MB total")
            print(f"  Model:     {model_memory_mb:>7.1f} MB")
            print(f"  Per pair:  {activation_mb:>7.1f} MB")
            print(f"  Max batch: {max_batch:>4d} (theoretical)")
            print(f"  Batched:   {batched_throughput:>8.2f} pairs/s (theoretical)")
