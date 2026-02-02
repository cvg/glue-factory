import argparse
from collections import defaultdict
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from .. import settings
from ..visualization.global_frame import GlobalFrame
from ..visualization.two_view_frame import TwoViewFrame
from . import eval_pipeline, get_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=str)
    parser.add_argument("--x", type=str, default=None)
    parser.add_argument("--y", type=str, default=None)
    parser.add_argument("--backend", type=str, default="WxAgg")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--default_plot", type=str, default=TwoViewFrame.default_conf["default"]
    )
    parser.add_argument("--show_plot", type=int, default=None)

    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    results = {}
    summaries = defaultdict(dict)

    predictions = {}

    if args.backend:
        matplotlib.use(args.backend)

    num_samples = args.num_samples
    for name in args.dotlist:
        possible_paths = [
            settings.EVAL_PATH / args.benchmark / name,  # Preferred
            settings.TRAINING_PATH / name / args.benchmark,
        ]
        experiment_dir = None
        for path in possible_paths:
            if path.exists():
                experiment_dir = path
                break
        if experiment_dir is None:
            raise FileNotFoundError(
                f"Experiment directory for {name} not found. "
                f" Checked: {possible_paths}"
            )
        pred_file = experiment_dir / "predictions.h5"
        s, results[name] = eval_pipeline.load_eval(experiment_dir)
        predictions[name] = pred_file
        for k, v in s.items():
            summaries[k][name] = v

        config = OmegaConf.load(experiment_dir / "conf.yaml")
        if config.get("num_samples") is not None and args.num_samples is None:
            num_samples = min(num_samples or 1e8, config.num_samples)

    pprint(summaries)
    plt.close("all")

    bm = get_benchmark(args.benchmark)
    if num_samples is not None:
        bm.num_samples = num_samples
    dataset = bm.get_dataset()

    argvars = vars(args)
    if args.x is None:
        argvars["x"] = bm.default_x
    if args.y is None:
        argvars["y"] = bm.default_y

    frame = GlobalFrame(
        {"child": {"default": args.default_plot}, **argvars},
        results,
        dataset,
        predictions,
        child_frame=TwoViewFrame,
    )
    frame.draw()
    print("Visualization done.")
    if args.show_plot is not None:
        print("Spawn child frame for detailed view...")
        frame.spawn_child(args.dotlist[0], args.show_plot, event=1)
    plt.show()
