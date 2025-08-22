import argparse
from collections import defaultdict
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt

from .. import settings
from ..visualization.global_frame import GlobalFrame
from ..visualization.two_view_frame import TwoViewFrame
from . import eval_pipeline, get_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=str)
    parser.add_argument("--x", type=str, default=None)
    parser.add_argument("--y", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument(
        "--default_plot", type=str, default=TwoViewFrame.default_conf["default"]
    )

    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    results = {}
    summaries = defaultdict(dict)

    predictions = {}

    if args.backend:
        matplotlib.use(args.backend)

    bm = get_benchmark(args.benchmark)
    dataset = bm.get_dataset()

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

    pprint(summaries)

    plt.close("all")

    frame = GlobalFrame(
        {"child": {"default": args.default_plot}, **vars(args)},
        results,
        dataset,
        predictions,
        child_frame=TwoViewFrame,
    )
    frame.draw()
    plt.show()
