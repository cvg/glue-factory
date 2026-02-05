import argparse
import os
import pprint
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import plotly.io as pio
from omegaconf import OmegaConf

import gluefactory

from .. import settings
from ..visualization.global_frame import GlobalFrame
from ..visualization.two_view_frame import TwoViewFrame
from . import eval_pipeline, get_benchmark
from .io import format_summaries

logger = gluefactory.logger


def select_backend(preferred: str | None = None) -> str:
    """Select matplotlib backend, preferring local display, falling back to webagg."""
    if preferred:
        matplotlib.use(preferred)
        return preferred
    # Check if local display is available
    has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if has_display:
        # Try backends in order of preference
        for backend in ["WxAgg", "QtAgg", "Qt5Agg", "TkAgg", "GTK3Agg"]:
            try:
                matplotlib.use(backend)
                return backend
            except ImportError:
                continue
    matplotlib.use("webagg")
    return "webagg"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=str)
    parser.add_argument("--x", type=str, default=None)
    parser.add_argument("--y", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--default_plot", type=str, default=None)
    parser.add_argument("--show_plot", type=int, default=None)
    parser.add_argument("--renderer", type=str, default="firefox")

    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    results = {}
    summaries = defaultdict(dict)

    predictions = {}
    eval_predictions = {}

    backend = select_backend(args.backend)
    logger.info(f"Using matplotlib backend: {backend}")
    pio.renderers.default = args.renderer

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
        eval_pred_file = experiment_dir / "eval_predictions.h5"
        s, results[name] = eval_pipeline.load_eval(experiment_dir)
        s = format_summaries(s)
        predictions[name] = pred_file
        if eval_pred_file.exists():
            eval_predictions[name] = eval_pred_file
        for k, v in s.items():
            summaries[k][name] = v

        config = OmegaConf.load(experiment_dir / "conf.yaml")
        if config.get("num_samples") is not None and args.num_samples is None:
            num_samples = min(num_samples or 1e8, config.num_samples)

    logger.info("Aggregated evaluation summaries:\n%s", pprint.pformat(dict(summaries)))
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

    default_plot = args.default_plot
    if default_plot is None:
        default_plot = bm.default_plot
    if default_plot is None:
        default_plot = TwoViewFrame.default_conf["default"]

    frame = GlobalFrame(
        {"child": {"default": default_plot}, **argvars},
        results,
        dataset,
        predictions,
        eval_predictions=eval_predictions,
        child_frame=TwoViewFrame,
    )
    frame.draw()
    print("Visualization done.")
    if args.show_plot is not None:
        print("Spawn child frame for detailed view...")
        frame.spawn_child(args.dotlist[0], args.show_plot, event=1)
    plt.show()
