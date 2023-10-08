from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import EVAL_PATH
from ..utils.export_predictions import export_predictions
from .eval_pipeline import EvalPipeline, load_eval
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import aggregate_pr_results, get_tp_fp_pts


def eval_dataset(loader, pred_file, suffix=""):
    results = defaultdict(list)
    results["num_pos" + suffix] = 0
    cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
    for data in tqdm(loader):
        pred = cache_loader(data)

        if suffix == "":
            scores = pred["matching_scores0"].numpy()
            sort_indices = np.argsort(scores)[::-1]
            gt_matches = pred["gt_matches0"].numpy()[sort_indices]
            pred_matches = pred["matches0"].numpy()[sort_indices]
        else:
            scores = pred["line_matching_scores0"].numpy()
            sort_indices = np.argsort(scores)[::-1]
            gt_matches = pred["gt_line_matches0"].numpy()[sort_indices]
            pred_matches = pred["line_matches0"].numpy()[sort_indices]
        scores = scores[sort_indices]

        tp, fp, scores, num_pos = get_tp_fp_pts(pred_matches, gt_matches, scores)
        results["tp" + suffix].append(tp)
        results["fp" + suffix].append(fp)
        results["scores" + suffix].append(scores)
        results["num_pos" + suffix] += num_pos

    # Aggregate the results
    return aggregate_pr_results(results, suffix=suffix)


class ETH3DPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "eth3d",
            "batch_size": 1,
            "train_batch_size": 1,
            "val_batch_size": 1,
            "test_batch_size": 1,
            "num_workers": 16,
        },
        "model": {
            "name": "gluefactory.models.two_view_pipeline",
            "ground_truth": {
                "name": "gluefactory.models.matchers.depth_matcher",
                "use_lines": False,
            },
            "run_gt_in_forward": True,
        },
        "eval": {"plot_methods": [], "plot_line_methods": [], "eval_lines": False},
    }

    export_keys = [
        "gt_matches0",
        "matches0",
        "matching_scores0",
    ]

    optional_export_keys = [
        "gt_line_matches0",
        "line_matches0",
        "line_matching_scores0",
    ]

    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf is not None else self.default_conf["data"]
        dataset = get_dataset("eth3d")(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file):
        eval_conf = self.conf.eval
        r = eval_dataset(loader, pred_file)
        if self.conf.eval.eval_lines:
            r.update(eval_dataset(loader, pred_file, conf=eval_conf, suffix="_lines"))
        s = {}

        return s, {}, r


def plot_pr_curve(
    models_name, results, dst_file="eth3d_pr_curve.pdf", title=None, suffix=""
):
    plt.figure()
    f_scores = np.linspace(0.2, 0.9, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color=[0, 0.5, 0], alpha=0.3)
        plt.annotate(
            "f={0:0.1}".format(f_score),
            xy=(0.9, y[45] + 0.02),
            alpha=0.4,
            fontsize=14,
        )

    plt.rcParams.update({"font.size": 12})
    # plt.rc('legend', fontsize=10)
    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.05, step=0.1), fontsize=16)
    plt.xlabel("Recall", fontsize=18)
    plt.ylabel("Precision", fontsize=18)
    plt.yticks(np.arange(0, 1.05, step=0.1), fontsize=16)
    plt.ylim([0.3, 1.0])
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for m, c in zip(models_name, colors):
        sAP_string = f'{m}: {results[m]["AP" + suffix]:.1f}'
        plt.plot(
            results[m]["curve_recall" + suffix],
            results[m]["curve_precision" + suffix],
            label=sAP_string,
            color=c,
        )

    plt.legend(fontsize=16, loc="lower right")
    if title:
        plt.title(title)

    plt.tight_layout(pad=0.5)
    print(f"Saving plot to: {dst_file}")
    plt.savefig(dst_file)
    plt.show()


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(ETH3DPipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = ETH3DPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval
    )

    # print results
    for k, v in r.items():
        if k.startswith("AP"):
            print(f"{k}: {v:.2f}")

    if args.plot:
        results = {}
        for m in conf.eval.plot_methods:
            exp_dir = output_dir / m
            results[m] = load_eval(exp_dir)[1]

        plot_pr_curve(conf.eval.plot_methods, results, dst_file="eth3d_pr_curve.pdf")
        if conf.eval.eval_lines:
            for m in conf.eval.plot_line_methods:
                exp_dir = output_dir / m
                results[m] = load_eval(exp_dir)[1]
            plot_pr_curve(
                conf.eval.plot_line_methods,
                results,
                dst_file="eth3d_pr_curve_lines.pdf",
                suffix="_lines",
            )
