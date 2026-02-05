import argparse
import logging
import pprint
import re
from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from .. import settings
from ..models import get_model
from ..utils import experiments

logger = logging.getLogger(__name__)


def format_summaries(summaries: dict) -> dict:
    """Format summaries, grouping keys that differ only by a number.

    Returns a dict with grouped keys like 'metric<[1,5,10]px' mapping to lists of values.
    """
    grouped = defaultdict(dict)
    standalone = {}
    # Match base@number, base<number, base>number, base_pN patterns
    pattern = re.compile(r"^(.+?)([@<>]|_p)(\d+(?:\.\d+)?)(.*?)$")

    for k, v in summaries.items():
        match = pattern.match(k)
        if match:
            base, sep, num, suffix = match.groups()
            grouped[(base, sep, suffix)][float(num)] = v
        else:
            standalone[k] = v

    result = {}
    for (base, sep, suffix), values in sorted(grouped.items()):
        nums = sorted(values.keys())
        nums_str = ",".join(str(int(n) if n == int(n) else n) for n in nums)
        key = f"{base}{sep}[{nums_str}]{suffix}"
        result[key] = [values[n] for n in nums]
    result.update(standalone)
    return result


def extract_benchmark_conf(conf, benchmark):
    mconf = OmegaConf.create(
        {
            "model": conf.get("model", {}),
        }
    )
    if "benchmarks" in conf.keys():
        return OmegaConf.merge(mconf, conf.benchmarks.get(benchmark, {}))
    else:
        return mconf


def parse_eval_args(benchmark, args, configs_path, default=None):
    conf = {"data": {}, "model": {}, "eval": {}}
    if args.conf:
        conf_path, custom_conf = experiments.compose_config(
            args.conf, default_config_dir=configs_path
        )
        conf = extract_benchmark_conf(OmegaConf.merge(conf, custom_conf), benchmark)
        args.tag = args.tag if args.tag is not None else conf_path.stem

    cli_conf = OmegaConf.from_cli(args.dotlist)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.checkpoint = args.checkpoint if args.checkpoint else conf.get("checkpoint")

    checkpoint_name = conf.checkpoint
    if conf.checkpoint and not conf.checkpoint.endswith(".tar"):
        if Path(conf.checkpoint).exists():
            checkpoint_dir = Path(conf.checkpoint).absolute()
            if checkpoint_dir.is_relative_to(settings.TRAINING_PATH):
                checkpoint_name = str(
                    checkpoint_dir.relative_to(settings.TRAINING_PATH)
                )
        else:
            checkpoint_dir = settings.TRAINING_PATH / conf.checkpoint
        checkpoint_conf = OmegaConf.load(checkpoint_dir / "config.yaml")
        conf = OmegaConf.merge(extract_benchmark_conf(checkpoint_conf, benchmark), conf)

    if default:
        conf = OmegaConf.merge(default, conf)

    name = ""
    if args.tag:
        name = args.tag
    elif args.conf and checkpoint_name:
        name = f"{args.conf}_{checkpoint_name}"
    elif args.conf:
        name = args.conf
    elif checkpoint_name:
        name = checkpoint_name

    cli_args = [k for k in args.dotlist if not k.startswith("num_samples=")]
    if len(cli_args) > 0 and not args.tag:
        # Append CLI args to name except num_samples
        name = name + "_" + ":".join(cli_args)

    if not name:
        raise ValueError("No tag provided. Please provide a tag with --tag or --conf.")
    logger.info("Running benchmark: %s", benchmark)
    logger.info("Experiment tag: %s", name)
    logger.info("Config:")
    logger.info(pprint.pformat(OmegaConf.to_container(conf)))
    return name, conf


def load_model(model_conf, checkpoint):
    if checkpoint:
        model = experiments.load_experiment(checkpoint, conf=model_conf).eval()
    else:
        model = get_model(model_conf.name)(model_conf).eval()
    if not model.is_initialized():
        raise ValueError(
            "The provided model has non-initialized parameters. "
            + "Try to load a checkpoint instead."
        )
    return model


def get_eval_parser(parser: argparse.ArgumentParser | None = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_eval", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("dotlist", nargs="*")
    return parser


def run_cli(eval_cls, name: str, parser: argparse.ArgumentParser | None = None):
    """Run the evaluation pipeline from the command line."""
    parser = parser if parser is not None else get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(eval_cls.default_conf)
    benchmark = name

    # mingle paths
    output_dir = Path(settings.EVAL_PATH, name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True, parents=True)

    pipeline = eval_cls(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    logger.info("Evaluation summaries:\n%s", pprint.pformat(format_summaries(s)))

    if args.plot:
        for fig_name, fig in f.items():
            fig.canvas.manager.set_window_title(fig_name)
        plt.show()

    logger.info(
        f"To inspect results: python -m gluefactory.eval.inspect {benchmark} {name}"
    )

    return s, f, r
