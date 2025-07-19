import argparse
import logging
from pathlib import Path
from pprint import pprint

from omegaconf import OmegaConf

from .. import settings
from ..models import get_model
from ..utils import experiments

logger = logging.getLogger(__name__)


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

    if args.tag:
        name = args.tag
    elif args.conf and checkpoint_name:
        name = f"{args.conf}_{checkpoint_name}"
    elif args.conf:
        name = args.conf
    elif checkpoint_name:
        name = checkpoint_name
    if len(args.dotlist) > 0 and not args.tag:
        name = name + "_" + ":".join(args.dotlist)
    logger.info("Running benchmark: %s", benchmark)
    logger.info("Experiment tag: %s", name)
    logger.info("Config:")
    pprint(OmegaConf.to_container(conf))
    return name, conf


def load_model(model_conf, checkpoint):
    if checkpoint:
        model = experiments.load_experiment(checkpoint, conf=model_conf).eval()
    else:
        model = get_model("two_view_pipeline")(model_conf).eval()
    if not model.is_initialized():
        raise ValueError(
            "The provided model has non-initialized parameters. "
            + "Try to load a checkpoint instead."
        )
    return model


def get_eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_eval", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("dotlist", nargs="*")
    return parser
