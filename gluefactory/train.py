"""
Simple training launcher.

Author: Paul-Edouard Sarlin (skydes), Philipp Lindenberger (Phil26AT)
"""

import argparse
import shutil
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from . import __module_name__, logger, settings, trainer
from .utils import experiments, stdout_capturing


def init_process(output_dir: Path, rank: int, world_size: int) -> int:
    assert torch.cuda.is_available(), "Distributed training requires CUDA"
    logger.info(f"Training in distributed mode with {world_size} GPUs")
    device = rank
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        init_method="file://" + str(output_dir / "distributed_lock"),
    )
    torch.cuda.set_device(device)
    return device


def create_training_dir(experiment_name: str, args) -> Path:
    import shutil

    output_dir = settings.TRAINING_PATH / experiment_name
    if args.ablate:
        subdirs = [int(p.stem) for p in output_dir.glob("*/") if p.stem.isdigit()]
        ablate_id = max(subdirs, default=-1) + 1
        output_dir = output_dir / str(ablate_id)
        logger.info(f"Creating ablation folder {output_dir}")
    # Setup output directory
    exist_ok = args.restore or args.overwrite or args.clean
    if output_dir.exists() and not exist_ok:
        raise FileExistsError(
            f"Output directory {output_dir} already exists. "
            "Use --restore to continue training or --overwrite to delete it."
        )
    if output_dir.exists() and args.clean:
        logger.info(f"Cleaning output directory {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=exist_ok, parents=True)
    logger.info(f"Output directory: {output_dir}")
    return output_dir


def compose_cli_config(output_dir: Path, args) -> DictConfig:
    conf = OmegaConf.from_cli(args.dotlist)
    OmegaConf.save(conf, str(output_dir / "cli_config.yaml"))
    if args.conf:
        conf_path, raw_conf = experiments.compose_config(args.conf)
        OmegaConf.set_struct(raw_conf, args.strict)
        conf = OmegaConf.merge(raw_conf, conf)
        # Copy a more readable config file to the output dir
        shutil.copy(conf_path, output_dir / "raw_config.yaml")
    if args.restore:
        restore_conf = OmegaConf.load(output_dir / "config.yaml")
        conf = OmegaConf.merge(restore_conf, conf)
        conf.train.load_experiment = args.experiment
        conf.train.load_state = True
    else:
        if conf.train.seed is None:
            conf.train.seed = torch.initial_seed() & (2**32 - 1)
    OmegaConf.save(conf, str(output_dir / "config.yaml"))
    return conf


def main_worker(rank, conf, output_dir, args):
    distributed = conf.train.num_devices > 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if rank == 0 and not args.quiet:
        logger.info(
            "Starting training with configuration:\n%s", OmegaConf.to_yaml(conf)
        )
    if distributed:
        device = init_process(output_dir, rank, conf.train.num_devices)
    if rank == 0:
        with stdout_capturing.capture_outputs(
            output_dir / "log.txt", cleanup_interval=args.cleanup_interval
        ):
            res = trainer.launch_training(output_dir, conf, device)
    else:
        res = trainer.launch_training(output_dir, conf, device)

    if distributed:
        # dist.barrier()
        dist.destroy_process_group()
    return res


def save_code_snapshot(
    output_dir: Path,
    extra_module_names: Sequence[str] = (),
    compression: str | None = "zip",
):
    """Create a snapshot of the codebase."""
    for module in [__module_name__] + list(extra_module_names):
        mod_dir = Path(__import__(str(module)).__file__).parent
        if compression:
            shutil.make_archive(output_dir / module, compression, mod_dir)
        else:
            shutil.copytree(mod_dir, output_dir / module, dirs_exist_ok=True)


def parse_args():
    """Parse command line arguments and return them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    default_config_names = experiments.list_configs(Path(__file__).parent / "configs")
    parser.add_argument(
        "--conf",
        type=str,
        help=f"Configuration path (.yaml) or one of: {default_config_names}",
    )
    parser.add_argument(
        "--cleanup_interval",
        default=120,  # Cleanup log files every 120 seconds.
        type=int,
        help="Interval in seconds to cleanup log files",
    )
    parser.add_argument(
        "--restore", action="store_true", help="Restore from previous experiment"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing experiment directory",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory if it exists",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Create an ablation folder (/XID) that increments on each run",
    )
    parser.add_argument(
        "--compress_snapshot",
        "--cs",
        type=str,
        default="zip",
    )
    parser.add_argument("--quiet", action="store_true", help="Mute some logging")
    parser.add_argument(
        "--distributed", action="store_true", help="Run in distributed mode"
    )
    parser.add_argument("--strict", action="store_true", help="Strict config merge")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    return args


if __name__ == "__main__":
    # Start actual training

    args = parse_args()
    output_dir = create_training_dir(args.experiment, args)
    conf = compose_cli_config(output_dir, args)
    conf.train.num_devices = conf.train.get("num_devices", 0)

    save_code_snapshot(
        output_dir,
        conf.train.get("submodules", ()),
        compression=args.compress_snapshot,
    )

    # Start actual training
    if args.distributed and conf.train.num_devices < 1:
        conf.train.num_devices = torch.cuda.device_count()

    if conf.train.num_devices > 0:
        assert torch.cuda.is_available(), "Distributed training requires CUDA"
        args.lock_file = output_dir / "distributed_lock"
        if args.lock_file.exists():
            args.lock_file.unlink()
        torch.multiprocessing.spawn(
            main_worker, nprocs=conf.train.num_devices, args=(conf, output_dir, args)
        )
    else:
        main_worker(0, conf, output_dir, args)
