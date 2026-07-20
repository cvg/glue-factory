"""
A set of utilities to manage and load checkpoints of training experiments.

Author: Paul-Edouard Sarlin (skydes)
"""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

import hydra
import omegaconf
import pkg_resources
import torch
from omegaconf import OmegaConf

from .. import models, settings
from . import misc

logger = logging.getLogger(__name__)


def list_configs(configs_path: Path) -> list[str]:
    """List all available configs in a given directory."""
    return list(sorted([x.stem for x in Path(configs_path).glob("*.yaml")]))


def parse_config_path(
    name_or_path: Optional[str], default_config_dir: str = "configs/"
) -> Path:
    default_configs = {}
    for c in pkg_resources.resource_listdir("gluefactory", str(default_config_dir)):
        if c.endswith(".yaml"):
            default_configs[Path(c).stem] = Path(
                pkg_resources.resource_filename("gluefactory", default_config_dir + c)
            )
    if name_or_path is None:
        return None
    if name_or_path in default_configs:
        return default_configs[name_or_path]
    path = Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find the config file: {name_or_path}. "
            f"Not in the default configs {list(default_configs.keys())} "
            "and not an existing path."
        )
    return Path(path)


def load_sweep_config(
    sweep_confs: omegaconf.ListConfig,
    sweep_idx: int,
) -> omegaconf.DictConfig:
    conf = {}
    for sweep_conf in sweep_confs:
        for k, v in sweep_conf.items():
            if len(v) == 1:
                conf[k] = v[0]
            else:
                conf[k] = v[sweep_idx]
    logger.info(f"Sweep config: {conf}")
    conf = misc.unflatten_dict(conf)
    return OmegaConf.create(conf)


def compose_config(
    name_or_path: Optional[str],
    default_config_dir: str = "configs/",
    overrides: Optional[list[str]] = None,
    sweep_idx: int | None = None,
    resolve: bool = True,
) -> tuple[Path, OmegaConf]:

    conf_path = parse_config_path(name_or_path, default_config_dir)
    logger.info(f"Hydra config directory: {conf_path.parent}.")

    # pathlib does not support walk_up with python < 3.12, so use os.path.relpath
    rel_conf_dir = Path(os.path.relpath(conf_path.parent, Path(__file__).parent))
    with hydra.initialize(version_base=None, config_path=str(rel_conf_dir)):
        custom_conf = hydra.compose(config_name=conf_path.stem, overrides=overrides)

    if sweep_idx is not None:
        logger.info(f"Loading sweep {sweep_idx}.")
        sweep_conf = load_sweep_config(custom_conf.get("sweep", []), sweep_idx)
        OmegaConf.set_struct(custom_conf, False)
        custom_conf = OmegaConf.merge(custom_conf, sweep_conf)
        del custom_conf["sweep"]
    if resolve:
        OmegaConf.resolve(custom_conf)
    return conf_path, custom_conf


def list_checkpoints(dir_):
    """List all valid checkpoints in a given directory."""
    checkpoints = []
    for p in dir_.glob("checkpoint_*.tar"):
        numbers = re.findall(r"(\d+)", p.name)
        assert len(numbers) <= 2
        if len(numbers) == 0:
            continue
        else:
            checkpoints.append((int(numbers[0]), p))
    return checkpoints


def get_last_checkpoint(exper, allow_interrupted=True):
    """Get the last saved checkpoint for a given experiment name."""
    if Path(exper).exists():
        experiment_dir = Path(exper)
    else:
        experiment_dir = Path(settings.TRAINING_PATH, exper)
    ckpts = list_checkpoints(experiment_dir)
    if not allow_interrupted:
        ckpts = [(n, p) for (n, p) in ckpts if "_interrupted" not in p.name]
    assert len(ckpts) > 0
    return sorted(ckpts)[-1][1]


def get_best_checkpoint(exper):
    """Get the checkpoint with the best loss, for a given experiment name."""
    if Path(exper).exists():
        experiment_dir = Path(exper)
    else:
        experiment_dir = Path(settings.TRAINING_PATH, exper)
    p = experiment_dir / "checkpoint_best.tar"
    return p


def delete_old_checkpoints(dir_, num_keep):
    """Delete all but the num_keep last saved checkpoints."""
    ckpts = list_checkpoints(dir_)
    ckpts = sorted(ckpts)[::-1]
    kept = 0
    for ckpt in ckpts:
        if ("_interrupted" in str(ckpt[1]) and kept > 0) or kept >= num_keep:
            logger.info(f"Deleting checkpoint {ckpt[1].name}")
            ckpt[1].unlink()
        else:
            kept += 1


def load_checkpoint(
    ckpt: Path | str,
    weights_only: bool = not settings.ALLOW_PICKLE,
    map_location="cpu",
    **kwargs,
) -> dict[str, Any]:
    ckpt = torch.load(
        str(ckpt), map_location=map_location, weights_only=weights_only, **kwargs
    )
    # Fix distributed model loading
    ckpt["model"] = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    # Fix compiled model loading (old)
    ckpt["model"] = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    return ckpt


def load_experiment(
    exper, conf={}, get_last=False, ckpt=None, weights_only=not settings.ALLOW_PICKLE
):
    """Load and return the model of a given experiment."""
    exper = Path(exper)
    if exper.suffix != ".tar":
        if get_last:
            ckpt = get_last_checkpoint(exper)
        else:
            ckpt = get_best_checkpoint(exper)
    else:
        ckpt = exper
    logger.info(f"Loading checkpoint {ckpt.name}")
    ckpt = load_checkpoint(ckpt, weights_only=weights_only, map_location="cpu")

    loaded_conf = OmegaConf.create(ckpt["conf"])
    OmegaConf.set_struct(loaded_conf, False)
    conf = OmegaConf.merge(loaded_conf.model, OmegaConf.create(conf))
    model = models.get_model(conf.name)(conf).eval()

    state_dict = ckpt["model"]
    dict_params = set(state_dict.keys())
    model_params = set(map(lambda n: n[0], model.named_parameters()))
    diff = model_params - dict_params
    logger.warning(f"Missing {len(diff)} / {len(model_params)} parameters.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}."
    )
    return model


# @TODO: also copy the respective module scripts (i.e. the code)
def save_experiment(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    conf: OmegaConf,
    results: dict[str, float],
    epoch: int,
    iter_i: int,
    output_dir: Path,
    stop: bool = False,
    distributed: bool = False,
    cp_name: Optional[str] = None,
    best_eval: float | None = None,  # This activates auto early stopping
    custom: dict[str, Any] | None = None,
):
    """Save the current model to a checkpoint
    and return the best result so far."""
    state = (model.module if distributed else model).state_dict()
    checkpoint = {
        "model": state,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "conf": OmegaConf.to_container(conf, resolve=True),
        "epoch": epoch,
        "eval": results,
        **(custom or {}),
    }
    if cp_name is None:
        cp_name = (
            f"checkpoint_{epoch}_{iter_i}" + ("_interrupted" if stop else "") + ".tar"
        )
    logger.info(f"Saving checkpoint {cp_name}")
    cp_path = str(output_dir / cp_name)
    torch.save(checkpoint, cp_path)
    if best_eval is not None and results[conf.train.best_key] < best_eval:
        best_eval = results[conf.train.best_key]
        logger.info(f"New best val: {conf.train.best_key}={best_eval}")
        shutil.copy(cp_path, str(output_dir / "checkpoint_best.tar"))
    elif best_eval is None and cp_name != "checkpoint_best.tar":
        shutil.copy(cp_path, str(output_dir / "checkpoint_best.tar"))
    delete_old_checkpoints(output_dir, conf.train.keep_last_checkpoints)
    return best_eval


# Adaptation of torch.profiler.tensorboard_trace_handler
def tensorboard_trace_handler(
    dir_name: str,
    worker_name: Optional[str] = None,
    use_gzip: bool = False,
    epoch: int | None = None,
):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import os
    import socket
    import time

    def file_replace(filepath, pattern, target):
        # Read in the file
        with open(filepath, "r") as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace(pattern, target)

        # Write the file out again
        with open(filepath, "w") as file:
            file.write(filedata)

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        if epoch is not None:
            worker_name += f"_epoch{epoch}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        file_path = os.path.join(dir_name, file_name)
        prof.export_chrome_trace(file_path)

        # Fix problem with tb profile: https://github.com/pytorch/kineto/issues/688
        file_replace(file_path, '"user_annotation', '"cpu_op')

        if use_gzip:
            import gzip

            with open(file_path, "rb") as fin:
                with gzip.open(file_path + ".gz", "wb") as fout:
                    fout.writelines(fin)
            os.remove(file_path)

    return handler_fn


def get_ablation_dir(output_dir: Path, mkdir: bool = False) -> Path:
    subdirs = [int(p.stem) for p in output_dir.glob("*/") if p.stem.isdigit()]
    ablate_id = max(subdirs, default=-1) + 1
    output_dir = output_dir / str(ablate_id)
    if mkdir:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
