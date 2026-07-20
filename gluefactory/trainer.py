"""
A generic, flexible trainer.

Author: Philipp Lindenberger
"""

import collections
import gc
import shutil
import signal
from pathlib import Path
from typing import Any, Callable, Sequence, TypeAlias

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gluefactory import datasets, models
from gluefactory.models import BaseModel
from gluefactory.utils import experiments, lr_schedule, misc, tools
from gluefactory.utils.summary_writer import SummaryWriter

from . import __module_name__, eval, logger, settings

Args: TypeAlias = DictConfig
Batch: TypeAlias = Any
Predictions: TypeAlias = Any
LossMetrics: TypeAlias = dict[str, torch.Tensor]
Writer: TypeAlias = SummaryWriter | None


def apply_batch_mask(
    pred: dict,
    data: dict,
    values: LossMetrics | Sequence[LossMetrics],
    key: str,
    exclude: tuple[str, ...],
) -> tuple[torch.Tensor, LossMetrics | Sequence[LossMetrics]]:
    """Zero out per-sample values where a batch mask is False.

    Returns the boolean mask and the filtered values dict.
    Entries whose key contains any of the exclude patterns are left untouched.
    """
    if key in data:
        mask = data[key].bool()
    elif key in pred:
        mask = pred[key].bool()
    else:
        raise KeyError(f"Batch mask key '{key}' not found in data or predictions.")

    def filter_values(v: LossMetrics) -> LossMetrics:
        return {
            k: (
                v
                if any(p in k for p in exclude) or not isinstance(v, torch.Tensor)
                else torch.where(mask, v, torch.zeros_like(v))
            )
            for k, v in v.items()
        }

    if isinstance(values, dict):
        values = filter_values(values)
    else:
        values = [filter_values(v) for v in values]
    return mask, values


def compose_loss(
    loss_dict: LossMetrics, compose_str: str, allow_missing: bool = False
) -> torch.Tensor:
    """Compose a loss from a string, e.g. '1.0*loss1 + 0.1*loss2'."""
    # @TODO: Support multiplicative loss terms
    addition_terms = compose_str.split("+")
    loss = 0.0
    for term in addition_terms:
        term = term.strip()
        if "*" in term:
            weight_str, key = term.split("*")
            weight = float(weight_str)
        else:
            weight = 1.0
            key = term
        key = key.strip()
        if key not in loss_dict:
            if not allow_missing:
                raise KeyError(f"Key {key} not found in loss dict.")
            continue
        val = loss_dict[key]
        if weight == 0.0:
            val = val.nan_to_num(0.0)
        loss = loss + weight * val
    return loss


def eval_model(
    model: torch.nn.parallel.DistributedDataParallel | BaseModel,
) -> BaseModel:
    model.eval()  # Set to eval mode to disable training-specific behavior (e.g. sync_bn stats update)
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    model = model.module if is_ddp else model  # Get the original model
    return model


@torch.compiler.set_stance("force_eager")
@torch.no_grad()
def run_evaluation(
    model: BaseModel | torch.nn.parallel.DistributedDataParallel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    conf: DictConfig,
    rank: int = 0,
    pbar: bool = True,
    max_iters: int | None = None,
    compose_loss_str: str | None = None,
) -> tuple[Any, ...]:
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    model = eval_model(model)
    results = {}
    pr_metrics = collections.defaultdict(tools.PRMetric)
    figures = []
    plot_ids = np.random.choice(
        len(loader), min(len(loader), conf.num_eval_plots), replace=False
    )
    max_iters = max_iters or len(loader)
    max_iters = min(max_iters, len(loader))
    eval_iter = iter(loader)
    for i in tqdm(range(len(loader)), desc="Evaluation", ascii=True, disable=not pbar):
        if i >= max_iters:
            break
        data = next(eval_iter)
        data = misc.batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses, metrics = model.loss_metrics(pred, data)
            if conf.get("batch_mask_key", None) is not None:
                exclude = conf.get("batch_mask_exclude", ())
                mask, (losses, metrics) = apply_batch_mask(
                    pred, data, (losses, metrics), conf.batch_mask_key, exclude
                )
                metrics["batch_mask"] = mask.float()
            pr_metrics_i = model.pr_metrics(pred, data)
            losses, metrics, pr_metrics_i = [
                misc.batch_to_device(x, "cpu", non_blocking=False)
                for x in (losses, metrics, pr_metrics_i)
            ]
            if is_ddp:
                # Gather all losses and metrics
                losses = misc.tree_all_gather(losses)
                metrics = misc.tree_all_gather(metrics)
                pr_metrics_i = misc.tree_all_gather(pr_metrics_i)
            if compose_loss_str is not None:
                losses["total"] = compose_loss(losses, compose_loss_str)
            if i in plot_ids:
                figures.append(model.visualize(pred, data))
            # add PR curves
            for k, (labels, preds) in pr_metrics_i.items():
                pr_metrics[k].update(labels, preds)
        del pred, data
        numbers = {**metrics, **{"loss/" + k: v for k, v in losses.items()}}
        exclude = (*conf.get("batch_mask_exclude", ()), "batch_mask")
        for k, v in numbers.items():
            if k not in results:
                results[k] = tools.AverageMetric()
                if k in conf.median_metrics:
                    results[k + "_median"] = tools.MedianMetric()
                if k in conf.recall_metrics.keys():
                    q = conf.recall_metrics[k]
                    results[k + f"_recall{int(q)}"] = tools.RecallMetric(q)
            m = None
            if (
                "batch_mask" in numbers
                and not any(p in k for p in exclude)
                and v.shape[0] == numbers["batch_mask"].shape[0]
            ):
                m = numbers["batch_mask"]
            results[k].update(v, mask=m)
            if k in conf.median_metrics:
                results[k + "_median"].update(v)
            if k in conf.recall_metrics.keys():
                q = conf.recall_metrics[k]
                results[k + f"_recall{int(q)}"].update(v)
        del numbers
    results = {k: results[k].compute() for k in results}
    pr_metrics = {k: v.compute() for k, v in pr_metrics.items()}
    return results, pr_metrics, figures


class Trainer:
    """
    Trainer class for managing the training process.

    Maintains model, params and training state (optim, step, ...)
    """

    default_conf = {
        "seed": "???",  # training seed
        "epochs": 1,  # number of epochs
        "optimizer": "adam",  # name of optimizer in [adam, sgd, rmsprop]
        "opt_regexp": None,  # regular expression to filter parameters to optimize
        "optimizer_options": {},  # optional arguments passed to the optimizer
        "lr": 0.001,  # learning rate
        "lr_schedule": {
            "type": None,  # string in {factor, exp, member of torch.optim.lr_scheduler}
            "start": 0,
            "exp_div_10": 0,
            "on_epoch": False,
            "factor": 1.0,
            "options": {},  # add lr_scheduler arguments here
            "warmup": 0.0,  # linear warmup: epochs if on_epoch, iterations if not
        },
        "lr_scaling": {},  # learning rate scaling for parameter name patterns
        "freeze_epochs": {},  # parameter name patterns -> epochs to keep frozen (lr=0)
        "eval_every_epoch": None,  # interval for evaluation on the validation set
        "eval_init": False,  # run evaluation on the validation set before training
        "train_split": "train",  # split to use for training
        "eval_split": "val",  # split to use for evaluation
        "benchmark_every_epoch": 1,  # interval for evaluation on the test benchmarks
        "benchmark_checkpoint": True,  # use the best checkpoint for benchmark evaluation (otherwise current model)
        "save_every_iter": 5000,  # interval for saving the current checkpoint
        "log_every_iter": 200,  # interval for logging the loss to the console
        "log_grad_every_iter": None,  # interval for logging gradient hists
        "keep_last_checkpoints": 1,  # keep only the last X checkpoints
        "load_experiment": None,  # initialize the model from a previous experiment
        "median_metrics": [],  # add the median of some metrics
        "recall_metrics": {},  # add the recall of some metrics
        "best_key": "loss/total",  # key to use to select the best checkpoint
        "clip_grad": None,
        "pr_curves": {},  # add pr curves, set labels/predictions/mask keys
        "num_eval_plots": 4,  # Number of plots to show during evaluation (0=skip)
        "plot_every_iter": None,  # plot figures every X iterations
        "submodules": [],
        "mixed_precision": None,
        "num_devices": 0,  # 0 means sequential.
        "compile": None,  # Compilation mode for the model. [None, default, ...]
        "profile": None,  # Profile the training with PyTorch profiler (# prof steps)
        "profile_every_epoch": None,  # Profile every N epochs, None means only first
        "record_memory": None,  # Record memory usage during training (# record steps)
        "log_it": False,  # Log tensorboard on iteration (default is num_samples)
        "print_arch": False,  # Print the model architecture
        "train_iters": None,
        "eval_iters": None,
        "stdout_metrics": [
            "loss/total"
        ],  # List of metrics to print to stdout (None=all)
        "detect_anomaly": False,  # Enable anomaly detection
        "matmul_precision": None,  # Set torch.matmul precision [None, highest, high, medium, low]
        "gradient_accumulation_steps": 1,  # Accumulate gradients over N steps
        "ddp_find_unused_parameters": False,  # DDP find_unused_parameters
        "overfit": False,  # Overfit a single batch
        "stop_immediately": False,  # Stop training immediately on SIGINT
        "run_benchmarks": (),
        "writer": "tensorboard",  # options: [tensorboard, wandb]
        "project_name": __module_name__,  # wandb project name
        "finetune_after": [],  # epochs at which to apply finetune_scales
        "finetune_scales": {},  # {dotted.key: scale} applied to dataset conf
        "batch_mask_key": None,  # data key (bool B) to mask losses per sample
        "batch_mask_exclude": (),  # loss name patterns excluded from masking
    }

    def __init__(
        self,
        conf: DictConfig,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        epoch_tracker: lr_schedule.EpochTracker | None = None,
        device: torch.device | str | None = None,
    ):
        # Initialize conf, model, optimizer and LR
        self.default_conf = OmegaConf.create(self.default_conf)
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch_tracker = epoch_tracker or lr_schedule.EpochTracker()

        # Setup distributed
        self.distributed = conf.num_devices > 0
        self.num_gpus = conf.num_devices or 1

        # Initialize rank and device
        if self.distributed:
            assert dist.is_initialized(), "Torch Distributed not initialized"
        self.rank = dist.get_rank() if self.distributed else 0

        # Initialize device
        self.device = device
        if self.device is None:
            self.device = (
                self.rank
                if self.distributed
                else "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Initialize model params and conf
        self.model_conf = self.model.conf
        if self.conf.print_arch:
            self.info("Model architecture:\n%s", str(self.model))

        # Setup scaler and dtype
        self.use_mp = self.setup_dtype_scaler(conf.mixed_precision)

        # Initialize step timer
        self.step_timer = tools.StepTimer()

        # Initialize rank and device
        if self.distributed:
            assert dist.is_initialized(), "Torch Distributed not initialized"
        self.rank = dist.get_rank() if self.distributed else 0

        # Initialize device
        self.device = device
        if self.device is None:
            self.device = (
                self.rank
                if self.distributed
                else "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Setup counters
        self.epoch = 0
        self.tot_n_samples = 0
        self.tot_it = 0

        # Handle KeyboardInterrupt
        self.setup_sigint_handler()

        # ToDo: Maybe call from outer scope after checkpoint init
        self.prepare_model()

        # Named benchmark configs
        self.benchmarks = {}

        # Setup torch global variables
        self.setup_torch()

    # ------------------------------------------------------------------------
    # Utility Initializers
    # ------------------------------------------------------------------------

    @classmethod
    def init(
        cls,
        conf: DictConfig,
        model: BaseModel,
        **kwargs,
    ) -> "Trainer":
        """Create a Trainer instance from a config."""

        conf = OmegaConf.merge(cls.default_conf, conf)
        optimizer = cls.construct_optimizer(conf, model)
        epoch_tracker = lr_schedule.EpochTracker()
        lr_scheduler = lr_schedule.get_lr_scheduler(
            optimizer=optimizer, conf=conf.lr_schedule, epoch_tracker=epoch_tracker
        )
        return cls(
            conf=conf,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch_tracker=epoch_tracker,
            **kwargs,
        )

    # ------------------------------------------------------------------------
    # Setup helper functions (public)
    # ------------------------------------------------------------------------

    def register_benchmark(
        self, benchmark_name: str, benchmark_conf: str, every_epoch: int | None = None
    ):
        every_epoch = every_epoch or self.conf.benchmark_every_epoch
        self.benchmarks[benchmark_name] = (benchmark_conf, every_epoch)

    def sequential_model(self) -> BaseModel:
        """Get the original model (without DDP)."""
        return (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )

    def load_checkpoint(
        self,
        checkpoint: Any,
        strict: bool = True,
        load_state: bool = False,
        load_modelconfig: bool = False,
        load_weights: bool = True,
    ):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            # Fix distributed model naming
            checkpoint["model"] = {
                (k.replace("module.", "") if k.startswith("module.") else k): v
                for k, v in checkpoint["model"].items()
            }
        if load_weights:
            missing, unexpected = self.model.load_state_dict(
                checkpoint["model"], strict=strict
            )
            self.info(
                f"state_dict loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}."
            )
        if load_modelconfig:
            self.conf.model = OmegaConf.merge(
                OmegaConf.create(checkpoint["conf"]).model, self.conf.model
            )
            self.info("Model config loaded.")
        if load_state:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "lr_scheduler" in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"]
            for metric in ["tot_it", "tot_n_samples"]:
                if metric in checkpoint:
                    setattr(self, metric, checkpoint[metric])
                    self.info(
                        f"Loaded {metric}={getattr(self, metric)} ({checkpoint[metric]})"
                    )
            if "epoch_tracker_fractional_epoch" in checkpoint:
                self.epoch_tracker.fractional_epoch = checkpoint[
                    "epoch_tracker_fractional_epoch"
                ]
            self.info(f"Training state loaded. Resuming at epoch {self.epoch}.")

    def maybe_load_checkpoint(self):
        if self.conf.load_experiment:
            if self.conf.get("load_last_checkpoint", False):
                self.info(
                    "Loading last checkpoint from experiment %s",
                    self.conf.load_experiment,
                )
                init_cp = experiments.get_last_checkpoint(self.conf.load_experiment)
            else:
                self.info(
                    "Loading best checkpoint from experiment %s",
                    self.conf.load_experiment,
                )
                init_cp = experiments.get_best_checkpoint(self.conf.load_experiment)
            self.info("Loading checkpoint %s", str(init_cp))
            init_cp = experiments.load_checkpoint(
                init_cp,
                map_location="cpu",
                weights_only=self.conf.get(
                    "load_weights_only", not settings.ALLOW_PICKLE
                ),
            )
            self.load_checkpoint(
                init_cp,
                load_state=self.conf.get("load_state", False),
                load_modelconfig=self.conf.get("load_modelconfig", False),
                strict=self.conf.get("load_strict", True),
                load_weights=self.conf.get("load_weights", True),
            )

    def save_checkpoint(
        self,
        output_dir: Path,
        conf: DictConfig,  # This is the full conf!
        results: dict | None = None,
        iter_i: int = 0,
        **kwargs,
    ) -> int | None:
        if self.rank == 0:
            try:
                return experiments.save_experiment(
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    conf,
                    results,
                    iter_i=iter_i,
                    epoch=self.epoch,
                    output_dir=output_dir,
                    custom={
                        "tot_it": self.tot_it,
                        "tot_n_samples": self.tot_n_samples,
                        "epoch_tracker_fractional_epoch": self.epoch_tracker.fractional_epoch,
                    },
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}. Continue.")

    # ------------------------------------------------------------------------
    # Setup helper functions (internal)
    # ------------------------------------------------------------------------

    def info(self, pattern: str, *args, **kwargs):
        if self.rank == 0:
            logger.info(pattern, *args, **kwargs)

    def warn(self, pattern: str, *args, **kwargs):
        if self.rank == 0:
            logger.warning(pattern, *args, **kwargs)

    def learning_rate_step(self, verbose: bool = False):
        old_lr = self.optimizer.param_groups[0]["lr"]
        self.epoch_tracker.step()
        self.lr_scheduler.step()
        for group in self.optimizer.param_groups:
            if self.epoch < group.get("freeze_until", 0):
                group["lr"] = 0.0
        if verbose:
            self.info(
                f'lr changed from {old_lr} to {self.optimizer.param_groups[0]["lr"]}'
            )

    def _apply_finetune_scales(self, dataset):
        """Scale dataset config values at configured epochs.

        Keys in finetune_scales must start with "data." and are resolved
        against the dataset config. Only int and float values (or lists
        thereof) are supported.
        """
        if self.epoch not in self.conf.finetune_after:
            return
        conf = dataset.conf
        OmegaConf.set_readonly(conf, False)
        for key, scale in self.conf.finetune_scales.items():
            if not key.startswith("data."):
                raise ValueError(f"Finetune: key '{key}' must start with 'data.'")
            key = key[len("data.") :]
            old_value = OmegaConf.select(conf, key)
            if old_value is None:
                raise KeyError(f"Finetune: key '{key}' not found in dataset config")
            if isinstance(old_value, int):
                new_value = max(1, int(old_value * scale))
            elif isinstance(old_value, float):
                new_value = old_value * scale
            else:
                raise TypeError(
                    f"Finetune: unsupported type {type(old_value).__name__} "
                    f"for '{key}', expected int or float"
                )
            OmegaConf.update(conf, key, new_value)
            self.info(
                f"Finetune (epoch {self.epoch}): {key} {old_value} -> {new_value}"
            )
        OmegaConf.set_readonly(conf, True)

    def setup_sigint_handler(self):
        def sigint_handler(signal, frame):
            logger.info("Caught keyboard interrupt signal, will terminate")
            if self.stop or self.conf.stop_immediately:
                raise KeyboardInterrupt
            self.stop = True

        self.stop = False
        signal.signal(signal.SIGINT, sigint_handler)

    def setup_torch(self):
        torch.backends.cudnn.benchmark = True
        if self.conf.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        # TODO
        if self.conf.matmul_precision is not None:
            torch.set_float32_matmul_precision(self.conf.matmul_precision)

    def prepare_model(self):
        if self.conf.compile is not None:
            # Compile before DDP
            self.model = self.model.compile(mode=self.conf.compile)
        if self.distributed:
            self.model = self.model.make_ddp(
                device_ids=[self.device],
                find_unused_parameters=self.conf.ddp_find_unused_parameters,
            )

    def construct_profiler(
        self, output_dir: Path, store_raw_trace: bool = False
    ) -> torch.profiler.profile:
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=1,
                active=self.conf.profile,
                repeat=1,
                skip_first=10,
            ),
            on_trace_ready=experiments.tensorboard_trace_handler(
                str(output_dir), use_gzip=not store_raw_trace, epoch=self.epoch
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
        )

    @classmethod
    def construct_optimizer(
        cls, conf: DictConfig, model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        """Construct the optimizer for training."""
        optimizer_fn = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "rmsprop": torch.optim.RMSprop,
        }[conf.optimizer]

        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if conf.opt_regexp:
            params = tools.filter_parameters(params, conf.opt_regexp)
        lr_params = tools.pack_lr_parameters(
            params, conf.lr, conf.lr_scaling, conf.freeze_epochs
        )
        optimizer = optimizer_fn(lr_params, lr=conf.lr, **conf.optimizer_options)
        return optimizer

    def setup_dtype_scaler(self, mixed_precision: str | None) -> bool:
        use_mp = mixed_precision is not None
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=use_mp)
            if hasattr(torch.amp, "GradScaler")
            else torch.cuda.amp.GradScaler(enabled=use_mp)
        )
        self.info(f"Training with mixed_precision={mixed_precision}")

        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            None: torch.float32,  # we disable it anyway
            "float32": torch.float32,
            False: torch.float32,
        }[mixed_precision]

        return use_mp

    def get_writer(self, output_dir: Path, log_conf: DictConfig) -> Writer:
        if self.rank == 0:
            writer = SummaryWriter(
                log_dir=output_dir,
                writer=self.conf.writer,
                project=self.conf.project_name,
                conf=log_conf,
                # WandB options
                run_id=self.conf.get("run_id", None),
                name_as_run_id=self.conf.get("name_as_run_id", False),
                reload_run_id=self.conf.get("reload_run_id", True),
            )
        else:
            writer = None
        return writer

    # ------------------------------------------------------------------------
    # Logging functions
    # ------------------------------------------------------------------------

    @property
    def current_it(self):
        """Get the current iteration identifier."""
        return self.tot_it if self.conf.log_it else self.tot_n_samples

    def record_memory(self, output_dir: Path, it: int, offset: int = 2):
        if it == offset:
            self.info(
                f"Recording memory usage over {self.conf.record_memory} iterations "
                f"(skipped first {offset} it)."
            )
            torch.cuda.memory._record_memory_history(enabled="all")
        elif it == offset + self.conf.record_memory:
            # Record memory usage every self.conf.record_memory iterations
            snapshot_path = output_dir / f"memory_snapshot_epoch{self.epoch}.json"
            torch.cuda.memory._dump_snapshot(snapshot_path)

    def log_train(
        self,
        writer: Writer,
        it: int,
        train_loss_metrics: LossMetrics,
        extra_str: str = "",
    ):
        tot_n_samples = self.current_it
        all_params = self.model.parameters()
        writer.add_scalar("l2/param_norm", misc.param_norm(all_params), tot_n_samples)
        loss_metrics = {k: v.compute() for k, v in train_loss_metrics.items()}
        if self.conf.stdout_metrics is None:
            str_loss_metrics = [f"{k} {v:.3E}" for k, v in loss_metrics.items()]
        else:
            str_loss_metrics = [
                f"{k} {v:.3E}"
                for k, v in loss_metrics.items()
                if k in self.conf.stdout_metrics
            ]
        # Write training losses
        logger.info(
            "[E {} | it {}] {} loss {{{}}}".format(
                self.epoch, it, extra_str, ", ".join(str_loss_metrics)
            )
        )
        tools.write_dict_summaries(writer, "training", loss_metrics, tot_n_samples)
        writer.add_scalar(
            "training/lr", self.optimizer.param_groups[0]["lr"], tot_n_samples
        )

        # Write Epoch
        writer.add_scalar("training/epoch", self.epoch, tot_n_samples)

    def log_eval(self, writer: Writer, it: int, eval_results: Any):
        tot_n_samples = self.current_it
        results, pr_metrics, figures = eval_results
        str_results = [
            f"{k} {v:.3E}" for k, v in results.items() if isinstance(v, float)
        ]
        logger.info(f'[Validation] {{{", ".join(str_results)}}}')
        tools.write_dict_summaries(writer, "eval", results, tot_n_samples)
        tools.write_dict_summaries(writer, "eval", pr_metrics, tot_n_samples)
        tools.write_image_summaries(writer, "eval", figures, tot_n_samples)
        # @TODO: optional always save checkpoint

    def log_time_and_memory(
        self,
        writer: Writer,
        it: int,
        batch_size: int,
    ) -> str:
        tot_n_samples = self.current_it
        steps_per_sec = 0.0
        if self.step_timer.num_steps() > 1:
            step_duration, section_times = self.step_timer.compute()
            steps_per_sec = 1 / step_duration
            writer.add_scalar("step/total", step_duration, tot_n_samples)
            writer.add_scalar("step/_per_sec", steps_per_sec, tot_n_samples)
            writer.add_scalar(
                "step/_samples_per_sec",
                steps_per_sec * batch_size * self.num_gpus,
                tot_n_samples,
            )
            # Write section timings and fractions of step duration.
            for section_name, duration in section_times.items():
                writer.add_scalar(f"step/{section_name}", duration, tot_n_samples)

            writer.add_scalar(
                "step/io_fraction",
                (section_times["data"] + section_times["to_device"]) / step_duration,
                tot_n_samples,
            )

            if it % (self.conf.log_every_iter * 20) == 0:
                writer.add_figure(
                    "step/sections",
                    self.step_timer.plot(),
                    tot_n_samples,
                )

        # Reset the stats after logging
        self.step_timer.stats.clear()

        # Log memory stats
        memory_used, memory_total = 0.0, 0.0
        if torch.cuda.is_available():
            device_stats = tools.collect_device_stats()
            memory_used = device_stats["global_used"]
            memory_total = device_stats["global_total"]
            tools.write_dict_summaries(writer, "memory", device_stats, tot_n_samples)

        import psutil

        cpu_rss_gb = psutil.Process().memory_info().rss / 1024**3
        if writer is not None:
            writer.add_scalar("memory/cpu_rss_gb", cpu_rss_gb, tot_n_samples)

        return (
            f"[VRAM {memory_used:.1f}/{memory_total:.1f} GB"
            f" | CPU {cpu_rss_gb:.1f} GB"
            f" | {steps_per_sec:.1f} it/s]"
        )

    def log_data(
        self, writer: Writer, it: int, loader: torch.utils.data.DataLoader, split: str
    ) -> str:

        tot_n_samples = self.current_it

        name = f"{split}_data"

        if hasattr(loader.dataset, "stats"):
            data_metrics, data_figures = loader.dataset.stats()
            tools.write_dict_summaries(writer, name, data_metrics, tot_n_samples)
            tools.write_image_summaries(writer, name, data_figures, tot_n_samples)

        writer.add_scalar(f"{name}/num_batches", len(loader), tot_n_samples)
        writer.add_scalar(
            f"{name}/batch_size", loader.batch_size * self.num_gpus, tot_n_samples
        )
        writer.add_scalar(
            f"{name}/num_samples",
            len(loader) * loader.batch_size * self.num_gpus,
            tot_n_samples,
        )

    def log_data_stats(
        self, writer: Writer, it: int, scene_num_samples: dict[str, int], split: str
    ) -> str:
        tot_n_samples = self.current_it

        if len(scene_num_samples) == 0:
            return

        writer.add_scalar(
            f"{split}_data/num_scenes", len(scene_num_samples), tot_n_samples
        )
        writer.add_scalar(
            f"{split}_data/avg_frames_per_scene",
            np.mean(list(scene_num_samples.values())),
            tot_n_samples,
        )
        writer.add_scalar(
            f"{split}_data/min_frames_per_scene",
            np.min(list(scene_num_samples.values())),
            tot_n_samples,
        )

    # ------------------------------------------------------------------------
    # Step functions (train, eval, visualize, ...)
    # ------------------------------------------------------------------------

    def train_step(
        self, data: Batch, do_update: bool = True, log_grad_norm: bool = False
    ) -> tuple[Predictions, LossMetrics]:
        device_t = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(
            device_type=device_t,
            enabled=self.use_mp,
            dtype=self.dtype,
        ):
            data = misc.batch_to_device(data, self.device, non_blocking=True)
            self.step_timer.measure("to_device")
            pred = self.model(data)
            self.step_timer.measure("forward")
            losses, metrics = self.model.loss_metrics(pred, data)
            if self.conf.batch_mask_key is not None:
                exclude = self.conf.batch_mask_exclude
                mask, (losses, metrics) = apply_batch_mask(
                    pred, data, (losses, metrics), self.conf.batch_mask_key, exclude
                )
                metrics["batch_mask"] = mask.float().detach()
            if self.conf.get("compose_loss", None) is not None:
                losses["total"] = compose_loss(
                    {**metrics, **losses}, self.conf.compose_loss
                )
            loss = torch.mean(losses["total"])
            loss = loss / self.conf.gradient_accumulation_steps
            loss_metrics = {
                **metrics,
                **{"loss/" + k: v for k, v in losses.items()},
            }
            for k, v in loss_metrics.items():
                val = v.detach()
                if self.distributed:
                    torch.distributed.all_reduce(val.contiguous())
                    val = val / self.num_gpus
                loss_metrics[k] = val
            self.step_timer.measure("loss_fn")

            if torch.isnan(loss).any():
                if not self.conf.get("allow_nan", True):
                    raise RuntimeError("NaN detected in training.")
                logger.warning("Detected NAN, skipping iteration..")
                if self.distributed:
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.zero_grad()
                    del pred, data, loss, losses
                    return None, None

        do_backward = loss.requires_grad
        if self.distributed:
            do_backward = torch.tensor(do_backward).float().to(self.device)
            torch.distributed.all_reduce(
                do_backward, torch.distributed.ReduceOp.PRODUCT
            )
            do_backward = do_backward > 0
        if do_backward:
            self.scaler.scale(loss).backward()
            self.step_timer.measure("backward")

            if self.conf.detect_anomaly:
                # Check for params without any gradient which causes
                # problems in distributed training with checkpointing
                detected_anomaly = False
                for name, param in self.model.named_parameters():
                    if param.grad is None and param.requires_grad:
                        logger.warning(f"param {name} has no gradient.")
                        detected_anomaly = True
                if detected_anomaly:
                    raise RuntimeError("Detected anomaly in training.")
            if do_update:
                self.scaler.unscale_(self.optimizer)
                step_taken = True
                if self.conf.get("clip_grad", None):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.conf.clip_grad,
                        error_if_nonfinite=False,
                    )
                    if not torch.isfinite(grad_norm):
                        logger.warning("NaN detected in gradients. Skipping iteration.")
                        step_taken = False
                        self.scaler.update()
                    else:
                        if log_grad_norm:
                            loss_metrics["l2/grad_norm"] = grad_norm.reshape(1)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    if log_grad_norm:
                        loss_metrics["l2/grad_norm"] = torch.Tensor(
                            [misc.grad_norm(self.model.parameters())]
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                if step_taken:
                    self.learning_rate_step()
                self.optimizer.zero_grad()
            self.step_timer.measure("step")
        else:
            self.warn("Skip iteration due to detach.")
        return pred, loss_metrics

    def eval_step(self, data: Batch) -> tuple[Predictions, LossMetrics]:
        raise NotImplementedError()

    def visualize(self, data: Batch):
        raise NotImplementedError()

    # ------------------------------------------------------------------------
    # Main loops (train_epoch, eval_loop, test_loop)
    # ------------------------------------------------------------------------

    def train_epoch(
        self,
        output_dir: Path,
        dataloader: torch.utils.data.DataLoader,
        writer: Writer,
        max_iters: int | None = None,
    ):
        if self.rank == 0:
            self.log_data(writer, 0, dataloader, "training")
        do_profile = self.conf.profile
        if self.conf.profile_every_epoch is not None:
            do_profile = do_profile and (
                (self.epoch % self.conf.profile_every_epoch) == 0
            )
        else:
            do_profile = do_profile and (self.epoch == 0)
        profiler = self.construct_profiler(output_dir) if do_profile else None
        train_loss_metrics = collections.defaultdict(tools.AverageMetric)
        pr_metrics = collections.defaultdict(tools.PRMetric)
        train_iter = iter(dataloader)
        self.step_timer.hard_reset()
        self.optimizer.zero_grad()
        scene_num_samples = collections.defaultdict(int)
        for it in range(len(dataloader)):
            if max_iters is not None and it >= max_iters:
                logger.info(
                    f"Reached max iters {max_iters}, stopping epoch {self.epoch}."
                )
                break
            data = next(train_iter)
            if "scene" in data:
                for scene_id in data["scene"]:
                    scene_num_samples[scene_id] += 1
            if (
                self.rank == 0
                and it == 0
                and self.epoch == 0
                and self.conf.get("print_batch", True)
            ):
                # Log a single batch of data
                misc.print_summary(data)
            self.step_timer.measure("data")
            self.tot_n_samples += dataloader.batch_size * self.num_gpus
            self.tot_it += 1

            self.model.train()

            # Perform gradient accumulation
            do_update = ((it + 1) % self.conf.gradient_accumulation_steps) == 0

            do_log = it % self.conf.log_every_iter == 0
            pred, loss_metrics = self.train_step(
                data, do_update=do_update, log_grad_norm=do_log
            )
            if pred is None:
                continue  # skip iteration due to NaN
            if self.rank == 0:
                batch_mask = loss_metrics.get("batch_mask", None)
                exclude = (*self.conf.batch_mask_exclude, "batch_mask")
                for k, val in loss_metrics.items():
                    m = None
                    if (
                        batch_mask is not None
                        and not any(p in k for p in exclude)
                        and val.shape[0] == batch_mask.shape[0]
                    ):
                        m = batch_mask
                    train_loss_metrics[k].update(val, mask=m)

                for k, labels_preds in self.model.pr_metrics(pred, data).items():
                    pr_metrics[k].update(*labels_preds)

            # Run profiler (stack trace, ...)
            if profiler is not None:
                profiler.step()

            # Record memory usage
            if self.conf.record_memory:
                self.record_memory(output_dir, it)

            # Log training metrics (loss, ...) and hardware usage
            if do_log and self.rank == 0:
                time_and_mem_str = self.log_time_and_memory(
                    writer, it, dataloader.batch_size
                )
                self.log_train(
                    writer,
                    it,
                    {**train_loss_metrics, **pr_metrics},
                    extra_str=time_and_mem_str,
                )

                self.log_data_stats(writer, it, scene_num_samples, split="training")

                train_loss_metrics.clear()  # Reset training loss aggregators
                pr_metrics.clear()  # Reset PR metrics

            # Make plots of training steps
            should_plot = it == 0 or (
                self.conf.plot_every_iter is not None
                and it % self.conf.plot_every_iter == 0
            )
            if should_plot and self.rank == 0:
                with torch.no_grad():
                    figures = eval_model(self.model).visualize(pred, data)
                tools.write_image_summaries(
                    writer, "training", figures, self.current_it
                )

            # Log gradients
            if self.conf.log_grad_every_iter is not None:
                raise NotImplementedError()

            del pred, data, loss_metrics
            if it == 0:
                torch.cuda.empty_cache()  # should be cleared at the first iter
            self.step_timer.reset()
        self.optimizer.zero_grad()

        del train_loss_metrics, train_iter
        if self.distributed:
            dist.barrier()

    def eval_loop(
        self,
        output_dir: Path,
        loader: torch.utils.data.DataLoader,
        max_iters: int | None = None,
    ):
        """Run evaluation loop."""
        self.model.eval()
        with torch.no_grad():
            with tools.fork_rng(seed=self.conf.seed):
                results, pr_metrics, figures = run_evaluation(
                    self.model,
                    loader,
                    self.device,
                    self.conf,
                    self.rank,
                    pbar=(self.rank == 0),
                    max_iters=max_iters,
                    compose_loss_str=self.conf.get("compose_loss", None),
                )
        return results, pr_metrics, figures

    @torch.compiler.set_stance("force_eager")
    def test_loop(
        self,
        output_dir: Path,
        benchmark_name: str,
        benchmark_conf: str,
        writer: SummaryWriter | None = None,
    ):
        """Interface for test loop."""
        logger.info(f"Running eval on {benchmark_name}")
        if not self.conf.benchmark_checkpoint:
            model = self.sequential_model()  # no DDP
        self.info("Configuration: \n%s", OmegaConf.to_yaml(benchmark_conf))
        with torch.no_grad():
            eval_dir = output_dir / f"test_{self.epoch}" / benchmark_name
            with tools.fork_rng(seed=self.conf.seed):
                summaries, figures, _ = eval.run_benchmark(
                    benchmark_name,
                    benchmark_conf,
                    eval_dir,
                    model=(None if self.conf.benchmark_checkpoint else model.eval()),
                )
            # Create symlink to eval_dir at head
            symlink_dir = output_dir / benchmark_name
            try:
                symlink_dir.unlink(missing_ok=True)
                symlink_dir.symlink_to(eval_dir)
            except OSError as e:
                logger.warning(
                    f"Could not create symlink {symlink_dir} -> {eval_dir}: {e}"
                )
            # Remove previous test dirs for this benchmark, keep only latest
            for old_dir in output_dir.glob(f"test_*/{benchmark_name}"):
                if old_dir != eval_dir:
                    shutil.rmtree(old_dir, ignore_errors=True)
                    # Remove parent test_N dir if now empty
                    parent = old_dir.parent
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
        str_summaries = [
            f"{k} {v:.3E}" for k, v in summaries.items() if isinstance(v, float)
        ]
        logger.info(f'[{benchmark_name}] {{{", ".join(str_summaries)}}}')
        if writer is not None:
            step = self.current_it
            tools.write_dict_summaries(
                writer, f"test_{benchmark_name}", summaries, step
            )
            tools.write_image_summaries(writer, f"test_{benchmark_name}", figures, step)
        return summaries, figures

    def run_all_benchmarks(
        self, output_dir: Path, writer: Writer = None, force: bool = False
    ):
        epoch = 0 if force else self.epoch
        for bench_name, (bench_conf, every_epoch) in self.benchmarks.items():
            if epoch % every_epoch == 0 and self.rank == 0:
                # TODO: Make benchmarks distributed!
                self.test_loop(output_dir, bench_name, bench_conf, writer)
            if self.distributed:
                dist.barrier()

    def run_eval(
        self,
        output_dir: Path,
        dataset: datasets.BaseDataset,
        writer: Writer = None,
        max_iters: int | None = None,
    ) -> tuple[Any, ...]:
        eval_loader = dataset.get_data_loader(
            self.conf.eval_split,
            overfit=self.conf.overfit,
            distributed=self.distributed,
            pinned=True,
        )
        if self.rank == 0 and writer is not None:
            self.log_data(writer, 0, eval_loader, "eval")
        self.info(f"Evaluation loader has {len(eval_loader)} batches")
        eval_results = self.eval_loop(output_dir, eval_loader, max_iters=max_iters)
        if self.rank == 0 and writer is not None:
            self.log_eval(writer, 0, eval_results)
        return eval_results

    # ------------------------------------------------------------------------
    # Run full training on dataset (train multiple epochs + validation + test)
    # ------------------------------------------------------------------------

    def train_loop(
        self,
        output_dir: Path,
        dataset: datasets.BaseDataset,
        writer: Writer = None,
    ):
        """The main function."""
        # Initialize writer
        full_conf = OmegaConf.create(
            {"data": dataset.conf, "model": self.model_conf, "train": self.conf}
        )
        if writer is None:
            writer = self.get_writer(output_dir, full_conf)
        if self.conf.eval_init:
            self.run_eval(output_dir, dataset, writer, max_iters=self.conf.eval_iters)
            self.save_checkpoint(output_dir, full_conf)
            self.run_all_benchmarks(output_dir, writer, force=True)

        # Start Loop
        while self.epoch < self.conf.epochs:
            self.info(f"Starting epoch {self.epoch}")

            # Re-seed epoch
            tools.set_seed(self.conf.seed + self.epoch)
            self.info("Setting up data loader")

            self._apply_finetune_scales(dataset)

            # Create data loader
            self.info("Creating train data loader (sampling)...")
            train_loader = dataset.get_data_loader(
                self.conf.train_split,
                distributed=self.distributed,
                epoch=self.epoch,
                pinned=True,
                overfit=self.conf.overfit,
            )
            self.info("Train data loader ready.")
            self.epoch_tracker.set_epoch_length(len(train_loader))
            self.info(f"Training loader has {len(train_loader)} batches")

            self.info("Start training")
            self.train_epoch(
                output_dir, train_loader, writer, max_iters=self.conf.train_iters
            )
            del train_loader  # shutdown multiprocessing pool
            gc.collect()  # ensure workers are dead before val spawns

            self.epoch += 1
            # Checkpointing
            self.save_checkpoint(output_dir, full_conf)
            # Validation
            if self.conf.eval_every_epoch:
                if (
                    self.epoch % self.conf.eval_every_epoch == 0
                    or self.epoch == self.conf.epochs  # Run eval in last epoch
                ):
                    self.run_eval(
                        output_dir, dataset, writer, max_iters=self.conf.eval_iters
                    )

            # Run test loops
            self.run_all_benchmarks(output_dir, writer)

        # Final evals
        self.run_eval(output_dir, dataset, writer, max_iters=self.conf.eval_iters)
        self.run_all_benchmarks(output_dir, writer, force=True)

        if writer is not None:
            writer.close()


def scale_by_device_count(
    data_conf: DictConfig, num_gpus: int, batch_size_per_gpu: bool | None = None
) -> DictConfig:
    """Scale data conf by device count (Maybe)."""
    batch_size_per_gpu = (
        batch_size_per_gpu
        if batch_size_per_gpu is not None
        else data_conf.get("batch_size_per_gpu", False)
    )
    # adjust batch size and num of workers since these are per GPU
    if "batch_size" in data_conf and not batch_size_per_gpu:
        data_conf.batch_size = int(data_conf.batch_size / num_gpus)

    ref_vram = data_conf.get("ref_vram", None)
    if ref_vram is not None and torch.cuda.is_available():
        per_device_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_scale = per_device_vram / ref_vram
        for key in ["batch_size"] + [
            f"{s}_batch_size" for s in ["train", "val", "test"]
        ]:
            if key in data_conf:
                scaled = int(max(1, round(data_conf[key] * vram_scale)))
                logger.info(
                    "VRAM scaling (%s): %d → %d (%.1f GB / %.1f GB ref)",
                    key,
                    data_conf[key],
                    scaled,
                    per_device_vram,
                    ref_vram,
                )
                data_conf[key] = scaled

    logger.info(
        "Batch size: global=%d, per-device=%d",
        data_conf.batch_size * num_gpus,
        data_conf.batch_size,
    )
    for split in ["train", "val", "test"]:
        split_batch_size_key = f"{split}_batch_size"
        if split_batch_size_key in data_conf and not batch_size_per_gpu:
            data_conf[split_batch_size_key] = int(
                data_conf[split_batch_size_key] / num_gpus
            )
    return data_conf


def init_trainer(
    output_dir: Path,
    conf: DictConfig,
    device: torch.device,
    dummy_batch_fn: Callable[[], Batch] | None = None,
) -> Trainer:
    if conf.train.get("reload_model"):
        assert conf.train.load_experiment is not None
        pretrain_dir = settings.TRAINING_PATH / conf.train.load_experiment
        logger.info(f"Finetuning: Loading model config from {pretrain_dir}.")
        pretrain_conf = OmegaConf.load(pretrain_dir / "config.yaml")
        conf.model = OmegaConf.merge(pretrain_conf.model, conf.model)
        OmegaConf.save(conf, str(output_dir / "config.yaml"))
    model = models.get_model(conf.model.name)(conf.model).to(device)
    if conf.get("lazy_init", True):
        logger.info("Running dummy forward pass to initialize lazy modules.")
        assert (
            dummy_batch_fn is not None
        ), "dummy_batch_fn must be provided for lazy_init"
        logger.info("Building dummy dataset and sampling...")
        dummy_batch = dummy_batch_fn()
        logger.info("Moving dummy batch to device...")
        dummy_batch = misc.batch_to_device(dummy_batch, device, non_blocking=False)
        logger.info("Running model forward on dummy batch...")
        with torch.no_grad():
            model(dummy_batch)
        del dummy_batch
        logger.info("Dummy forward pass completed.")
    trainer = Trainer.init(conf.train, model, device=device)

    # Register benchmarks (e.g. MegaDepth1500)
    num_samples = conf.get("eval", {}).pop("num_samples", None)
    for bench in conf.train.get("run_benchmarks", ()):
        bench_name, every_epoch = (bench, None) if isinstance(bench, str) else bench
        eval.get_benchmark(bench_name)  # Check if benchmark exists
        bench_conf = (
            {}
            if conf.get("benchmarks") is None
            else conf.benchmarks.get(bench_name, {})
        )
        bench_conf = OmegaConf.merge(
            {
                "eval": conf.get("eval", {}),
                "num_samples": num_samples,
                "checkpoint": str(output_dir.relative_to(settings.TRAINING_PATH)),
            },
            OmegaConf.create(bench_conf),
        )
        trainer.register_benchmark(bench_name, bench_conf, every_epoch=every_epoch)
    # Maybe load experiment
    trainer.maybe_load_checkpoint()
    return trainer


def launch_training(output_dir: Path, conf: DictConfig, device: torch.device):
    tools.set_seed(conf.train.seed)
    dataset = datasets.get_dataset(conf.data.name)(
        scale_by_device_count(conf.data, conf.train.num_devices or 1)
    )
    trainer = init_trainer(
        output_dir, conf, device, dummy_batch_fn=dataset.get_dummy_batch
    )
    # Run actual training loop
    trainer.train_loop(output_dir, dataset)
