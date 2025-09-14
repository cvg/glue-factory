"""
A generic, flexible trainer.

Author: Philipp Lindenberger
"""

import collections
import signal
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gluefactory import datasets, models
from gluefactory.models import BaseModel
from gluefactory.utils import experiments, misc, tools

from . import eval, logger, settings

Args: TypeAlias = DictConfig
Batch: TypeAlias = Any
Predictions: TypeAlias = Any
LossMetrics: TypeAlias = dict[str, torch.Tensor]
Writer: TypeAlias = SummaryWriter | None


def compose_loss(loss_dict: LossMetrics, compose_str: str) -> torch.Tensor:
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
            raise KeyError(f"Key {key} not found in loss dict.")
        loss = loss + weight * loss_dict[key]
    return loss


@torch.no_grad()
def run_evaluation(
    model: BaseModel | torch.nn.parallel.DistributedDataParallel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    conf: DictConfig,
    rank: int = 0,
    pbar: bool = True,
    max_iters: int | None = None,
) -> tuple[Any, ...]:
    model.eval()
    model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )  # Get the original model
    results = {}
    pr_metrics = collections.defaultdict(tools.PRMetric)
    figures = []
    plot_ids = np.random.choice(
        len(loader), min(len(loader), conf.num_eval_plots), replace=False
    )
    max_iters = max_iters or len(loader)
    for i, data in enumerate(
        tqdm(loader, desc="Evaluation", ascii=True, disable=not pbar)
    ):
        if i >= max_iters:
            break
        data = misc.batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses, metrics = model.loss(pred, data)
            if i in plot_ids:
                figures.append(model.visualize(pred, data))
            # add PR curves
            for k, labels_preds in model.pr_metrics(pred, data).items():
                pr_metrics[k].update(*labels_preds)
        del pred, data
        numbers = {**metrics, **{"loss/" + k: v for k, v in losses.items()}}
        for k, v in numbers.items():
            if k not in results:
                results[k] = tools.AverageMetric()
                if k in conf.median_metrics:
                    results[k + "_median"] = tools.MedianMetric()
                if k in conf.recall_metrics.keys():
                    q = conf.recall_metrics[k]
                    results[k + f"_recall{int(q)}"] = tools.RecallMetric(q)
            results[k].update(v)
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
        },
        "lr_scaling": [(100, ["dampingnet.const"])],
        "eval_every_iter": 1000,  # interval for evaluation on the validation set
        "eval_every_epoch": None,  # interval for evaluation on the validation set
        "benchmark_every_epoch": 1,  # interval for evaluation on the test benchmarks
        "save_every_iter": 5000,  # interval for saving the current checkpoint
        "log_every_iter": 200,  # interval for logging the loss to the console
        "log_grad_every_iter": None,  # interval for logging gradient hists
        "test_every_epoch": 1,  # interval for evaluation on the test benchmarks
        "keep_last_checkpoints": 3,  # keep only the last X checkpoints
        "load_experiment": None,  # initialize the model from a previous experiment
        "median_metrics": [],  # add the median of some metrics
        "recall_metrics": {},  # add the recall of some metrics
        "best_key": "loss/total",  # key to use to select the best checkpoint
        "dataset_callback_fn": None,  # data func called at the start of each epoch
        "dataset_callback_on_val": False,  # call data func on val data?
        "clip_grad": None,
        "pr_curves": {},  # add pr curves, set labels/predictions/mask keys
        "num_eval_plots": 4,  # Number of plots to show during evaluation (0=skip)
        "plot_every_iter": None,  # plot figures every X iterations
        "submodules": [],
        "mixed_precision": None,
        "num_devices": 0,  # 0 means sequential.
        "compile": None,  # Compilation mode for the model. [None, default, ...]
        "profile": None,  # Profile the training with PyTorch profiler (# prof steps)
        "record_memory": None,  # Record memory usage during training (# record steps)
        "log_it": False,  # Log tensorboard on iteration (default is num_samples)
        "detect_anomaly": False,  # Enable anomaly detection
        "gradient_accumulation_steps": 1,  # Accumulate gradients over N steps
        "ddp_find_unused_parameters": False,  # DDP find_unused_parameters
        "run_benchmarks": (),
    }

    def __init__(
        self,
        conf: DictConfig,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        device: torch.device | str | None = None,
    ):
        # Initialize conf, model, optimizer and LR
        self.default_conf = OmegaConf.create(self.default_conf)
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

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
        self.all_params = self.model.parameters()
        self.model_conf = self.model.conf

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
        lr_scheduler = tools.get_lr_scheduler(
            optimizer=optimizer, conf=conf.lr_schedule
        )
        return cls(
            conf=conf,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
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
        strict: bool = False,
        load_state: bool = False,
        load_modelconfig: bool = False,
    ):
        # TODO: Fix bug when loading distributed cp from single gpu
        self.model.load_state_dict(checkpoint["model"], strict=strict)
        if load_modelconfig:
            self.conf.model = OmegaConf.merge(
                OmegaConf.create(checkpoint["conf"]).model, self.conf.model
            )
        if load_state:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "lr_scheduler" in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"]

    def maybe_load_checkpoint(self):
        if self.conf.load_experiment:
            init_cp = experiments.get_last_checkpoint(self.conf.load_experiment)
            self.info("Loading checkpoint %s", str(init_cp))
            init_cp = torch.load(
                str(init_cp), map_location="cpu", weights_only=not settings.ALLOW_PICKLE
            )
            self.load_checkpoint(
                init_cp,
                load_state=self.conf.get("load_state", False),
                load_modelconfig=self.conf.get("load_modelconfig", False),
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
            return experiments.save_experiment(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                conf,
                results,
                iter_i=iter_i,
                epoch=self.epoch,
                output_dir=output_dir,
                **kwargs,
            )

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
        self.lr_scheduler.step()
        if verbose:
            self.info(
                f'lr changed from {old_lr} to {self.optimizer.param_groups[0]["lr"]}'
            )

    def setup_sigint_handler(self):
        def sigint_handler(signal, frame):
            logger.info("Caught keyboard interrupt signal, will terminate")
            if self.stop:
                raise KeyboardInterrupt
            self.stop = True

        self.stop = False
        signal.signal(signal.SIGINT, sigint_handler)

    def setup_torch(self):
        torch.backends.cudnn.benchmark = True
        if self.conf.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        # TODO

    def prepare_model(self):
        if self.conf.compile:
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
                wait=5, warmup=1, active=True, repeat=1, skip_first=10
            ),
            on_trace_ready=experiments.tensorboard_trace_handler(
                str(output_dir), use_gzip=not store_raw_trace
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
        lr_params = tools.pack_lr_parameters(params, conf.lr, conf.lr_scaling)
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
        }[mixed_precision]

        return use_mp

    def get_writer(self, output_dir: Path):
        if self.rank == 0:
            writer = SummaryWriter(log_dir=str(output_dir))
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

    def log_train(self, writer: Writer, it: int, train_loss_metrics: LossMetrics):
        tot_n_samples = self.current_it
        all_params = self.all_params
        writer.add_scalar("l2/param_norm", misc.param_norm(all_params), tot_n_samples)
        writer.add_scalar("l2/grad_norm", misc.grad_norm(all_params), tot_n_samples)
        loss_metrics = {k: v.compute() for k, v in train_loss_metrics.items()}
        str_loss_metrics = [f"{k} {v:.3E}" for k, v in loss_metrics.items()]
        # Write training losses
        logger.info(
            "[E {} | it {}] loss {{{}}}".format(
                self.epoch, it, ", ".join(str_loss_metrics)
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
        batch_size: int,
    ):
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

            writer.add_figure(
                "step/sections",
                self.step_timer.plot(),
                tot_n_samples,
                close=True,
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

        self.info(
            f"[Used {memory_used:.1f}/{memory_total:.1f} GB | {steps_per_sec:.1f} it/s]"
        )

    # ------------------------------------------------------------------------
    # Step functions (train, eval, visualize, ...)
    # ------------------------------------------------------------------------

    def train_step(
        self, data: Batch, do_update: bool = True
    ) -> tuple[Predictions, LossMetrics]:
        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=self.use_mp,
            dtype=self.dtype,
        ):
            data = misc.batch_to_device(data, self.device, non_blocking=True)
            self.step_timer.measure("to_device")
            pred = self.model(data)
            self.step_timer.measure("forward")
            losses, metrics = self.model.loss(pred, data)
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
                    torch.distributed.all_reduce(val)
                    val = val / self.num_gpus
                loss_metrics[k] = val
            self.step_timer.measure("loss_fn")

            if torch.isnan(loss).any():
                logger.warning("Detected NAN, skipping iteration..")
                del pred, data, loss, losses
                return

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
                    if self.conf.get("clip_grad", None):
                        self.scaler.unscale_(self.optimizer)
                        try:
                            torch.nn.utils.clip_grad_norm_(
                                self.all_params,
                                max_norm=self.conf.clip_grad,
                                error_if_nonfinite=True,
                            )
                            self.scaler.step(self.optimizer)
                        except RuntimeError:
                            logger.warning(
                                "NaN detected in gradients. Skipping iteration."
                            )
                        self.scaler.update()
                    else:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    self.optimizer.zero_grad()
                self.step_timer.measure("step")
                if not self.conf.lr_schedule.on_epoch:
                    self.learning_rate_step()
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
        if self.distributed:
            dataloader.sampler.set_epoch(self.epoch)
        do_profile = self.conf.profile and self.epoch == 0
        profiler = self.construct_profiler(output_dir) if do_profile else None
        train_loss_metrics = collections.defaultdict(tools.AverageMetric)
        train_iter = iter(dataloader)
        self.step_timer.hard_reset()
        self.optimizer.zero_grad()
        for it in range(len(dataloader)):
            if max_iters is not None and it >= max_iters:
                logger.info(
                    f"Reached max iters {max_iters}, stopping epoch {self.epoch}."
                )
                break
            data = next(train_iter)
            self.step_timer.measure("data")
            self.tot_n_samples += dataloader.batch_size * self.num_gpus
            self.tot_it += 1

            self.model.train()

            # Perform gradient accumulation
            do_update = ((it + 1) % self.conf.gradient_accumulation_steps) == 0
            pred, loss_metrics = self.train_step(data, do_update=do_update)
            for k, val in loss_metrics.items():
                train_loss_metrics[k].update(val)

            # Run profiler (stack trace, ...)
            if profiler is not None:
                profiler.step()

            # Record memory usage
            if self.conf.record_memory:
                self.record_memory(output_dir, it)

            # Log training metrics (loss, ...) and hardware usage
            if (it % self.conf.log_every_iter == 0) and self.rank == 0:
                self.log_train(writer, it, train_loss_metrics)
                train_loss_metrics.clear()  # Reset training loss aggregators
                self.log_time_and_memory(writer, dataloader.batch_size)

            # Make plots of training steps
            if self.conf.plot_every_iter is not None:
                if it % self.conf.plot_every_iter == 0 and self.rank == 0:
                    figures = self.model.visualize(pred, data)
                    tools.write_image_summaries(
                        writer, "training", figures, self.current_it
                    )

            # Log gradients
            if self.conf.log_grad_every_iter is not None:
                raise NotImplementedError()

            del pred, data, loss_metrics
            torch.cuda.empty_cache()  # should be cleared at the first iter
            self.step_timer.reset()
        self.optimizer.zero_grad()
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
                )
        return results, pr_metrics, figures

    def test_loop(
        self,
        output_dir: Path,
        benchmark_name: str,
        benchmark_conf: str,
        writer: SummaryWriter | None = None,
    ):
        """Interface for test loop."""
        logger.info(f"Running eval on {benchmark_name}")
        model = self.sequential_model()  # no DDP
        self.info("Configuration: \n%s", OmegaConf.to_yaml(benchmark_conf))
        with torch.no_grad():
            eval_dir = output_dir / f"test_{self.epoch}" / benchmark_name
            with tools.fork_rng(seed=self.conf.seed):
                summaries, figures, _ = eval.run_benchmark(
                    benchmark_name,
                    benchmark_conf,
                    eval_dir,
                    model.eval(),
                )
            # Create symlink to eval_dir at head
            symlink_dir = output_dir / benchmark_name
            symlink_dir.unlink(missing_ok=True)
            symlink_dir.symlink_to(eval_dir)
        # TODO: Cleanup? Maybe not so necessary
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

    # ------------------------------------------------------------------------
    # Run full training on dataset (train multiple epochs + validation + test)
    # ------------------------------------------------------------------------

    def train_loop(
        self,
        output_dir: Path,
        dataset: datasets.BaseDataset,
    ):
        """The main function."""
        # Initialize writer
        writer = self.get_writer(output_dir)

        full_conf = OmegaConf.create(
            {"data": dataset.conf, "model": self.model_conf, "train": self.conf}
        )

        for bench_name, (bench_conf, _) in self.benchmarks.items():
            if self.rank == 0 and self.conf.get("eval_init", False):
                # TODO: Make benchmarks distributed!
                self.test_loop(output_dir, bench_name, bench_conf, writer)
            if self.distributed:
                dist.barrier()

        # Start Loop
        while self.epoch < self.conf.epochs:
            self.info(f"Starting epoch {self.epoch}")

            # Re-seed epoch
            tools.set_seed(self.conf.seed + self.epoch)
            self.info("Setting up data loader")

            if self.conf.lr_schedule.on_epoch and self.epoch > 0:
                self.learning_rate_step(verbose=True)

            # Create data loader
            train_loader = dataset.get_data_loader(
                "train", distributed=self.distributed, epoch=self.epoch
            )
            self.info(f"Training loader has {len(train_loader)} batches")

            self.info("Start training")
            self.train_epoch(output_dir, train_loader, writer)
            del train_loader  # shutdown multiprocessing pool

            self.epoch += 1
            # Checkpointing
            self.save_checkpoint(output_dir, full_conf)
            # Validation
            if self.conf.eval_every_epoch:
                if self.epoch % self.conf.eval_every_epoch == 0:
                    val_loader = dataset.get_data_loader("val")
                    self.info(f"Validation loader has {len(val_loader)} batches")
                    eval_results = self.eval_loop(output_dir, val_loader)
                    self.log_eval(writer, 0, eval_results)

            # Run test loops
            for bench_name, (bench_conf, every_epoch) in self.benchmarks.items():
                if self.epoch % every_epoch == 0 and self.rank == 0:
                    # TODO: Make benchmarks distributed!
                    self.test_loop(output_dir, bench_name, bench_conf, writer)
                if self.distributed:
                    dist.barrier()


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

    logger.info(
        "Batch size: global=%d, per-device=%d",
        data_conf.batch_size * num_gpus,
        data_conf.batch_size,
    )
    if "train_batch_size" in data_conf and not batch_size_per_gpu:
        data_conf.train_batch_size = int(data_conf.train_batch_size / num_gpus)
    if "num_workers" in data_conf:
        # We always scale the workers
        data_conf.num_workers = int((data_conf.num_workers + num_gpus - 1) / num_gpus)
    return data_conf


def launch_training(output_dir: Path, conf: DictConfig, device: torch.device):
    tools.set_seed(conf.train.seed)
    dataset = datasets.get_dataset(conf.data.name)(
        scale_by_device_count(conf.data, conf.train.num_devices or 1)
    )
    model = models.get_model(conf.model.name)(conf.model).to(device)
    if conf.get("lazy_init", True):
        logger.info("Running dummy forward pass to initialize lazy modules.")
        dummy_batch = dataset.get_dummy_batch()
        dummy_batch = misc.batch_to_device(dummy_batch, device, non_blocking=False)
        with torch.no_grad():
            model(dummy_batch)
        del dummy_batch
        logger.info("Dummy forward pass completed.")
    trainer = Trainer.init(conf.train, model, device=device)
    # Register benchmarks (e.g. MegaDepth1500)
    for bench in conf.train.get("run_benchmarks", ()):
        bench_name, every_epoch = (bench, None) if isinstance(bench, str) else bench
        eval.get_benchmark(bench_name)  # Check if benchmark exists
        bench_conf = (
            {}
            if conf.get("benchmarks") is None
            else conf.benchmarks.get(bench_name, {})
        )
        trainer.register_benchmark(bench_name, bench_conf, every_epoch=every_epoch)
    # Maybe load experiment
    trainer.maybe_load_checkpoint()

    # Run actual training loop
    trainer.train_loop(output_dir, dataset)
