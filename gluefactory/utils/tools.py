"""
Various handy Python and PyTorch utils.

Author: Paul-Edouard Sarlin (skydes)
"""

import collections
import functools
import logging
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AverageMetric:
    def __init__(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, tensor):
        assert tensor.dim() == 1
        self._sum += torch.nansum(tensor)
        self._num_examples += len(tensor)

    def compute(self):
        if self._num_examples == 0:
            return np.nan
        else:
            return self._sum.item() / self._num_examples


# same as AverageMetric, but tracks all elements
class FAverageMetric:
    def __init__(self):
        self._sum = 0
        self._num_examples = 0
        self._elements = []

    def update(self, tensor):
        self._elements += tensor.cpu().numpy().tolist()
        assert tensor.dim() == 1
        tensor = tensor[~torch.isnan(tensor)]
        self._sum += tensor.sum().item()
        self._num_examples += len(tensor)

    def compute(self):
        if self._num_examples == 0:
            return np.nan
        else:
            return self._sum / self._num_examples


class MedianMetric:
    def __init__(self):
        self._elements = []

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return np.nanmedian(self._elements)


class PRMetric:
    def __init__(self):
        self.labels = []
        self.predictions = []

    @torch.no_grad()
    def update(self, labels, predictions, mask=None):
        assert labels.shape == predictions.shape
        self.labels += (
            (labels[mask] if mask is not None else labels).cpu().numpy().tolist()
        )
        self.predictions += (
            (predictions[mask] if mask is not None else predictions)
            .cpu()
            .numpy()
            .tolist()
        )

    @torch.no_grad()
    def compute(self):
        return np.array(self.labels), np.array(self.predictions)

    def reset(self):
        self.labels = []
        self.predictions = []


class QuantileMetric:
    def __init__(self, q=0.05):
        self._elements = []
        self.q = q

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return np.nanquantile(self._elements, self.q)


class RecallMetric:
    def __init__(self, ths, elements=[]):
        self._elements = elements
        self.ths = ths

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if isinstance(self.ths, Iterable):
            return [self.compute_(th) for th in self.ths]
        else:
            return self.compute_(self.ths[0])

    def compute_(self, th):
        if len(self._elements) == 0:
            return np.nan
        else:
            s = (np.array(self._elements) < th).sum()
            return s / len(self._elements)


def cal_error_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.round((np.trapz(r, x=e) / t), 4))
    return aucs


class AUCMetric:
    def __init__(self, thresholds, elements=None):
        if elements is None:
            self._elements = []
        else:
            self._elements = elements
        self.thresholds = thresholds
        if not isinstance(thresholds, Sequence):
            self.thresholds = [thresholds]

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return cal_error_auc(self._elements, self.thresholds)


class Timer(object):
    """A simpler timer context object.
    Usage:
    ```
    > with Timer('mytimer'):
    >   # some computations
    [mytimer] Elapsed: X
    ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.duration = time.time() - self.tstart
        if self.name is not None:
            print("[%s] Elapsed: %s" % (self.name, self.duration))


class RunningStats:
    """
    A numerically stable running statistics tracker using Welford's algorithm.
    Avoids overflow and maintains precision for large datasets.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics to initial state."""
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of deviations from mean

    def update(self, val: float) -> None:
        # Welford's online algorithm
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.M2 += delta * delta2

    def compute(self) -> tuple[float, float]:
        """Compute the mean and standard deviation."""
        if self.count == 0:
            return 0.0, 0.0
        elif self.count == 1:
            return self.mean, 0.0
        else:
            variance = self.M2 / self.count  # Population variance
            std = np.sqrt(variance)
            return self.mean, std


class StepTimer:
    def __init__(self):
        self.stats = collections.defaultdict(RunningStats)
        self.start = None

    def reset(self):
        """Reset the timer for a specific name."""
        self.start = time.time()

    def hard_reset(self):
        self.stats = collections.defaultdict(RunningStats)
        self.start = time.time()

    def measure(self, name: str):
        """Measure the time taken for a specific operation."""
        elapsed = time.time() - self.start
        self.stats[name].update(elapsed)
        self.start = time.time()

    def compute(self) -> tuple[float, dict[str, float]]:
        """Compute the average time for each operation (in seconds)."""
        avg_step_times = {k: v.compute()[0] for k, v in self.stats.items()}
        total_time = sum(avg_step_times.values())
        return total_time, avg_step_times

    def plot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        total_time = 0
        for name, stats in self.stats.items():
            section_time, section_var = stats.compute()
            ax.bar(name, section_time * 1000, label=name, yerr=section_var * 1000)
            total_time += section_time
        ax.set_ylabel("Duration (ms)")
        ax.set_title(
            f"Step Time Composition "
            f"(total: {total_time:.2f}s = {1 / total_time:.2f} steps/s)"
        )
        return fig

    def num_steps(self):
        """Return the number of steps measured."""
        return list(self.stats.values())[0].count if self.stats else 0


def collect_device_stats() -> dict[str, float]:
    """Collect device usage statistics."""

    def _per_device_stats(device: torch.device | None = None) -> dict[str, float]:
        free, total = torch.cuda.mem_get_info(device)
        used = total - free
        bytes_stats = {
            # "z_allocated": torch.cuda.memory_allocated(device),
            # "z_reserved": torch.cuda.memory_reserved(device),
            "z_allocated_peak": torch.cuda.max_memory_allocated(device),
            "z_reserved_peak": torch.cuda.max_memory_reserved(device),
            "used": used,
            "total": total,
        }

        device_stats = {k: v / 10**9 for k, v in bytes_stats.items()}
        device_stats["utilization"] = device_stats["used"] / device_stats["total"]
        # Reset peak memory stats for next cycle
        torch.cuda.reset_peak_memory_stats(device)
        return device_stats

    num_devices = torch.cuda.device_count()
    all_devices = [torch.cuda.device(i) for i in range(num_devices)]
    all_device_stats = [_per_device_stats(d) for d in all_devices]
    all_device_stats = {
        k: [pds[k] for pds in all_device_stats] for k in all_device_stats[0]
    }
    device_stats = {k: np.mean(v).item() for k, v in all_device_stats.items()}
    for i in range(num_devices):
        device_stats[f"utilization_{i}"] = all_device_stats["utilization"][i]
    # Assumes all devices have the same memory stats.
    device_stats["global_total"] = sum(all_device_stats["total"])
    device_stats["global_used"] = sum(all_device_stats["used"])
    return device_stats


def get_class(mod_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
    the module named mod_name, child of base_path.
    """
    import inspect

    mod = __import__(mod_path, fromlist=[""])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]


def set_num_threads(nt):
    """Force numpy and other libraries to use a limited number of threads."""
    try:
        import mkl
    except ImportError:
        pass
    else:
        mkl.set_num_threads(nt)
    torch.set_num_threads(1)
    os.environ["IPC_ENABLE"] = "1"
    for o in [
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]:
        os.environ[o] = str(nt)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_random_state(with_cuda):
    pth_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    if torch.cuda.is_available() and with_cuda:
        cuda_state = torch.cuda.get_rng_state_all()
    else:
        cuda_state = None
    return pth_state, np_state, py_state, cuda_state


def set_random_state(state):
    pth_state, np_state, py_state, cuda_state = state
    torch.set_rng_state(pth_state)
    np.random.set_state(np_state)
    random.setstate(py_state)
    if (
        cuda_state is not None
        and torch.cuda.is_available()
        and len(cuda_state) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def fork_rng(seed=None, with_cuda=True):
    state = get_random_state(with_cuda)
    if seed is not None:
        set_seed(seed)
    try:
        yield
    finally:
        set_random_state(state)


def filter_parameters(params, regexp):
    """Filter trainable parameters based on regular expressions."""

    # Examples of regexp:
    #     '.*(weight|bias)$'
    #     'cnn\.(enc0|enc1).*bias'
    def filter_fn(x):
        n, p = x
        match = re.search(regexp, n)
        if not match:
            p.requires_grad = False
        return match

    params = list(filter(filter_fn, params))
    assert len(params) > 0, regexp
    logger.info("Selected parameters:\n" + "\n".join(n for n, p in params))
    return params


def get_lr_scheduler(optimizer, conf):
    """Get lr scheduler specified by conf.train.lr_schedule."""
    if conf.type not in ["factor", "exp", "cos", "cos_log", None]:
        if hasattr(conf.options, "schedulers"):
            # Add option to chain multiple schedulers together
            # This is useful for e.g. warmup, then cosine decay
            schedulers = []
            for scheduler_conf in conf.options.schedulers:
                scheduler = get_lr_scheduler(optimizer, scheduler_conf)
                schedulers.append(scheduler)

            options = {k: v for k, v in conf.options.items() if k != "schedulers"}
            return getattr(torch.optim.lr_scheduler, conf.type)(
                optimizer, schedulers, **options
            )

        return getattr(torch.optim.lr_scheduler, conf.type)(optimizer, **conf.options)

    elif conf.type.startswith("cos"):

        def log_decay(x: float, end_val: float) -> float:
            return 10 ** (np.log10(end_val) * (1 - x))

        def linear_decay(x: float, end_val: float) -> float:
            return x * (1 - end_val) + end_val

        def cosine_decay(scale_fn: Callable[[float, float], float], it: int) -> float:
            n_min = conf.min_factor
            tmax = conf.end - conf.start
            it = it - conf.start
            if it < 0:
                return 1.0
            elif it >= tmax:
                return n_min
            return scale_fn(0.5 * (1 + np.cos(np.pi * it / tmax)), n_min)

        scale_fn = log_decay if conf.type == "cos_log" else linear_decay

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, functools.partial(cosine_decay, scale_fn)
        )

    # backward compatibility
    def lr_fn(it):  # noqa: E306
        if conf.type is None:
            return 1
        if conf.type == "factor":
            return 1.0 if it < conf.start else conf.factor
        if conf.type == "exp":
            gam = 10 ** (-1 / conf.exp_div_10)
            return 1.0 if it < conf.start else gam
        else:
            raise ValueError(conf.type)

    return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)


def pack_lr_parameters(params, base_lr, lr_scaling):
    """Pack each group of parameters with the respective scaled learning rate."""
    filters, scales = tuple(zip(*[(n, s) for s, names in lr_scaling for n in names]))
    scale2params = collections.defaultdict(list)
    for n, p in params:
        scale = 1
        # TODO: use proper regexp rather than just this inclusion check
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))
    logger.info(
        "Parameters with scaled learning rate:\n%s",
        {s: [n for n, _ in ps] for s, ps in scale2params.items() if s != 1},
    )
    lr_params = [
        {"lr": scale * base_lr, "params": [p for _, p in ps]}
        for scale, ps in scale2params.items()
    ]
    return lr_params


def write_dict_summaries(writer, name, items, step):
    for k, v in items.items():
        key = f"{name}/{k}"
        if isinstance(v, dict):
            writer.add_scalars(key, v, step)
        elif isinstance(v, tuple):
            writer.add_pr_curve(key, *v, step)
        else:
            writer.add_scalar(key, v, step)


def write_image_summaries(writer, name, figures, step):
    # Stacked grayscale is not supported, convert to RGB!
    def _add_plot(tag, fig_or_image, step):
        if isinstance(fig_or_image, (np.ndarray, torch.Tensor)):
            if fig_or_image.ndim in (2, 3):
                writer.add_image(tag, fig_or_image, step)
            else:
                assert fig_or_image.ndim == 4
                writer.add_images(tag, fig_or_image, step)
        else:
            # Figure or list[Figure]
            writer.add_figure(tag, fig_or_image, step, close=True)

    if isinstance(figures, list):
        for i, figs in enumerate(figures):
            for k, fig in figs.items():
                _add_plot(f"{name}/{i}_{k}", fig, step)
    else:
        for k, fig in figures.items():
            _add_plot(f"{name}/{k}", fig, step)
