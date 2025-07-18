"""
Various handy Python and PyTorch utils.

Author: Paul-Edouard Sarlin (skydes)
"""

import collections
import os
import random
import time
from collections.abc import Iterable
from contextlib import contextmanager

import numpy as np
import torch


class AverageMetric:
    def __init__(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, tensor):
        assert tensor.dim() == 1
        self._sum += torch.nansum(tensor).item()
        self._num_examples += len(tensor)

    def compute(self):
        if self._num_examples == 0:
            return np.nan
        else:
            return self._sum / self._num_examples


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
        self._elements = elements
        self.thresholds = thresholds
        if not isinstance(thresholds, list):
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
