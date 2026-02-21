"""LR scheduling with fractional epoch tracking.

Always stepped per-iteration. Epoch-based schedules use a fractional epoch
counter so they interpolate smoothly within epochs, even when epoch lengths
vary.
"""

import functools
import math

import numpy as np
import torch


class EpochTracker:
    """Tracks fractional epochs from per-iteration steps."""

    def __init__(self):
        self.fractional_epoch = 0.0
        self._epoch_length = 1

    def set_epoch_length(self, n: int):
        self._epoch_length = n

    def step(self):
        self.fractional_epoch += 1.0 / self._epoch_length


def get_lr_scheduler(optimizer, conf, epoch_tracker: EpochTracker | None = None):
    """Build an LR scheduler that is always stepped per-iteration.

    When ``conf.on_epoch`` is True the schedule function receives
    ``epoch_tracker.fractional_epoch`` instead of the raw PyTorch step
    counter, so epoch-based schedules interpolate smoothly.

    Args:
        optimizer: The optimizer.
        conf: LR schedule config with fields: type, start, on_epoch,
              factor, exp_div_10, options, warmup, etc.
        epoch_tracker: Fractional epoch counter (required when on_epoch=True).
    """
    on_epoch = getattr(conf, "on_epoch", False)
    warmup = getattr(conf, "warmup", 0.0)
    interpolate_epoch = getattr(conf, "interpolate_epoch", False)

    def _wrap(fn):
        """Wrap a schedule fn: read fractional_epoch when on_epoch, add warmup."""

        def wrapped(_it):
            if on_epoch and epoch_tracker is not None:
                t = epoch_tracker.fractional_epoch
                if not interpolate_epoch:
                    t = int(t)  # staircase: constant within each epoch
            else:
                t = _it

            # Linear warmup: warmup is in same units as the schedule
            # (epochs if on_epoch, iterations if not).
            # Warmup always interpolates smoothly, even with staircase.
            if warmup > 0:
                if on_epoch and epoch_tracker is not None:
                    w = epoch_tracker.fractional_epoch
                else:
                    w = _it
                if w < warmup:
                    return w / warmup
            return fn(t)

        return wrapped

    stype = getattr(conf, "type", None)

    # Passthrough to torch.optim.lr_scheduler.* (e.g. SequentialLR, ChainedScheduler)
    if stype not in ["factor", "exp", "cos", "cos_log", None]:
        options = getattr(conf, "options", {})
        if hasattr(options, "schedulers"):
            schedulers = []
            for scheduler_conf in options.schedulers:
                schedulers.append(get_lr_scheduler(optimizer, scheduler_conf, epoch_tracker))
            opts = {k: v for k, v in options.items() if k != "schedulers"}
            return getattr(torch.optim.lr_scheduler, stype)(
                optimizer, schedulers, **opts
            )
        return getattr(torch.optim.lr_scheduler, stype)(optimizer, **options)

    if stype is not None and stype.startswith("cos"):

        def log_decay(x: float, end_val: float) -> float:
            return 10 ** (np.log10(end_val) * (1 - x))

        def linear_decay(x: float, end_val: float) -> float:
            return x * (1 - end_val) + end_val

        def cosine_decay(scale_fn, it):
            n_min = getattr(conf, "min_factor", 0.0)
            start = getattr(conf, "start", 0)
            end = getattr(conf, "end", 100)
            tmax = end - start
            it = it - start
            if it < 0:
                return 1.0
            elif it >= tmax:
                return n_min
            return scale_fn(0.5 * (1 + math.cos(math.pi * it / tmax)), n_min)

        scale_fn = log_decay if stype == "cos_log" else linear_decay
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, _wrap(functools.partial(cosine_decay, scale_fn))
        )

    # factor / exp / None
    def lr_fn(it):
        if stype is None:
            return 1.0
        if stype == "factor":
            start = getattr(conf, "start", 0)
            factor = getattr(conf, "factor", 1.0)
            return 1.0 if it < start else factor
        if stype == "exp":
            start = getattr(conf, "start", 0)
            exp_div_10 = getattr(conf, "exp_div_10", 0)
            gam = 10 ** (-1 / exp_div_10) if exp_div_10 else 1.0
            return 1.0 if it < start else gam
        raise ValueError(stype)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _wrap(lr_fn))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def _test():
    """Test LR schedule with varying epoch lengths and warmup."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from types import SimpleNamespace

    base_lr = 1e-3

    # --- Test 1: Cosine decay with warmup, on_epoch=True, varying epoch lengths ---
    print("Test 1: cosine + warmup (on_epoch=True, varying epoch lengths)")
    tracker = EpochTracker()
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=base_lr)
    conf = SimpleNamespace(
        type="cos",
        on_epoch=True,
        start=0,
        end=5,  # 5 epochs of cosine decay (after warmup)
        min_factor=0.01,
        warmup=0.5,
    )
    sched = get_lr_scheduler(opt, conf, tracker)

    epoch_lengths = [100, 80, 120, 90, 110, 100]  # 6 epochs, varying lengths
    lrs, epochs = [], []
    for epoch_len in epoch_lengths:
        tracker.set_epoch_length(epoch_len)
        for _ in range(epoch_len):
            lrs.append(opt.param_groups[0]["lr"])
            epochs.append(tracker.fractional_epoch)
            tracker.step()
            sched.step()

    # --- Test 1b: Staircase (interpolate_epoch=False) ---
    print("Test 1b: cosine + warmup (on_epoch=True, staircase)")
    tracker1b = EpochTracker()
    opt1b = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    conf1b = SimpleNamespace(
        type="cos",
        on_epoch=True,
        interpolate_epoch=False,
        start=0,
        end=5,
        min_factor=0.01,
        warmup=0.5,
    )
    sched1b = get_lr_scheduler(opt1b, conf1b, tracker1b)

    lrs1b, epochs1b = [], []
    for epoch_len in epoch_lengths:
        tracker1b.set_epoch_length(epoch_len)
        for _ in range(epoch_len):
            lrs1b.append(opt1b.param_groups[0]["lr"])
            epochs1b.append(tracker1b.fractional_epoch)
            tracker1b.step()
            sched1b.step()

    # --- Test 2: Cosine decay with warmup, on_epoch=False (per-iteration) ---
    print("Test 2: cosine + warmup (on_epoch=False)")
    tracker2 = EpochTracker()
    opt2 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    total_iters = sum(epoch_lengths)
    conf2 = SimpleNamespace(
        type="cos",
        on_epoch=False,
        start=0,
        end=total_iters,  # in iterations
        min_factor=0.01,
        warmup=0.5 * epoch_lengths[0],  # 0.5 epochs in iteration units
    )
    sched2 = get_lr_scheduler(opt2, conf2, tracker2)

    lrs2, epochs2 = [], []
    for epoch_len in epoch_lengths:
        tracker2.set_epoch_length(epoch_len)
        for _ in range(epoch_len):
            lrs2.append(opt2.param_groups[0]["lr"])
            epochs2.append(tracker2.fractional_epoch)
            tracker2.step()
            sched2.step()

    # --- Plot all three (x-axis = iterations, epoch boundaries as vertical lines) ---
    epoch_boundaries = list(np.cumsum(epoch_lengths))
    iters1 = list(range(len(lrs)))
    iters1b = list(range(len(lrs1b)))
    iters2 = list(range(len(lrs2)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Warmup ends at 0.5 epochs = 0.5 * first epoch length in iterations
    warmup_iter = int(0.5 * epoch_lengths[0])

    for ax, x, y, title in [
        (ax1, iters1, lrs, "cosine + warmup (on_epoch, smooth)"),
        (ax2, iters1b, lrs1b, "cosine + warmup (on_epoch, staircase)"),
        (ax3, iters2, lrs2, "cosine + warmup (on_iter)"),
    ]:
        ax.plot(x, y)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("LR")
        ax.set_title(title)
        ax.axvline(x=warmup_iter, color="r", linestyle="--", label="warmup end")
        for eb in epoch_boundaries[:-1]:
            ax.axvline(x=eb, color="gray", linestyle=":", alpha=0.5,
                        label="epoch" if eb == epoch_boundaries[0] else None)
        ax.legend()

    plt.tight_layout()
    plt.savefig("outputs/lr_schedule_test.png", dpi=150)
    print("Saved plot to outputs/lr_schedule_test.png")

    # --- Test 3: Constant LR with warmup ---
    print("Test 3: constant + warmup")
    tracker3 = EpochTracker()
    opt3 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    conf3 = SimpleNamespace(
        type=None,
        on_epoch=False,
        warmup=1.0,
    )
    sched3 = get_lr_scheduler(opt3, conf3, tracker3)

    lrs3 = []
    for epoch_len in epoch_lengths[:3]:
        tracker3.set_epoch_length(epoch_len)
        for _ in range(epoch_len):
            lrs3.append(opt3.param_groups[0]["lr"])
            tracker3.step()
            sched3.step()

    assert lrs3[0] < base_lr * 0.05, f"LR should start near 0, got {lrs3[0]}"
    assert abs(lrs3[-1] - base_lr) < 1e-6, f"LR should reach base_lr, got {lrs3[-1]}"
    print(f"  Start LR: {lrs3[0]:.6f}, End LR: {lrs3[-1]:.6f} — OK")

    # --- Test 4: Verify warmup is smooth across varying epoch lengths ---
    print("Test 4: warmup smoothness with varying epoch lengths")
    tracker4 = EpochTracker()
    opt4 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    conf4 = SimpleNamespace(type=None, on_epoch=False, warmup=2.0)
    sched4 = get_lr_scheduler(opt4, conf4, tracker4)

    lrs4 = []
    varying_lengths = [50, 200]  # epoch 0 short, epoch 1 long
    for epoch_len in varying_lengths:
        tracker4.set_epoch_length(epoch_len)
        for _ in range(epoch_len):
            lrs4.append(opt4.param_groups[0]["lr"])
            tracker4.step()
            sched4.step()

    # LR should be monotonically increasing during warmup
    for i in range(1, len(lrs4)):
        assert lrs4[i] >= lrs4[i - 1] - 1e-9, (
            f"LR not monotonic at iter {i}: {lrs4[i-1]:.6f} -> {lrs4[i]:.6f}"
        )
    print("  Monotonically increasing — OK")

    print("\nAll tests passed.")


if __name__ == "__main__":
    _test()
