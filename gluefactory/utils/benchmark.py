import time

import numpy as np
import torch


def benchmark(model, data, device, r=100):
    timings = np.zeros((r, 1))
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    # warmup
    for _ in range(10):
        _ = model(data)
    # measurements
    with torch.no_grad():
        for rep in range(r):
            if device.type == "cuda":
                starter.record()
                _ = model(data)
                ender.record()
                # sync gpu
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
            else:
                start = time.perf_counter()
                _ = model(data)
                curr_time = (time.perf_counter() - start) * 1e3
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / r
    std_syn = np.std(timings)
    return {"mean": mean_syn, "std": std_syn}
