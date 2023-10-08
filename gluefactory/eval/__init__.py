import torch

from ..utils.tools import get_class
from .eval_pipeline import EvalPipeline


def get_benchmark(benchmark):
    return get_class(f"{__name__}.{benchmark}", EvalPipeline)


@torch.no_grad()
def run_benchmark(benchmark, eval_conf, experiment_dir, model=None):
    """This overwrites existing benchmarks"""
    experiment_dir.mkdir(exist_ok=True, parents=True)
    bm = get_benchmark(benchmark)

    pipeline = bm(eval_conf)
    return pipeline.run(
        experiment_dir, model=model, overwrite=True, overwrite_eval=True
    )
