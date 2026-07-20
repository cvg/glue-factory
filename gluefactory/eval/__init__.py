import importlib.util

import torch

from ..settings import SUBMODULES
from ..utils.tools import get_class
from .eval_pipeline import EvalPipeline


def get_benchmark(name):
    import_paths = [name, f"{__name__}.{name}"] + [f"{sm}.{name}" for sm in SUBMODULES]
    import_paths += [f"{sm}.eval.{name}" for sm in SUBMODULES]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                return get_class(path, EvalPipeline)
            except AssertionError as exc:
                continue
    raise RuntimeError(
        f'Benchmark {name} not found in any of [{" ".join(import_paths)}]'
    )


@torch.no_grad()
def run_benchmark(benchmark, eval_conf, experiment_dir, model=None):
    """This overwrites existing benchmarks"""
    experiment_dir.mkdir(exist_ok=True, parents=True)
    bm = get_benchmark(benchmark)

    pipeline = bm(eval_conf)
    return pipeline.run(
        experiment_dir, model=model, overwrite=True, overwrite_eval=True
    )
