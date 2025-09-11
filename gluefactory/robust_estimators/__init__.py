import importlib
import inspect

from ..utils.tools import get_class
from .base_estimator import BaseEstimator


def load_estimator(type, estimator):
    import_paths = [
        estimator,
        f"{__name__}.{type}.{estimator}",
    ]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                return get_class(path, BaseEstimator)
            except AssertionError:
                continue
    raise RuntimeError(
        f'Model {estimator} not found in any of [{" ".join(import_paths)}]'
    )
