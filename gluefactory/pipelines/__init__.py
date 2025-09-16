"""Pipelines are I/O intensive and not differentiable."""

import importlib
import importlib.util

from ..utils.tools import get_class


def get_pipeline(baseclass, name):
    import_paths = [
        name,
        f"{__name__}.{name}",
    ]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                return get_class(path, baseclass)
            except AssertionError:
                mod = __import__(path, fromlist=[""])
                try:
                    return mod.__main_model__
                except AttributeError as exc:
                    print(exc)
                    continue

    raise RuntimeError(
        f'Pipeline {name} not found in any of [{" ".join(import_paths)}]'
    )
