import importlib.util

from ..utils.tools import get_class
from .base_model import BaseModel


def get_model(name):
    import_paths = [
        name,
        f"{__name__}.{name}",
        f"{__name__}.extractors.{name}",  # backward compatibility
        f"{__name__}.matchers.{name}",  # backward compatibility
    ]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                return get_class(path, BaseModel)
            except AssertionError:
                mod = __import__(path, fromlist=[""])
                try:
                    return mod.__main_model__
                except AttributeError as exc:
                    print(exc)
                    continue

    raise RuntimeError(f'Model {name} not found in any of [{" ".join(import_paths)}]')
