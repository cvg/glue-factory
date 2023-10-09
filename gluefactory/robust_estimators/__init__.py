import inspect

from .base_estimator import BaseEstimator


def load_estimator(type, estimator):
    module_path = f"{__name__}.{type}.{estimator}"
    module = __import__(module_path, fromlist=[""])
    classes = inspect.getmembers(module, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseEstimator)]
    assert len(classes) == 1, classes
    return classes[0][1]
