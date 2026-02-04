import functools
import inspect
from typing import Any, Callable, Dict, NamedTuple, Union

import numpy as np
import torch
import torch._C._functorch as cft  # type: ignore
from tensordict import TensorClass, tensorclass


def autocast(func):
    """Cast the inputs of a TensorWrapper method to PyTorch tensors
    if they are numpy arrays. Use the device and dtype of the wrapper.
    """

    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device("cpu")
        dtype = None
        if isinstance(self, torch.Tensor):
            device = self.device
            dtype = self.dtype
        # elif not inspect.isclass(self) or not issubclass(self, torch.Tensor):
        #     raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)

    return wrap


def autovmap(func):
    """Batch and vectorize a simple TensorWrapper method."""

    @functools.wraps(func)
    def wrap(self, arg):
        # assert isinstance(self, TensorWrapper), self
        cls = self.__class__
        _wrap = lambda d, x: wrap(cls(d), x)
        if arg.ndim == self.data_.ndim and arg.ndim == 1:
            return func(self, arg)
        elif (
            arg.ndim == 3
            and self.data_.ndim == 2
            and arg.shape[0] == self.data_.shape[0]
        ):
            # Handle the lazy scenario: wrapper BxP, arg BxNxD
            return torch.vmap(_wrap)(self.data_, arg)
        elif arg.ndim > self.data_.ndim:
            return torch.vmap(_wrap, in_dims=(None, 0))(self.data_, arg)
        else:
            arg = arg.broadcast_to(self.shape + arg.shape[-1:])
            if arg.ndim == self.data_.ndim:
                return torch.vmap(_wrap)(self.data_, arg)
            else:
                raise ValueError(
                    f"Broadcast failed: self.data_={self.data_.shape}, arg={arg.shape}."
                )

    return wrap


class TensorWrapper(TensorClass, tensor_only=True):
    """Wrapper for PyTorch tensors."""

    data_: torch.Tensor

    def __post_init__(self):
        self.batch_size = self.data_.shape[:-1]

    @property
    def device(self) -> torch.device:
        return self.data_.device

    def __deepcopy__(self, memo: Dict[int, Any]) -> "TensorWrapper":
        # The tensorclass __deepcopy__ will be deprecated, so we clone instead.
        return self.clone()

    @classmethod
    def where(cls, condition, input, other, *, out=None):
        if not (isinstance(input, cls) and isinstance(other, cls)):
            raise ValueError(f"Incorrect inputs: {input}, {other}.")
        if out is not None and isinstance(out, cls):
            out = out.data_
        ret = torch.where(condition.unsqueeze(-1), input.data_, other.data_, out=out)
        return cls(ret)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        if func == torch.concat:
            func = torch.cat
        if func == torch.where:
            func = cls.where
        return getattr(cls, func.__name__)(*args, **(kwargs or {}))

    def to_h5(self, grp, key: str, **kwargs) -> None:
        """Serialize to h5 dataset with _type attribute."""
        raise NotImplementedError(f"{self.__class__.__name__}.to_h5 not implemented")

    @classmethod
    def from_h5(cls, ds) -> "TensorWrapper":
        """Deserialize from h5 dataset."""
        raise NotImplementedError(f"{cls.__name__}.from_h5 not implemented")
