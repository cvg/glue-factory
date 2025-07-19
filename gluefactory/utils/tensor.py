import functools
import inspect

import numpy as np
import torch


def autocast(func):
    """Cast the inputs of a TensorWrapper method to PyTorch tensors
    if they are numpy arrays. Use the device and dtype of the wrapper.
    """

    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device("cpu")
        dtype = None
        if isinstance(self, TensorWrapper):
            if self._data is not None:
                device = self.device
                dtype = self.dtype
        elif not inspect.isclass(self) or not issubclass(self, TensorWrapper):
            raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)

    return wrap


class TensorWrapper:
    """Wrapper for PyTorch tensors."""

    _data = None

    @autocast
    def __init__(self, data: torch.Tensor):
        """Wrapper for PyTorch tensors."""
        self._data = data

    @property
    def shape(self) -> torch.Size:
        """Shape of the underlying tensor."""
        return self._data.shape[:-1]

    @property
    def device(self) -> torch.device:
        """Get the device of the underlying tensor."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the underlying tensor."""
        return self._data.dtype

    def __getitem__(self, index) -> torch.Tensor:
        """Get the underlying tensor."""
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        """Set the underlying tensor."""
        self._data[index] = item.data

    def to(self, *args, **kwargs):
        """Move the underlying tensor to a new device."""
        return self.__class__(self._data.to(*args, **kwargs))

    def cpu(self):
        """Move the underlying tensor to the CPU."""
        return self.__class__(self._data.cpu())

    def cuda(self):
        """Move the underlying tensor to the GPU."""
        return self.__class__(self._data.cuda())

    def pin_memory(self):
        """Pin the underlying tensor to memory."""
        return self.__class__(self._data.pin_memory())

    def float(self):
        """Cast the underlying tensor to float."""
        return self.__class__(self._data.float())

    def double(self):
        """Cast the underlying tensor to double."""
        return self.__class__(self._data.double())

    def detach(self):
        """Detach the underlying tensor."""
        return self.__class__(self._data.detach())

    def numpy(self):
        """Convert the underlying tensor to a numpy array."""
        return self._data.detach().cpu().numpy()

    def new_tensor(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_tensor(*args, **kwargs)

    def new_zeros(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_zeros(*args, **kwargs)

    def new_ones(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_ones(*args, **kwargs)

    def new_full(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_full(*args, **kwargs)

    def new_empty(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_empty(*args, **kwargs)

    def unsqueeze(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self.__class__(self._data.unsqueeze(*args, **kwargs))

    def squeeze(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self.__class__(self._data.squeeze(*args, **kwargs))

    @classmethod
    def stack(cls, objects: list, dim=0, *, out=None):
        """Stack a list of objects with the same type and shape."""
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    @classmethod
    def cat(cls, objects: list, dim=0, *, out=None):
        if out is not None and isinstance(out, cls):
            out = out._data
        data = torch.cat([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    @classmethod
    def where(cls, condition, input, other, *, out=None):
        if not (isinstance(input, cls) and isinstance(other, cls)):
            raise ValueError(f"Incorrect inputs: {input}, {other}.")
        if out is not None and isinstance(out, cls):
            out = out._data
        ret = torch.where(condition.unsqueeze(-1), input._data, other._data, out=out)
        return cls(ret)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Support for torch functions."""
        if kwargs is None:
            kwargs = {}
        if func.__name__ == "stack":
            return cls.stack(*args, **kwargs)
        if func.__name__ == "cat":
            return cls.cat(*args, **kwargs)
        elif func.__name__ == "where":
            return cls.where(*args, **kwargs)
        else:
            return NotImplemented
