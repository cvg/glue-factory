"""
Base class for dataset.
See mnist.py for an example of dataset.
"""

import collections
import logging
from abc import ABCMeta, abstractmethod

import omegaconf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Sampler, get_worker_info
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
)

from ..utils.tensor import string_classes
from ..utils.tools import set_num_threads, set_seed

logger = logging.getLogger(__name__)


class LoopSampler(Sampler):
    def __init__(self, loop_size, total_size=None):
        self.loop_size = loop_size
        self.total_size = total_size - (total_size % loop_size)

    def __iter__(self):
        return (i % self.loop_size for i in range(self.total_size))

    def __len__(self):
        return self.total_size


def worker_init_fn(i):
    info = get_worker_info()
    if hasattr(info.dataset, "conf"):
        conf = info.dataset.conf
        set_seed(info.id + conf.seed)
        set_num_threads(conf.num_threads)
    else:
        set_num_threads(1)


def collate(batch):
    """Difference with PyTorch default_collate: it can stack of other objects."""
    if not isinstance(batch, list):  # no batching
        return batch
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            try:
                storage = elem.untyped_storage()._new_shared(numel)  # noqa: F841
            except AttributeError:
                storage = elem.storage()._new_shared(numel)  # noqa: F841
        return torch.stack(batch, dim=0)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif elem is None:
        return elem
    else:
        # try to stack anyway in case the object implements stacking.
        return torch.stack(batch, 0)


class BaseDataset(metaclass=ABCMeta):
    """
    What the dataset model is expect to declare:
        default_conf: dictionary of the default configuration of the dataset.
        It overwrites base_default_conf in BaseModel, and it is overwritten by
        the user-provided configuration passed to __init__.
        Configurations can be nested.

        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unknown configuration entries will raise an error.

        get_dataset(self, split): method that returns an instance of
        torch.utils.data.Dataset corresponding to the requested split string,
        which can be `'train'`, `'val'`, or `'test'`.
    """

    base_default_conf = {
        "name": "???",
        "num_workers": "???",
        "train_batch_size": "???",
        "val_batch_size": "???",
        "test_batch_size": "???",
        "shuffle_training": True,
        "batch_size": 1,
        "num_threads": 1,
        "seed": 0,
        "prefetch_factor": 2,
    }
    default_conf = {}

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf),
            OmegaConf.create(self.default_conf),
        )
        OmegaConf.set_struct(default_conf, True)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(self.conf, True)
        logger.info(f"Creating dataset {self.__class__.__name__}")
        self._init(self.conf)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def get_dataset(self, split):
        """To be implemented by the child class."""
        raise NotImplementedError

    def get_data_loader(self, split, shuffle=None, pinned=False, distributed=False):
        """Return a data loader for a given split."""
        assert split in ["train", "val", "test"]
        dataset = self.get_dataset(split)
        try:
            batch_size = self.conf[split + "_batch_size"]
        except omegaconf.MissingMandatoryValue:
            batch_size = self.conf.batch_size
        num_workers = self.conf.get("num_workers", batch_size)
        drop_last = True if split == "train" else False
        if distributed:
            shuffle = False
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, drop_last=drop_last
            )
        else:
            sampler = None
            if shuffle is None:
                shuffle = split == "train" and self.conf.shuffle_training
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=pinned,
            collate_fn=collate,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            prefetch_factor=self.conf.prefetch_factor,
            drop_last=drop_last,
        )

    def get_overfit_loader(self, split):
        """Return an overfit data loader.
        The training set is composed of a single duplicated batch, while
        the validation and test sets contain a single copy of this same batch.
        This is useful to debug a model and make sure that losses and metrics
        correlate well.
        """
        assert split in ["train", "val", "test"]
        dataset = self.get_dataset("train")
        sampler = LoopSampler(
            self.conf.batch_size,
            len(dataset) if split == "train" else self.conf.batch_size,
        )
        num_workers = self.conf.get("num_workers", self.conf.batch_size)
        return DataLoader(
            dataset,
            batch_size=self.conf.batch_size,
            pin_memory=True,
            num_workers=num_workers,
            sampler=sampler,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
