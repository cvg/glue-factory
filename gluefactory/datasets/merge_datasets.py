import logging
from random import shuffle

import torch
from omegaconf import OmegaConf

from gluefactory.datasets import BaseDataset, get_dataset
from gluefactory.datasets.minidepth import MiniDepthDataset
from gluefactory.datasets.oxford_paris_mini_1view_jpldd import (
    OxfordParisMiniOneViewJPLDD,
)
from gluefactory.datasets.scannet import Scannet

logger = logging.getLogger(__name__)


class MergedDataset(BaseDataset):
    """
    Merges 2 or more datasets.
    It simply is a wrapper for getting images with or without groundtruth from both.
    It can overwrite split of child datasets but this can also be done in child ds.
    All other configs should be done in child datasets
    """

    default_conf = {
        "train_batch_size": 2,  # prefix must match split
        "test_batch_size": 1,
        "val_batch_size": 1,
        "all_batch_size": 1,
        "split": "train",  # train, val, test
        "seed": 0,
        "num_workers": 0,  # number of workers used by the Dataloader
        "prefetch_factor": None,
        "inter_dataset_shuffle": True,  # if True, all images are shuffled (from all datasets) -> scale selection needs to be random in this case as otherwise datsaets will have always choosing same size
        "use_multiscale_learning": True,  # if True, we assume that all datasets included use multiscale learning. -> will make the datasets output same size for a batch
        "datasets": {  # Here list datasets with their (file)name. As an example we have Oxparis and Minidepth here
            #"minidepth": {
            #    "name": "gluefactory.datasets.minidepth",
            #    **MiniDepthDataset.default_conf,
            #},
            "oxparis": {
                "name": "gluefactory.datasets.oxford_paris_mini_1view_jpldd",
                **OxfordParisMiniOneViewJPLDD.default_conf,
            },
            "scannet": {
                "name": "gluefactory.datasets.scannet",
                **Scannet.default_conf,
            }
        },
    }

    def _init(self, conf):
        # if multiscale learning is activated for this dataset, check that all sub datsets have it activated as well.
        # Also make sure the same sclaes list is used
        if conf["use_multiscale_learning"]:
            for dset_key, dset_conf in conf["datasets"].items():
                assert dset_conf["multiscale_learning"]["do"]
        self.config = conf

    def get_dataset(self, split):
        return _Dataset(self.config, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        super().__init__()
        self.conf = conf
        # load datasets, set split
        # self.set_split_for_all_datasets(split)
        self.datasets = {}  # store dataset objects
        self.img_index_collection = []  # store image indices
        if self.conf.use_multiscale_learning:
            self.relevant_batch_size = self.conf[f"{split}_batch_size"]
            self.num_selected_this_batch = 0
            self.current_scale = None

        logging.info(
            f"Initialize Merged Dataset with following datasets: {conf['datasets'].keys()}"
        )
        for key, dset_conf in conf["datasets"].items():
            if self.conf.use_multiscale_learning:
                # 1st check if mulitscale learning scale selection is random for all
                scale_selection = dset_conf["multiscale_learning"]["scale_selection"]
                assert scale_selection == "random"
                # 2nd set batch size of sub datasets to overall batch size so that sub datasets don't change scale mid loading
                if OmegaConf.is_readonly(dset_conf):
                    OmegaConf.set_readonly(dset_conf, False)
                OmegaConf.update(
                    dset_conf,
                    f"{split}_batch_size",
                    self.relevant_batch_size,
                    force_add=True,
                )
                OmegaConf.set_readonly(dset_conf, True)
                assert dset_conf[f"{split}_batch_size"] == self.relevant_batch_size
            # Now initialize
            dset = get_dataset(dset_conf.name)(dset_conf)
            dset_initialized = dset.get_dataset(split)
            self.datasets[key] = dset_initialized
            num_img = len(dset_initialized)
            self.img_index_collection += [(key, i) for i in range(num_img)]

        if self.conf.inter_dataset_shuffle:
            shuffle(self.img_index_collection)

        logging.info(
            f"Merged Dataset using split {conf['split']} has {len(self.img_index_collection)} Images...."
        )

    def __len__(self):
        return len(self.img_index_collection)

    def get_dataset(self, split):
        return self

    def __getitem__(self, idx):
        dataset_key, in_dataset_idx = self.img_index_collection[idx]
        dset = self.datasets[dataset_key]
        logging.debug(f"Image from {dataset_key}")
        if not self.conf.use_multiscale_learning:
            return self.datasets[dataset_key][in_dataset_idx]
        else:
            # If multiscale learning is activated behaviour will be:
            #  -> At begin of new batch: Select random element from dataset (by setting its current selected items for batch to 0 the dataset will choose a size randomly (RANDOM SCALE SELECTION NEEDED))
            #  -> During batch: set current size of dataset to current size and set counter to 1 (this is needed so dset does not change size itself)
            if self.is_new_batch_starting_now():
                dset.set_num_selected_with_current_scale(0)
                img_data = dset[in_dataset_idx]
                self.num_selected_this_batch += 1
                self.current_scale = dset.get_current_scale()
                logging.debug(f"New batch start, chose size {self.current_scale}")
                return img_data
            else:
                logging.debug(f"in batch, current_scale: {self.current_scale}")
                dset.set_current_scale(self.current_scale)
                dset.set_num_selected_with_current_scale(1)
                self.num_selected_this_batch += 1
                return dset[in_dataset_idx]

    def is_new_batch_starting_now(self) -> bool:
        """
        Based on current state descides whether to change shape to reshape images to.
        This decision is needed as all images in a batch need same shape. So we only potentially change shape
        when a new batch is starting.

        Returns:
            bool: should shape be potentially changed?
        """
        # check if batch changes
        if self.num_selected_this_batch % self.relevant_batch_size == 0:
            self.num_selected_this_batch = (
                0  # Initially OR if batch changes set counter to 0
            )
            return True
        else:
            return False
