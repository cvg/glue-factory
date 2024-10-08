import logging
from random import shuffle
import torch


from gluefactory.datasets import BaseDataset, get_dataset
from gluefactory.datasets.oxford_paris_mini_1view_jpldd import OxfordParisMiniOneViewJPLDD
from gluefactory.datasets.minidepth import MiniDepthDataset


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
        #"respect_sub_dataset_splits": True,     # if True the validation set of the merged dataset will be the joint validation sets from all sub datasets
                                                # if False all images from all subdatasets are considered as a whole to generate splits
        "inter_dataset_shuffle": True,  # if True, all images are shuffled (from all datasets)
        "datasets": {  # Here list datasets with their (file)name. As an example we have Oxparis and Minidepth here
            "minidepth": {
                "name": "gluefactory.datasets.minidepth",
                **MiniDepthDataset.default_conf
            },
            "oxparis": {
                "name": "gluefactory.datasets.oxford_paris_mini_1view_jpldd",
                **OxfordParisMiniOneViewJPLDD.default_conf
            },
        },
    }

    def _init(self, conf):
        self.config = conf


    def get_dataset(self, split):
        return _Dataset(self.config, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        super().__init__()
        self.conf = conf
        # load datasets, set split
        #self.set_split_for_all_datasets(split)
        self.datasets = {}  # store dataset objects
        self.img_index_collection = []  # store image indices

        logging.info(f"Initialize Merged Dataset with following datasets: {conf['datasets'].keys()}")
        for key, dset_conf in conf["datasets"].items():
            dset = get_dataset(dset_conf.name)(dset_conf)
            dset_initialized = dset.get_dataset(conf.split)
            self.datasets[key] = dset_initialized
            num_img = len(dset_initialized)
            self.img_index_collection += [(key, i) for i in range(num_img)]

        if self.conf.inter_dataset_shuffle:
            shuffle(self.img_index_collection)

        logging.info(f"Merged Dataset using split {conf['split']} has {len(self.img_index_collection)} Images....")


    def __len__(self):
        return len(self.img_index_collection)


    def get_dataset(self, split):
        return self

    def __getitem__(self, idx):
        dataset_key, in_dataset_idx = self.img_index_collection[idx]
        return self.datasets[dataset_key][in_dataset_idx]
