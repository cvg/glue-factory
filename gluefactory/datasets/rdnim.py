""" Rotated Day-Night Image Matching dataset. """

import os
import numpy as np
import logging
import torch
import cv2
from pathlib import Path
from gluefactory.datasets import BaseDataset
import zipfile
from gluefactory.utils.image import ImagePreprocessor, load_image
from gluefactory.datasets.base_dataset import BaseDataset
from gluefactory.datasets.utils import read_timestamps
from gluefactory.settings import DATA_PATH

logger = logging.getLogger(__name__)

class RDNIM(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        'preprocessing': ImagePreprocessor.default_conf,
        'data_dir': 'RDNIM',
        'reference': 'day',
        'grayscale': False
    }

    url = "https://cvg-data.inf.ethz.ch/RDNIM/RDNIM.zip"

    def download(self):
        data_dir = DATA_PATH
        zip_path = DATA_PATH / (self.conf.data_dir + ".zip")
        torch.hub.download_url_to_file(self.url, zip_path)
        with zipfile.ZipFile(zip_path,"r") as zip_file:
            zip_file.extractall(data_dir)
        zip_path.unlink()


    def _init(self, conf):
        self._root_dir = Path(DATA_PATH, conf.data_dir)
        if not self._root_dir.exists():
            logger.info("Downloading the RDNIM dataset.")
            self.download()
        ref = conf.reference
        self.preprocessor = ImagePreprocessor(conf.preprocessing)
        
        # Extract the timestamps
        timestamp_files = [p for p
                           in Path(self._root_dir, 'time_stamps').iterdir()]
        timestamps = {}
        for f in timestamp_files:
            id = f.stem
            timestamps[id] = read_timestamps(str(f))

        # Extract the reference images paths
        references = {}
        seq_paths = [p for p in Path(self._root_dir, 'references').iterdir()]
        for seq in seq_paths:
            id = seq.stem
            references[id] = str(Path(seq, ref + '.jpg'))

        # Extract the images paths and the homographies
        seq_path = [p for p in Path(self._root_dir, 'images').iterdir()]
        self._files = []
        for seq in seq_path:
            id = seq.stem
            images_path = [x for x in seq.iterdir() if x.suffix == '.jpg']
            for img in images_path:
                timestamp = timestamps[id]['time'][
                    timestamps[id]['name'].index(img.name)]
                H = np.loadtxt(str(img)[:-4] + '.txt').astype(float)
                self._files.append({
                    'img': str(img),
                    'ref': str(references[id]),
                    'H': H,
                    'timestamp': timestamp})

    def _read_image(self, idx: int, type: str) -> dict:
        img = load_image(self._files[idx][type], self.conf.grayscale)
        return self.preprocessor(img)

    def __getitem__(self, idx: int) -> dict:
        img0 = self._read_image(idx,"ref")
        img1 = self._read_image(idx,"img")
        #img_size = img0.shape[:2]
        H = self._files[idx]['H']
        H = img1["transform"] @ H  @ np.linalg.inv(img0["transform"])
        #H = torch.tensor(H, dtype=torch.float)

        return {
            'view0': img0, 
            'view1': img1, 
            'H_0to1': H,
            'idx': idx,
            'name': self._files[idx]["img"] + self.conf.reference,
            'timestamp': self._files[idx]['timestamp'],
        }

    def __len__(self):
        return len(self._files)

    def get_dataset(self, split: str) -> "RDNIM":
        assert split in ['test']
        return self

    # # Overwrite the parent data loader to handle custom collate_fn
    # def get_data_loader(self, split, shuffle=False):
    #     """Return a data loader for a given split."""
    #     assert split in ['test']
    #     batch_size = self.conf.get(split+'_batch_size')
    #     num_workers = self.conf.get('num_workers', batch_size)
    #     return DataLoader(self, batch_size=batch_size,
    #                       shuffle=shuffle or split == 'train',
    #                       pin_memory=True, num_workers=num_workers)