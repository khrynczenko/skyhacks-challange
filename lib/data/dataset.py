import enum
import glob
import os
from typing import Iterable, Callable, Union

import cv2
import pandas as pd
import numpy as np
from torch.utils import data
from torchvision import transforms
from lib.image import io
import torch


class Column(enum.Enum):
    FILENAME = 'filename'


class ImageDataset(data.Dataset):
    def __init__(self, dir_path: str, csv_path: str):
        self.dir_path = dir_path
        self.df = pd.read_csv(csv_path)

    def __getitem__(self, item):
        img_name = self.df.iloc[item, 0]
        paths = glob.glob(os.path.join(self.dir_path, '*', img_name))
        assert len(paths) == 1
        img_path = paths[0]
        img = io.read_image(img_path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = img[..., :-1]
        return img, torch.FloatTensor(np.asarray(self.df.iloc[item, 1:], dtype=np.uint8))

    def __len__(self):
        return len(self.df.shape[0])


class TransformedImageDataset(data.Dataset):
    def __init__(self, dataset: ImageDataset,
                 image_transformations: Iterable[Callable[[np.ndarray], np.ndarray]]):
        self._dataset = dataset
        self._transformations = image_transformations

    def __getitem__(self, item):
        img, label = self._dataset[item]
        for transformation in self._transformations:
            img = transformation(img)
        return img, label

    def __len__(self):
        return len(self._dataset)
