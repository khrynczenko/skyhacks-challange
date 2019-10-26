import enum
import glob
import os

import cv2
import pandas as pd
from torch.utils import data
from torchvision import transforms
from lib.image import io
import torch


class Column(enum.Enum):
    FILENAME = 'filename'


class ImageDataset(data.Dataset):
    def __init__(self, dir_path: str, csv_path: str):
        paths = glob.glob(os.path.join(dir_path, '*', '*'))
        df = pd.read_csv(csv_path)
        img_names = list(df[Column.FILENAME.value])
        del df[Column.FILENAME.value]
        self.data = []
        self.labels = []
        paths_names = [os.path.basename(elem) for elem in paths]
        for i, name in enumerate(img_names):
            if name in paths_names:
                self.data.append(paths[paths_names.index(name)])
                self.labels.append(df.iloc[i].to_numpy())

    def __getitem__(self, item):
        img = io.read_image(self.data[item])
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img, self.labels[item]

    def __len__(self):
        return len(self.data)


class TensorImageDataset(data.Dataset):
    def __init__(self, dataset: ImageDataset):
        self.dataset = dataset
        self.transorms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, item):
        img, label = self.dataset[item]
        return self.transorms(img), torch.FloatTensor(label)

    def __len__(self):
        return len(self.dataset)
