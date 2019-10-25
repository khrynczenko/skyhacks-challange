import enum
import glob
import os

import cv2
import pandas as pd
from torch.utils import data


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
        paths_names = [elem[elem.rindex('\\') + 1:] for elem in paths]
        for i, name in enumerate(img_names):
            if name in paths_names:
                self.data.append(paths[paths_names.index(name)])
                self.labels.append(df.iloc[i].to_numpy())

    def __getitem__(self, item):
        img = cv2.imread(self.data[item])
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img, self.labels[item]

    def __len__(self):
        return len(self.data)
