import os
import functools
import cv2
import numpy as np

from lib.models import FCNN1
from lib.training import train_model
from lib.data.dataset import ImageDataset, TransformedImageDataset

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.nn import BCELoss
from torchvision import transforms

if __name__ == '__main__':
    train_dir_path = r'D:/hackathon/main_task_data/'
    train_csv_path = r'data/task1.csv'
    artifacts_path = r'epochs-size500-fcnn-1/'
    os.makedirs(artifacts_path, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Prepare data
    resize = functools.partial(cv2.resize, dsize=(500, 500))
    to_tensor = transforms.ToTensor()
    whole_dataset = ImageDataset(train_dir_path, train_csv_path) 
    whole_dataset = TransformedImageDataset(whole_dataset, [resize, to_tensor])

    part = int(0.15 * len(whole_dataset)) - 1
    train_dataset, val_dataset = random_split(whole_dataset, (len(whole_dataset) - part, part))
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    model = FCNN1(4)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = BCELoss()

    dataloaders = {'train': train_dataloader,
                   'val': val_dataloader}
    best_model = train_model(model, dataloaders, criterion, optimizer,
                             num_epochs=500, device=device, artifacts_directory=artifacts_path)
    torch.save(model.state_dict(), os.path.join(artifacts_path, "best_model.pt"))
