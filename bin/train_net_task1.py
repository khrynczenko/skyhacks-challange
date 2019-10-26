import os
import functools
import cv2

from lib.models import ModelTask1
from lib.training import train_model
from lib.data.dataset import ImageDataset, TransformedImageDataset

import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.nn import MSELoss, BCEWithLogitsLoss
from torchvision import transforms

if __name__ == '__main__':
    train_dir_path = r'D:\Hypernet\skyhacks\main_task_data'
    val_dir_path = r'D:\Hypernet\skyhacks\main_task_data'
    train_csv_path = r'D:\Hypernet\skyhacks\skyhacks-challange-master\skyhacks-challange-master\data\task1_train.csv'
    val_csv_path = r'D:\Hypernet\skyhacks\skyhacks-challange-master\skyhacks-challange-master\data\task1_valid.csv'
    artifacts_path = r'D:\Hypernet\skyhacks\artifacts'
    os.makedirs(artifacts_path, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Prepare data
    resize = functools.partial(cv2.resize, dsize=(224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_dataset = TransformedImageDataset(ImageDataset(train_dir_path, train_csv_path), [resize, to_tensor, normalize])
    val_dataset = TransformedImageDataset(ImageDataset(val_dir_path, val_csv_path), [resize, to_tensor, normalize])

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = ModelTask1(53)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = BCEWithLogitsLoss()

    dataloaders = {'train': train_dataloader,
                   'val': val_dataloader}
    best_model = train_model(model, dataloaders, criterion, optimizer,
                             num_epochs=100, device=device)
    torch.save(model.state_dict(), os.path.join(artifacts_path, "best_model.pt"))
