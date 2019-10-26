import os

from lib.models import ModelTask1
from lib.training import train_model
from lib.data.dataset import ImageDataset, TensorImageDataset

import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.nn import MSELoss, BCEWithLogitsLoss

if __name__ == '__main__':
    train_dir_path = r'D:\Hypernet\skyhacks\main_task_data'
    val_dir_path = r'D:\Hypernet\skyhacks\main_task_data'
    train_csv_path = r'D:\Hypernet\skyhacks\skyhacks-challange-master\skyhacks-challange-master\data\task1_train.csv'
    val_csv_path = r'D:\Hypernet\skyhacks\skyhacks-challange-master\skyhacks-challange-master\data\task1_valid.csv'
    artifacts_path = r'D:\Hypernet\skyhacks\artifacts'
    os.makedirs(artifacts_path, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Prepare data
    train_dataset = TensorImageDataset(ImageDataset(train_dir_path, train_csv_path))
    val_dataset = TensorImageDataset(ImageDataset(val_dir_path, val_csv_path))
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
