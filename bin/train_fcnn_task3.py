import os
import functools

from lib.models import FCNN3
from lib.training import train_model_3
from lib.data.dataset import ImageDataset, TransformedImageDataset
import cv2
import torch
from torch.optim import Adam

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision import transforms

if __name__ == '__main__':
    train_dir_path = r'C:\skyhacks\main_task_data'
    val_dir_path = r'C:\skyhacks\main_task_data'
    train_csv_path = r'C:\Users\Michał Myller\PycharmProjects\skyhacks-challange\data\task3_train.csv'
    val_csv_path = r'C:\Users\Michał Myller\PycharmProjects\skyhacks-challange\data\task3_valid.csv'
    artifacts_path = r'epochs-xd/'
    os.makedirs(artifacts_path, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Prepare data
    to_tensor = transforms.ToTensor()
    resize = functools.partial(cv2.resize, dsize=(224, 224))
    train_dataset = TransformedImageDataset(
        ImageDataset(train_dir_path, train_csv_path), [resize, to_tensor])
    val_dataset = TransformedImageDataset(
        ImageDataset(val_dir_path, val_csv_path), [resize, to_tensor])
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    model = FCNN3(4)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    dataloaders = {'train': train_dataloader,
                   'val': val_dataloader}
    best_model = train_model_3(model, dataloaders, criterion, optimizer,
                             num_epochs=2, device=device)
    torch.save(model.state_dict(),
               os.path.join(artifacts_path, "best_model.pt"))
