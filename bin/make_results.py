import os
import csv
import torch
import cv2
import numpy as np

from lib.models import ModelTask1, FCNN
from lib.image import io
from torch import nn

from torchvision import transforms

# model = ModelTask1(53)
model = FCNN(4)
weights_path = "weights/FCNN-125-BN-DROPOUT.pt"
model.load_state_dict(torch.load(weights_path, map_location="cpu"))

test_image_directory = "test_dataset"

labels_task_1 = ['Bathroom', 'Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Bed', 'Bed frame',
                 'Bed sheet', 'Bedroom', 'Cabinetry', 'Ceiling', 'Chair', 'Chandelier', 'Chest of drawers',
                 'Coffee table', 'Couch', 'Countertop', 'Cupboard', 'Curtain', 'Dining room', 'Door', 'Drawer',
                 'Facade', 'Fireplace', 'Floor', 'Furniture', 'Grass', 'Hardwood', 'House', 'Kitchen',
                 'Kitchen & dining room table', 'Kitchen stove', 'Living room', 'Mattress', 'Nightstand',
                 'Plumbing fixture', 'Property', 'Real estate', 'Refrigerator', 'Roof', 'Room', 'Rural area',
                 'Shower', 'Sink', 'Sky', 'Table', 'Tablecloth', 'Tap', 'Tile', 'Toilet', 'Tree', 'Urban area',
                 'Wall', 'Window']

image_names = []
images = []
with open("results-FCNN-125-BN-DROPOUT.csv", "w", newline="") as csvfile:
    # writer = csv.DictWriter(csvfile,
    #                         fieldnames=['filename', 'standard', 'task2_class', 'tech_cond'] + labels_task_1)
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'standard', 'task2_class', 'tech_cond'] + labels_task_1)
    # writer.writeheader()
    for entry in os.scandir(test_image_directory):
        image_names.append(entry.name)
        image = io.read_image(entry.path) 
        image = cv2.resize(image, (224, 224))
        images.append(image)
        tensored = transforms.ToTensor()(image)
        as_batch = torch.unsqueeze(tensored, 0)
        output = model(as_batch)
        # output = nn.Sigmoid()(output)
        result_vector = output.detach().cpu().numpy()[0]
        result_vector = [int(x) for x in np.array(result_vector > 0.5, dtype=int)]
        full_row = [entry.name, 3, "bathroom", 4] + result_vector
        full_row = [str(x) for x in full_row]
        writer.writerow(full_row)
