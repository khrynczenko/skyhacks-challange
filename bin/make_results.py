import os
import csv
import torch
import cv2
import numpy as np

from lib.models import ModelTask1, FCNN1, FCNN2, FCNN3
from lib.image import io
from torch import nn
from torch.nn.functional import softmax
from torchvision import transforms

# model = ModelTask1(53)
model_t1 = FCNN1(4)
model_t2 = FCNN2(4)
model_t3 = FCNN3(4)
weights_t1_path = "weights/FCNN1-e22.pt"
weights_t2_path = "weights/task2model-e8.pt"
weights_t3_path = "weights/task3model-e40.pt"
model_t1.load_state_dict(torch.load(weights_t1_path, map_location="cpu"))
model_t2.load_state_dict(torch.load(weights_t2_path, map_location="cpu"))
model_t3.load_state_dict(torch.load(weights_t3_path, map_location="cpu"))

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
with open("results-task1-e22-task2-e8-task3-e40.csv", "w", newline="") as csvfile:
    # writer = csv.DictWriter(csvfile,
    #                         fieldnames=['filename', 'standard', 'task2_class', 'tech_cond'] + labels_task_1)
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'standard', 'task2_class', 'tech_cond'] + labels_task_1)
    # writer.writeheader()
    for entry in os.scandir(test_image_directory):
        image_names.append(entry.name)
        image = io.read_image(entry.path) 
        image_t1 = cv2.resize(image, (500, 500))
        image_t2 = cv2.resize(image, (224, 224))
        image_t3 = image_t2

        # T1 RESULTS 
        tensored = transforms.ToTensor()(image_t1)
        as_batch = torch.unsqueeze(tensored, 0)
        output = model_t1(as_batch)
        result_t1_vector = output.detach().cpu().numpy()[0]
        result_t1_vector = [int(x) for x in np.array(result_t1_vector > 0.5, dtype=int)]

        # T2 RESULTS
        tensored = transforms.ToTensor()(image_t2)
        as_batch = torch.unsqueeze(tensored, 0)
        output = model_t2(as_batch)
        output = torch.argmax(output, dim=1)
        digit_to_string = {
            0: "house",
            1: "dining_room",
            2: "kitchen",
            3: "bathroom",
            4: "living_room",
            5: "bedroom",
        }
        room_type = digit_to_string[int(output)]


        # T3 RESULTS
        tensored = transforms.ToTensor()(image_t3)
        as_batch = torch.unsqueeze(tensored, 0)
        standard, tech_cond = model_t3(as_batch)
        standard, tech_cond = torch.argmax(softmax(standard)), torch.argmax(softmax(tech_cond))

        full_row = [entry.name, int(standard + 1), room_type, int(tech_cond + 1)] + result_t1_vector 
        full_row = [str(x) for x in full_row]
        writer.writerow(full_row)
