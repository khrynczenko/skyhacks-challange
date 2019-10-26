import numpy as np
import torch

from torch import nn, Tensor
from torchvision import models


class ModelTask1(nn.Sequential):
    def __init__(self, no_of_classes, num_of_neurons: int = 512,
                 freeze_extractor_weights: bool = True):
        super(ModelTask1, self).__init__()
        self.extractor = models.resnet50(pretrained=True)
        if freeze_extractor_weights:
            for param in self.extractor.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1000, out_features=num_of_neurons),
            nn.ReLU(),
            nn.Linear(in_features=num_of_neurons, out_features=no_of_classes),
            nn.Sigmoid()
        )

    def forward(self, input: Tensor) -> Tensor:
        x = self.extractor(input)
        return self.classifier(x)


class ModelTask2(nn.Module):
    def __init__(self):
        super(ModelTask2, self).__init__()
        original_model = models.resnet18(pretrained=True)
        modules = list(original_model.children())[:-1]
        self.features = nn.Sequential(*modules)
        for param in self.features.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(original_model.fc.in_features, 6)

    def forward(self, input):
        out = self.features(input)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class FCNN(nn.Module):
    def __init__(self, non_linear_mapping_layers: int):
        super().__init__()
        self.feature_extraction_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=56,
                      kernel_size=5,
                      padding=2),
            nn.PReLU())
        self.feature_extraction_2 = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=20,
                      kernel_size=5, padding=2),
            nn.PReLU())
        self.non_linear_mapping = []
        for _ in range(non_linear_mapping_layers):
            self.non_linear_mapping.append(
                nn.Conv2d(in_channels=20, out_channels=20,
                          kernel_size=3,
                          padding=1))
            self.non_linear_mapping.append(nn.PReLU())
        self.adaptive_pooling = nn.AdaptiveMaxPool2d((10, 10))
        self.fc = nn.Sequential(
            nn.Linear(in_features=2000, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=53),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extraction_1(x)
        x = self.feature_extraction_2(x)
        for mapping in self.non_linear_mapping:
            x = mapping(x)
        x = self.adaptive_pooling(x)
        x = x.reshape((1, 1, 2000))
        x = self.fc(x)
        return x