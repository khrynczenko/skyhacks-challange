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
