import torch

from torch import nn, Tensor
from torch.nn.init import kaiming_normal
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


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        original_model = models.resnet18(pretrained=True)

        self.features = nn.Sequential(*list(original_model.children())[:-1])

        num_feats = original_model.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(num_feats, num_classes)
        )

        for m in self.classifier:
            kaiming_normal(m.weight)

        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
