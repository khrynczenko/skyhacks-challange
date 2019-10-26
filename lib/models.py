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
    def __init__(self):
        super(ResNet, self).__init__()
        original_model = models.vgg19(pretrained=True).eval()
        self.features = nn.Sequential(*list(original_model.features.children()))
        for p in self.features.parameters():
            p.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 6),
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(1, 512 * 7 * 7)
        x = self.classifier(x)
        return x
