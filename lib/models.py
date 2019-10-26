from torch import nn, Tensor
from torchvision import models


class ModelTask1(nn.Sequential):
    def __init__(self, no_of_classes, num_of_neurons: int = 64,
                 freeze_extractor_weights: bool = True):
        super(ModelTask1, self).__init__()
        self.extractor = models.resnet34(pretrained=True)
        if freeze_extractor_weights:
            for param in self.extractor.parameters():
                param.requires_grad = False
        num_ftrs = self.extractor.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_ftrs, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=no_of_classes),
        )
        self.extractor.fc = self.classifier

    def forward(self, input: Tensor) -> Tensor:
        return self.extractor(input)


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

class FCNN1(nn.Module):
    def __init__(self, non_linear_mapping_layers: int):
        super().__init__()
        self.feature_extraction_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=56,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(56),
            nn.PReLU())
        self.feature_extraction_2 = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=20,
                      kernel_size=5, padding=2),
            nn.Dropout2d(),
            nn.PReLU())
        self.non_linear_mapping = []
        for i in range(non_linear_mapping_layers):
            self.non_linear_mapping.append(
                nn.Conv2d(in_channels=20, out_channels=20,
                          kernel_size=3,
                          padding=1))
            if i % 2 == 0:
                self.non_linear_mapping.append(nn.BatchNorm2d(20))
            else:
                self.non_linear_mapping.append(nn.Dropout2d())
            self.non_linear_mapping.append(nn.PReLU())
        self.non_linear_mapping = nn.Sequential(*self.non_linear_mapping)
        self.adaptive_pooling = nn.AdaptiveMaxPool2d((5, 5))
        self.fc = nn.Sequential(
            nn.Linear(in_features=500, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=53),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extraction_1(x)
        x = self.feature_extraction_2(x)
        for mapping in self.non_linear_mapping:
            x = mapping(x)
        x = self.adaptive_pooling(x)
        x = x.reshape((x.shape[0], 500))
        x = self.fc(x)
        return x


class FCNN3(nn.Module):
    def __init__(self, non_linear_mapping_layers: int):
        super().__init__()
        self.feature_extraction_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=56,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(56),
            nn.PReLU())
        self.feature_extraction_2 = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=20,
                      kernel_size=5, padding=2),
            nn.PReLU())
        self.non_linear_mapping = []
        for i in range(non_linear_mapping_layers):
            self.non_linear_mapping.append(
                nn.Conv2d(in_channels=20, out_channels=20,
                          kernel_size=3,
                          padding=1))
            if i % 2 == 0:
                self.non_linear_mapping.append(nn.BatchNorm2d(20))
            self.non_linear_mapping.append(nn.PReLU())
        self.non_linear_mapping = nn.Sequential(*self.non_linear_mapping)
        self.adaptive_pooling = nn.AdaptiveMaxPool2d((5, 5))
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=500, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=500, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4),
        )

    def forward(self, x):
        x = self.feature_extraction_1(x)
        x = self.feature_extraction_2(x)
        for mapping in self.non_linear_mapping:
            x = mapping(x)
        x = self.adaptive_pooling(x)
        x = x.reshape((x.shape[0], 500))
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2