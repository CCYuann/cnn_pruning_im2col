"""CIFAR-native models used in the pruning literature (Li et al. 2017): VGG16 with BN and ResNet-56.

Both use only 3x3 and 1x1 convolutions with groups=1, so every layer fits the mask-aware accelerator.
ResNet-56 uses 1x1 convolution shortcuts (option B) instead of zero-padded identity shortcuts.
"""
import torch.nn as nn

VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class VGG(nn.Module):
    def __init__(self, num_classes, config=VGG16):
        super().__init__()
        layers, channels = [], 3
        for item in config:
            if item == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(channels, item, 3, padding=1, bias=False), nn.BatchNorm2d(item), nn.ReLU(inplace=True)]
                channels = item
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class Block(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inputs != outputs:
            self.downsample = nn.Sequential(nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.BatchNorm2d(outputs))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out+(x if self.downsample is None else self.downsample(x)))


class CifarResNet(nn.Module):
    def __init__(self, num_classes, blocks=9, widths=(16, 32, 64)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.relu = nn.ReLU(inplace=True)
        stages, channels = [], widths[0]
        for i, width in enumerate(widths):
            layers = []
            for j in range(blocks):
                layers.append(Block(channels, width, 2 if i and not j else 1))
                channels = width
            stages.append(nn.Sequential(*layers))
        self.layer1, self.layer2, self.layer3 = stages
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(self.pool(x).flatten(1))
