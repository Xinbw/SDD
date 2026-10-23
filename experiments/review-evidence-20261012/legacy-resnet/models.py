from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_input_channels, get_num_classes


class SimpleCNN(nn.Module):
    """Two-layer CNN used for light clients (CIFAR-10)."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7 if in_channels == 1 else 64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TinyCNN(nn.Module):
    """Narrow two-layer CNN for light clients on 28×28 single-channel datasets (MNIST)."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """Configurable small ResNet backbone.

    base_channels controls the width: 64 for CIFAR-10 (32×32),
    32 for MNIST-like 28×28 datasets to avoid over-parameterisation.
    """
    def __init__(self, block, num_blocks: Tuple[int, int, int, int], in_channels: int, num_classes: int,
                 base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.in_planes = c
        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.layer1 = self._make_layer(block, c,     num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, c * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, c * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, c * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c * 8, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def build_model(arch: str, dataset_name: str) -> nn.Module:
    arch = arch.lower()
    in_channels = get_input_channels(dataset_name)
    num_classes = get_num_classes(dataset_name)
    # MNIST-like datasets use narrower models suited to 28×28 single-channel images
    mnist_like = dataset_name.lower() in ("mnist", "fashion-mnist")
    base_ch = 32 if mnist_like else 64
    if arch == "light":
        if mnist_like:
            return TinyCNN(in_channels, num_classes)
        return SimpleCNN(in_channels, num_classes)
    if arch == "standard":
        return ResNet(BasicBlock, (1, 1, 1, 1), in_channels, num_classes, base_channels=base_ch)
    if arch == "strong":
        return ResNet(BasicBlock, (2, 2, 2, 2), in_channels, num_classes, base_channels=base_ch)
    raise ValueError(f"Unknown architecture {arch}")