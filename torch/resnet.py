import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_dim, dim, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_dim != self.expansion * dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, self.expansion * dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * dim),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_dim, dim, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, self.expansion * dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * dim)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_dim != self.expansion * dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, self.expansion * dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * dim),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, n_blocks, n_class):
        super(ResNet, self).__init__()
        self.in_dim = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, n_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, n_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, n_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, n_blocks[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, n_class)

    def _make_layer(self, block, dim, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_dim, dim, stride))
            self.in_dim = dim * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(n_class):
    return ResNet(BasicBlock, [2, 2, 2, 2], n_class)


def ResNet34(n_class):
    return ResNet(BasicBlock, [3, 4, 6, 3], n_class)


def ResNet50(n_class):
    return ResNet(Bottleneck, [3, 4, 6, 3], n_class)


def ResNet101(n_class):
    return ResNet(Bottleneck, [3, 4, 23, 3], n_class)


def ResNet152(n_class):
    return ResNet(Bottleneck, [3, 8, 36, 3], n_class)
