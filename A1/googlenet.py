import torch
import torch.nn as nn


class Inception(nn.Module):
    """
    Inception block for GoogLeNet.

    Attributes
    ----------
    in_planes : int
        Number of input feature maps
    n1x1 : int
        Number of direct 1x1 convolutions
    n3x3red : int
        Number of 1x1 reductions before 3x3 convolutions
    n3x3 : int
        Number of 3x3 convolutions
    n5x5red : int
        Number of 1x1 reductions before 5x5 convolutions
    n5x5 : int
        Number of 5x5 convolutions (implemented as two 3x3)
    pool_planes : int
        Number of 1x1 convolutions after 3x3 max pooling
    b1 : Sequential
        Branch 1 — direct 1x1 convolutions
    b2 : Sequential
        Branch 2 — reduction then 3x3 convolutions
    b3 : Sequential
        Branch 3 — reduction then two 3x3 convolutions (approximates 5x5)
    b4 : Sequential
        Branch 4 — max pooling then 1x1 reduction
    """

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super().__init__()
        self.in_planes   = in_planes
        self.n1x1        = n1x1
        self.n3x3red     = n3x3red
        self.n3x3        = n3x3
        self.n5x5red     = n5x5red
        self.n5x5        = n5x5
        self.pool_planes = pool_planes

        # Branch 1: 1x1 conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True),
        )

        # Branch 2: 1x1 conv -> 3x3 conv
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True),
        )

        # Branch 3: 1x1 conv -> 3x3 -> 3x3 (approximates 5x5)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
        )

        # Branch 4: 3x3 max pool -> 1x1 conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) for CIFAR-10 with 32x32 inputs.

    Uses CIFAR-10 sized inputs (32x32) following the kuangliu implementation
    from the professor's notebook, which is memory efficient and well suited
    for CIFAR-10.

    Attributes
    ----------
    pre_layers : Sequential
        Initial convolutional stem
    a3, b3 : Inception
        First and second inception blocks
    maxpool : MaxPool2d
        Pooling layer after b3 and after e4
    a4, b4, c4, d4, e4 : Inception
        Inception blocks in the fourth stage
    a5, b5 : Inception
        Inception blocks in the fifth stage
    avgpool : AdaptiveAvgPool2d
        Global average pooling before classifier
    linear : Linear
        Final fully connected layer
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear  = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class AuxClassifier(nn.Module):
    """
    Auxiliary classifier used in GoogLeNet for Q3.
    """

    def __init__(self, in_channels, num_classes=10):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class GoogLeNetAux(nn.Module):
    """
    Q3: ImageNet-style GoogLeNet for 224x224 inputs with two auxiliary classifiers.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Correct GoogLeNet backbone before first Inception module
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        # Two side classifiers
        self.aux1 = AuxClassifier(512, num_classes)
        self.aux2 = AuxClassifier(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)

        x = self.a4(x)

        aux1 = None
        if self.training:
            aux1 = self.aux1(x)

        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)

        aux2 = None
        if self.training:
            aux2 = self.aux2(x)

        x = self.e4(x)
        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)

        if self.training:
            return x, aux1, aux2

        return x
