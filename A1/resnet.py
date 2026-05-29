import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Basic ResNet block: two 3x3 convolutions with a skip connection.
    If in_channels != out_channels, the shortcut uses a 1x1 conv to match dimensions.

    Attributes
    ----------
    conv1, conv2 : Conv2d
        Two 3x3 convolutional layers
    bn1, bn2 : BatchNorm2d
        Batch normalisation after each conv
    relu : ReLU
        Activation function
    shortcut : Sequential or Identity
        Skip connection — 1x1 conv if dimensions change, identity otherwise
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.relu     = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images).

    Uses a 3x3 stem convolution instead of the original 7x7 to preserve
    spatial resolution on the smaller CIFAR images.

    Attributes
    ----------
    stem : Sequential
        Initial 3x3 conv + BN + ReLU
    layer1–layer4 : Sequential
        Four stages of residual blocks (2 blocks each)
    pool : AdaptiveAvgPool2d
        Global average pooling to 1x1
    fc : Linear
        Final classifier head
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, n_classes)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(n_blocks - 1):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)