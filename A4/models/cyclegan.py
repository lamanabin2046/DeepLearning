"""
CycleGAN architecture for 64x64 CelebA hair-color translation.
Matches the A4 notebook (Part 2): ResNet generator + PatchGAN discriminator.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGenerator(nn.Module):
    """ResNet generator for CycleGAN (64x64 images, 6 residual blocks)."""
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_res=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            # Downsample x2
            nn.Conv2d(ngf,   ngf*2, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
        ]
        for _ in range(n_res):
            layers.append(ResidualBlock(ngf * 4))
        layers += [
            # Upsample x2
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7), nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator: judges 70x70 patches as real/fake."""
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()

        def block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers += block(in_ch, ndf, stride=2, norm=False)
        layers += block(ndf, ndf * 2, stride=2)
        layers += block(ndf * 2, ndf * 4, stride=2)
        layers += block(ndf * 4, ndf * 8, stride=1)
        layers += [nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)