# src/model.py — Arquitectura DCGAN 128x128

import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_dim=128, feature_maps=64, num_channels=3):
        super().__init__()
        nf = feature_maps
        self.net = nn.Sequential(
            self._block(latent_dim, nf*16, 4, 1, 0),
            self._block(nf*16, nf*8, 4, 2, 1),
            self._block(nf*8,  nf*4, 4, 2, 1),
            self._block(nf*4,  nf*2, 4, 2, 1),
            self._block(nf*2,  nf,   4, 2, 1),
            nn.ConvTranspose2d(nf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(weights_init)

    @staticmethod
    def _block(in_ch, out_ch, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, feature_maps=64, num_channels=3):
        super().__init__()
        nf = feature_maps
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(nf,    nf*2,  4, 2, 1),
            self._block(nf*2,  nf*4,  4, 2, 1),
            self._block(nf*4,  nf*8,  4, 2, 1),
            self._block(nf*8,  nf*16, 4, 2, 1),
            nn.Conv2d(nf*16, 1, 4, 1, 0, bias=False),
        )
        self.apply(weights_init)

    @staticmethod
    def _block(in_ch, out_ch, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img):
        return self.net(img).view(-1)
