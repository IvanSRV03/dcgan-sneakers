# src/model.py — Arquitectura DCGAN para imágenes 128x128

import torch
import torch.nn as nn


def weights_init(m):
    """
    Inicialización de pesos según el paper original de DCGAN:
    Conv y ConvTranspose → Normal(0, 0.02)
    BatchNorm → Normal(1.0, 0.02), bias = 0
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR
# Entrada: vector z de ruido [B, latent_dim, 1, 1]
# Salida:  imagen RGB       [B, 3, 128, 128]
#
# Cada ConvTranspose2d duplica la resolución espacial:
#   4 → 8 → 16 → 32 → 64 → 128
# ─────────────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128, feature_maps: int = 64, num_channels: int = 3):
        super().__init__()
        nf = feature_maps  # alias corto

        self.net = nn.Sequential(
            # Bloque 0: z → 4x4
            self._block(latent_dim, nf * 16, kernel=4, stride=1, padding=0),  # 1→4

            # Bloque 1: 4→8
            self._block(nf * 16, nf * 8, kernel=4, stride=2, padding=1),

            # Bloque 2: 8→16
            self._block(nf * 8, nf * 4, kernel=4, stride=2, padding=1),

            # Bloque 3: 16→32
            self._block(nf * 4, nf * 2, kernel=4, stride=2, padding=1),

            # Bloque 4: 32→64
            self._block(nf * 2, nf, kernel=4, stride=2, padding=1),

            # Bloque final: 64→128  (sin BN, con Tanh)
            nn.ConvTranspose2d(nf, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # Salida en [-1, 1], igual que las imágenes normalizadas
        )
        self.apply(weights_init)

    @staticmethod
    def _block(in_ch, out_ch, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─────────────────────────────────────────────────────────────────────────────
# DISCRIMINATOR
# Entrada: imagen RGB [B, 3, 128, 128]
# Salida:  logit real/fake [B, 1]
#
# Cada Conv2d reduce la resolución a la mitad:
#   128 → 64 → 32 → 16 → 8 → 4 → 1
# ─────────────────────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, feature_maps: int = 64, num_channels: int = 3):
        super().__init__()
        nf = feature_maps

        self.net = nn.Sequential(
            # Bloque 0: 128→64  (sin BN en la primera capa, recomendado en DCGAN)
            nn.Conv2d(num_channels, nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Bloque 1: 64→32
            self._block(nf,     nf * 2,  kernel=4, stride=2, padding=1),

            # Bloque 2: 32→16
            self._block(nf * 2, nf * 4,  kernel=4, stride=2, padding=1),

            # Bloque 3: 16→8
            self._block(nf * 4, nf * 8,  kernel=4, stride=2, padding=1),

            # Bloque 4: 8→4
            self._block(nf * 8, nf * 16, kernel=4, stride=2, padding=1),

            # Clasificador final: 4→1
            nn.Conv2d(nf * 16, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # Sin Sigmoid aquí: usaremos BCEWithLogitsLoss (más estable numéricamente)
        )
        self.apply(weights_init)

    @staticmethod
    def _block(in_ch, out_ch, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img).view(-1)  # Aplanar a [B]
