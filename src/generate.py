# src/generate.py — Generación e interpolación de tenis nuevos

"""
Uso:
    # Generar 64 imágenes aleatorias desde un checkpoint:
    python src/generate.py --checkpoint outputs/checkpoints/checkpoint_epoch_0300.pt

    # Interpolación esférica entre dos vectores z (combina características):
    python src/generate.py --checkpoint ... --interpolate --steps 10
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torchvision.utils as vutils

import config
from src.model import Generator
from src.utils import get_device


def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    """
    Interpolación esférica (SLERP) entre dos vectores latentes.
    Es mejor que la interpolación lineal para espacios latentes de GANs,
    ya que mantiene la norma del vector y evita zonas de baja densidad.
    """
    z1_norm = z1 / z1.norm(dim=1, keepdim=True)
    z2_norm = z2 / z2.norm(dim=1, keepdim=True)
    omega = torch.acos((z1_norm * z2_norm).sum(dim=1, keepdim=True).clamp(-1, 1))
    sin_omega = torch.sin(omega)
    # Evitar división por cero cuando los vectores son casi paralelos
    if sin_omega.abs().item() < 1e-6:
        return (1 - t) * z1 + t * z2
    return (torch.sin((1 - t) * omega) / sin_omega) * z1 + \
           (torch.sin(t * omega) / sin_omega) * z2


def load_generator(checkpoint_path: str, device: torch.device) -> Generator:
    G = Generator(
        latent_dim=config.LATENT_DIM,
        feature_maps=config.G_FEATURES,
        num_channels=config.NUM_CHANNELS,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt["generator_state_dict"])
    G.eval()
    print(f"[Generator] Cargado desde: {checkpoint_path}")
    return G


def generate_random(G: Generator, n: int, device: torch.device, out_path: str):
    """Genera N imágenes aleatorias y las guarda como grilla."""
    z = torch.randn(n, config.LATENT_DIM, 1, 1, device=device)
    with torch.no_grad():
        imgs = G(z).cpu()
    imgs = (imgs * 0.5) + 0.5  # Desnormalizar
    vutils.save_image(imgs, out_path, nrow=8, padding=2)
    print(f"[Generate] {n} imágenes guardadas en: {out_path}")


def generate_interpolation(G: Generator, steps: int, device: torch.device, out_path: str):
    """
    Genera una secuencia de interpolación entre dos tenis.
    El resultado muestra cómo el espacio latente 'mezcla' características visuales.
    """
    z1 = torch.randn(1, config.LATENT_DIM, 1, 1, device=device)
    z2 = torch.randn(1, config.LATENT_DIM, 1, 1, device=device)

    frames = []
    for i, t in enumerate([i / (steps - 1) for i in range(steps)]):
        z = slerp(z1, z2, t)
        with torch.no_grad():
            img = G(z).cpu()
        img = (img * 0.5) + 0.5
        frames.append(img)

    grid = torch.cat(frames, dim=0)
    vutils.save_image(grid, out_path, nrow=steps, padding=2)
    print(f"[Interpolate] Secuencia de {steps} pasos guardada en: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generación con DCGAN entrenada")
    parser.add_argument("--checkpoint", required=True,
                        help="Ruta al checkpoint .pt del entrenamiento")
    parser.add_argument("--interpolate", action="store_true",
                        help="Generar interpolación entre dos puntos del espacio latente")
    parser.add_argument("--steps", type=int, default=10,
                        help="Número de pasos en la interpolación (default: 10)")
    parser.add_argument("--n", type=int, default=64,
                        help="Número de imágenes aleatorias a generar (default: 64)")
    parser.add_argument("--out", type=str, default="outputs",
                        help="Carpeta de salida (default: outputs/)")
    args = parser.parse_args()

    device = get_device()
    G = load_generator(args.checkpoint, device)
    os.makedirs(args.out, exist_ok=True)

    if args.interpolate:
        out_path = os.path.join(args.out, "interpolacion.png")
        generate_interpolation(G, steps=args.steps, device=device, out_path=out_path)
    else:
        out_path = os.path.join(args.out, "generadas.png")
        generate_random(G, n=args.n, device=device, out_path=out_path)


if __name__ == "__main__":
    main()
