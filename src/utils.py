# src/utils.py — Utilidades para guardado, visualización y logging

import os
import random
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def set_seed(seed: int):
    """Fija seeds para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detecta y retorna el mejor dispositivo disponible."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] GPU detectada: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA no disponible, usando CPU (será lento)")
    return device


def save_samples(generator, fixed_noise: torch.Tensor, epoch: int,
                 samples_dir: str, device: torch.device, n_images: int = 64):
    """
    Genera una grilla de imágenes de muestra y la guarda en disco.
    Las imágenes se desnormalizan de [-1,1] a [0,1] antes de guardar.
    """
    os.makedirs(samples_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise.to(device)).detach().cpu()

    # Desnormalizar: [-1, 1] → [0, 1]
    fake = (fake * 0.5) + 0.5

    grid = vutils.make_grid(fake[:n_images], nrow=8, padding=2, normalize=False)
    save_path = os.path.join(samples_dir, f"epoch_{epoch:04d}.png")
    vutils.save_image(grid, save_path)

    generator.train()
    return save_path


def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer,
                    epoch: int, checkpoints_dir: str):
    """Guarda el estado completo del entrenamiento para poder resumirlo."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    torch.save({
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "g_optimizer_state_dict": g_optimizer.state_dict(),
        "d_optimizer_state_dict": d_optimizer.state_dict(),
    }, path)
    print(f"[Checkpoint] Guardado en: {path}")


def load_checkpoint(path: str, generator, discriminator, g_optimizer, d_optimizer,
                    device: torch.device) -> int:
    """Carga un checkpoint y retorna la epoch desde donde continuar."""
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    g_optimizer.load_state_dict(ckpt["g_optimizer_state_dict"])
    d_optimizer.load_state_dict(ckpt["d_optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    print(f"[Checkpoint] Cargado desde epoch {ckpt['epoch']}. Continuando en epoch {start_epoch}")
    return start_epoch


def plot_losses(g_losses: list, d_losses: list, save_path: str = None):
    """Grafica las curvas de loss del Generator y Discriminator."""
    plt.figure(figsize=(12, 5))
    plt.plot(g_losses, label="Generator Loss", alpha=0.8)
    plt.plot(d_losses, label="Discriminator Loss", alpha=0.8)
    plt.xlabel("Iteraciones")
    plt.ylabel("Loss (BCE)")
    plt.title("DCGAN — Curvas de Entrenamiento")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Guardado en: {save_path}")
    plt.show()
