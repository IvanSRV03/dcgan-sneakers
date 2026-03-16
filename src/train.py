# src/train.py — Loop principal de entrenamiento DCGAN

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from src.dataset import get_dataloader
from src.model import Generator, Discriminator
from src.utils import (
    set_seed, get_device,
    save_samples, save_checkpoint, load_checkpoint,
    plot_losses,
)

# ─── Etiquetas suavizadas (Label Smoothing) ───────────────────────────────────
# En lugar de 1.0 para "real", usamos 0.9. Esto hace al Discriminator menos
# confiado y estabiliza el entrenamiento.
REAL_LABEL = 0.9
FAKE_LABEL = 0.0


def train():
    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(config.SEED)
    device = get_device()
    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = get_dataloader(
        root_dir=config.DATA_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # ── Modelos ───────────────────────────────────────────────────────────────
    G = Generator(
        latent_dim=config.LATENT_DIM,
        feature_maps=config.G_FEATURES,
        num_channels=config.NUM_CHANNELS,
    ).to(device)

    D = Discriminator(
        feature_maps=config.D_FEATURES,
        num_channels=config.NUM_CHANNELS,
    ).to(device)

    print(f"\n[Generator]     Parámetros: {sum(p.numel() for p in G.parameters()):,}")
    print(f"[Discriminator] Parámetros: {sum(p.numel() for p in D.parameters()):,}\n")

    # ── Optimizadores ─────────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()

    g_optimizer = torch.optim.Adam(G.parameters(), lr=config.LEARNING_RATE,
                                    betas=(config.BETA1, config.BETA2))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=config.LEARNING_RATE,
                                    betas=(config.BETA1, config.BETA2))

    # ── Ruido fijo para muestras visuales comparables entre epochs ────────────
    fixed_noise = torch.randn(64, config.LATENT_DIM, 1, 1)

    # ── Opción de resumir desde checkpoint ────────────────────────────────────
    start_epoch = 0
    resume_path = os.environ.get("RESUME_CHECKPOINT", "")
    if resume_path and os.path.isfile(resume_path):
        start_epoch = load_checkpoint(resume_path, G, D, g_optimizer, d_optimizer, device)

    # ── Historial de losses ───────────────────────────────────────────────────
    g_losses, d_losses = [], []

    # ═════════════════════════════════════════════════════════════════════════
    # LOOP DE ENTRENAMIENTO
    # ═════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print(f"Entrenando por {config.NUM_EPOCHS} epochs en {device}")
    print("=" * 60)

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        G.train()
        D.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch [{epoch+1:>4}/{config.NUM_EPOCHS}]", leave=False)

        for real_imgs in pbar:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # ─── 1. Entrenar Discriminator ────────────────────────────────────
            # Objetivo: maximizar log(D(x)) + log(1 - D(G(z)))
            D.zero_grad()

            # 1a. Con imágenes reales
            real_labels = torch.full((batch_size,), REAL_LABEL, device=device)
            d_real_out = D(real_imgs)
            d_real_loss = criterion(d_real_out, real_labels)

            # 1b. Con imágenes falsas (Generator no actualiza aquí)
            z = torch.randn(batch_size, config.LATENT_DIM, 1, 1, device=device)
            fake_imgs = G(z)
            fake_labels = torch.full((batch_size,), FAKE_LABEL, device=device)
            d_fake_out = D(fake_imgs.detach())   # .detach() para no propagar al G
            d_fake_loss = criterion(d_fake_out, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # ─── 2. Entrenar Generator ────────────────────────────────────────
            # Objetivo: maximizar log(D(G(z))) → engañar al Discriminador
            G.zero_grad()

            # Las imágenes falsas ahora quieren ser clasificadas como "reales"
            real_labels_for_g = torch.full((batch_size,), 1.0, device=device)
            g_out = D(fake_imgs)   # Re-usa las mismas imágenes falsas (sin .detach())
            g_loss = criterion(g_out, real_labels_for_g)
            g_loss.backward()
            g_optimizer.step()

            # ─── Logging ─────────────────────────────────────────────────────
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            pbar.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}",
                "D(x)":   f"{torch.sigmoid(d_real_out).mean().item():.3f}",
                "D(G)":   f"{torch.sigmoid(d_fake_out).mean().item():.3f}",
            })

        # ── Fin de epoch ──────────────────────────────────────────────────────
        avg_g = epoch_g_loss / len(loader)
        avg_d = epoch_d_loss / len(loader)
        print(f"Epoch {epoch+1:>4}/{config.NUM_EPOCHS} | "
              f"G_loss: {avg_g:.4f} | D_loss: {avg_d:.4f}")

        # Guardar muestras visuales
        if (epoch + 1) % config.SAMPLE_INTERVAL == 0 or epoch == 0:
            path = save_samples(G, fixed_noise, epoch + 1, config.SAMPLES_DIR, device)
            print(f"  → Muestras guardadas: {path}")

        # Guardar checkpoint
        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(G, D, g_optimizer, d_optimizer,
                            epoch + 1, config.CHECKPOINTS_DIR)

    # ── Checkpoint final + gráfica de losses ─────────────────────────────────
    save_checkpoint(G, D, g_optimizer, d_optimizer,
                    config.NUM_EPOCHS, config.CHECKPOINTS_DIR)
    plot_losses(g_losses, d_losses,
                save_path=os.path.join(config.OUTPUT_DIR, "loss_curve.png"))
    print("\n✓ Entrenamiento completado.")


if __name__ == "__main__":
    train()
