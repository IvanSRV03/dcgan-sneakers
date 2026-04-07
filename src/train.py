# src/train.py — DCGAN mejorado con LR Scheduler + Data Augmentation

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from src.dataset import get_dataloader
from src.model import Generator, Discriminator
from src.utils import set_seed, get_device, save_samples, save_checkpoint, plot_losses


def train():
    set_seed(config.SEED)
    device = get_device()
    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    loader = get_dataloader(
        root_dir=config.DATA_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        augment=True,
    )

    G = Generator(config.LATENT_DIM, config.G_FEATURES, config.NUM_CHANNELS).to(device)
    D = Discriminator(config.D_FEATURES, config.NUM_CHANNELS).to(device)

    print(f"\n[Generator]     Parámetros: {sum(p.numel() for p in G.parameters()):,}")
    print(f"[Discriminator] Parámetros: {sum(p.numel() for p in D.parameters()):,}\n")

    criterion = nn.BCEWithLogitsLoss()

    g_optimizer = torch.optim.Adam(G.parameters(), lr=config.LEARNING_RATE,
                                    betas=(config.BETA1, config.BETA2))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=config.LEARNING_RATE,
                                    betas=(config.BETA1, config.BETA2))

    # LR Scheduler — reduce el LR a la mitad en epoch 250 y 400
    # Ayuda a refinar detalles en la segunda mitad del entrenamiento
    g_scheduler = torch.optim.lr_scheduler.MultiStepLR(g_optimizer, milestones=[250, 400], gamma=0.5)
    d_scheduler = torch.optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=[250, 400], gamma=0.5)

    fixed_noise = torch.randn(64, config.LATENT_DIM, 1, 1)
    g_losses, d_losses = [], []

    print("=" * 60)
    print(f"Entrenando DCGAN por {config.NUM_EPOCHS} epochs en {device}")
    print(f"LR={config.LEARNING_RATE} | Scheduler en epochs 250 y 400")
    print("=" * 60)

    for epoch in range(config.NUM_EPOCHS):
        G.train(); D.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch [{epoch+1:>4}/{config.NUM_EPOCHS}]", leave=False)

        for real_imgs in pbar:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # ── Entrenar Discriminator ────────────────────────────────────────
            D.zero_grad()

            real_labels = torch.full((batch_size,), config.REAL_LABEL, device=device)
            d_real_out = D(real_imgs)
            d_real_loss = criterion(d_real_out, real_labels)

            z = torch.randn(batch_size, config.LATENT_DIM, 1, 1, device=device)
            fake_imgs = G(z)
            fake_labels = torch.full((batch_size,), config.FAKE_LABEL, device=device)
            d_fake_out = D(fake_imgs.detach())
            d_fake_loss = criterion(d_fake_out, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # ── Entrenar Generator ────────────────────────────────────────────
            G.zero_grad()
            real_labels_g = torch.full((batch_size,), 1.0, device=device)
            g_out = D(fake_imgs)
            g_loss = criterion(g_out, real_labels_g)
            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            pbar.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}",
                "D(x)":  f"{torch.sigmoid(d_real_out).mean().item():.3f}",
                "D(G)":  f"{torch.sigmoid(d_fake_out).mean().item():.3f}",
            })

        # Actualizar LR schedulers
        g_scheduler.step()
        d_scheduler.step()

        avg_g = epoch_g_loss / len(loader)
        avg_d = epoch_d_loss / len(loader)
        current_lr = g_optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:>4}/{config.NUM_EPOCHS} | "
              f"G_loss: {avg_g:.4f} | D_loss: {avg_d:.4f} | LR: {current_lr:.6f}")

        if (epoch + 1) % config.SAMPLE_INTERVAL == 0 or epoch == 0:
            path = save_samples(G, fixed_noise, epoch+1, config.SAMPLES_DIR, device)
            print(f"  → Muestras: {path}")

        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(G, D, g_optimizer, d_optimizer,
                            epoch+1, config.CHECKPOINTS_DIR)

    save_checkpoint(G, D, g_optimizer, d_optimizer,
                    config.NUM_EPOCHS, config.CHECKPOINTS_DIR)
    plot_losses(g_losses, d_losses,
                save_path=os.path.join(config.OUTPUT_DIR, "loss_curve.png"))
    print("\n✓ Entrenamiento completado.")


if __name__ == "__main__":
    train()
