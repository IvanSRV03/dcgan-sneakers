# src/dataset.py — Carga y preprocesamiento del dataset de tenis

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class SneakerDataset(Dataset):
    """
    Dataset de tenis para DCGAN.
    Espera una carpeta con imágenes .jpg/.jpeg/.png (sin subcarpetas necesarias).
    Las imágenes tienen fondo blanco, lo cual es ideal para normalización estándar.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root_dir: str, image_size: int = 128):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        # Recolecta todas las imágenes recursivamente
        self.image_paths = [
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        ]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No se encontraron imágenes en '{root_dir}'.\n"
                f"Extensiones soportadas: {self.EXTENSIONS}"
            )

        print(f"[Dataset] {len(self.image_paths)} imágenes encontradas en '{root_dir}'")

        # Transformaciones estándar para DCGAN
        # Normalize a [-1, 1] porque el Generator usa Tanh como activación final
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
            T.CenterCrop(image_size),
            T.ToTensor(),                          # [0, 255] → [0.0, 1.0]
            T.Normalize([0.5, 0.5, 0.5],           # → [-1.0, 1.0]
                        [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error al abrir imagen '{img_path}': {e}")
        return self.transform(img)


def get_dataloader(root_dir: str, image_size: int, batch_size: int,
                   num_workers: int = 4, shuffle: bool = True) -> DataLoader:
    """Crea y retorna un DataLoader listo para entrenamiento."""
    dataset = SneakerDataset(root_dir=root_dir, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,       # Acelera transferencia CPU→GPU
        drop_last=True,        # Evita batches incompletos al final
    )
    print(f"[DataLoader] {len(dataset)} imgs | batch_size={batch_size} | "
          f"{len(loader)} batches por epoch")
    return loader
