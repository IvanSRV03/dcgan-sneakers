# src/dataset.py — Data Augmentation para DCGAN

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class SneakerDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root_dir, image_size=128, augment=True):
        self.root_dir = Path(root_dir)
        self.image_paths = [
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        ]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No se encontraron imágenes en '{root_dir}'.")
        print(f"[Dataset] {len(self.image_paths)} imágenes encontradas en '{root_dir}'")

        if augment:
            self.transform = T.Compose([
                T.Resize((int(image_size * 1.1), int(image_size * 1.1)),
                         interpolation=T.InterpolationMode.LANCZOS),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size),
                         interpolation=T.InterpolationMode.LANCZOS),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def get_dataloader(root_dir, image_size, batch_size,
                   num_workers=4, shuffle=True, augment=True):
    dataset = SneakerDataset(root_dir=root_dir, image_size=image_size, augment=augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    print(f"[DataLoader] {len(dataset)} imgs | batch_size={batch_size} | "
          f"{len(loader)} batches por epoch | augment={augment}")
    return loader
