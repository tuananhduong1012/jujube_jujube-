"""utils/dataset.py – Dataset building utilities."""

import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class JujubeDataset(Dataset):
    """Generic folder-per-class image dataset."""

    def __init__(self, samples, transform=None):
        self.samples   = samples      # list of (abs_path, int_label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_splits(
    dataset_root: str,
    class_names: list,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
):
    """
    Scan dataset_root for folder-per-class layout and return
    stratified (train_samples, val_samples, test_samples).
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    all_samples  = []

    for class_name in class_names:
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            print(f"  WARNING: directory not found → {class_dir}")
            continue
        files = [
            f for f in os.listdir(class_dir)
            if Path(f).suffix.lower() in VALID_EXTS
        ]
        for f in files:
            all_samples.append(
                (os.path.join(class_dir, f), class_to_idx[class_name])
            )
        print(f"  {class_name:<15}: {len(files)} images")

    print(f"\n  Total: {len(all_samples)} images across {len(class_names)} classes")

    # Stratified split
    random.seed(seed)
    class_samples = defaultdict(list)
    for path, label in all_samples:
        class_samples[label].append((path, label))

    train_s, val_s, test_s = [], [], []
    for label, samples in class_samples.items():
        random.shuffle(samples)
        n     = len(samples)
        n_tr  = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_s += samples[:n_tr]
        val_s   += samples[n_tr : n_tr + n_val]
        test_s  += samples[n_tr + n_val :]

    return train_s, val_s, test_s
