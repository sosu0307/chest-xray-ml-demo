from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
from torchsummary import summary

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score, f1_score, recall_score

# =========================
# 1) Config
# =========================

DATA_ROOT = Path("/kaggle/input/data-bin/data_bin")
MODEL_SAVE_PATH = Path("/kaggle/working/models/resnet18_best.pt")
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4  # Lowered for stable Transfer Learning
SEED = 42
NUM_WORKERS = 4
SOFT_BG_ALPHA = 0.15

# =========================
# 2) Dataset
# =========================


class XrayMaskedBinaryDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        img_size: int = 224,
        alpha: float = 0.15,
        augment: bool = False,
    ):
        self.split_dir = split_dir
        self.alpha = alpha
        self.augment = augment
        self.samples: List[dict] = []

        # Define augmentations using v2
        self.augmentor = v2.Compose(
            [
                v2.RandomRotation(degrees=15),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            ]
        )

        for cls in ["covid", "noncovid"]:
            label = 1 if cls == "covid" else 0
            img_dir = split_dir / cls / "images"
            mask_dir = split_dir / cls / "masks"

            for img_path in sorted(img_dir.glob("*.png")):
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    self.samples.append(
                        {"img": img_path, "mask": mask_path, "label": label}
                    )

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["img"]).convert("L")
        mask = Image.open(s["mask"]).convert("L")

        if self.augment:
            # Apply same random transform to both image and mask for rotation
            state = torch.get_rng_state()
            img = self.augmentor(img)
            torch.set_rng_state(state)
            # We don't apply color jitter to the mask
            mask = v2.RandomRotation(degrees=15)(mask)

        # Mask processing
        img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        mask_np = np.array(mask.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)

        # Apply soft masking
        masked = img_np * (self.alpha + (1.0 - self.alpha) * mask_np)
        masked = Image.fromarray(np.clip(masked, 0, 255).astype(np.uint8)).convert(
            "RGB"
        )

        return self.img_transform(masked), torch.tensor(s["label"], dtype=torch.long)


# =========================
# 4) Training Logic
# =========================


def seed_everything(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        all_y.extend(y.cpu().numpy())
        all_pred.extend(pred.cpu().numpy())
    return (
        accuracy_score(all_y, all_pred),
        f1_score(all_y, all_pred),
        recall_score(all_y, all_pred),
    )


def main():
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = XrayMaskedBinaryDataset(DATA_ROOT / "train", augment=True)
    val_ds = XrayMaskedBinaryDataset(DATA_ROOT / "val", augment=False)

    # Sampler for imbalance
    labels = [s["label"] for s in train_ds.samples]
    class_sample_count = np.array(
        [len(np.where(labels == t)[0]) for t in np.unique(labels)]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc, val_f1, val_rec = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f} | Rec: {val_rec:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
