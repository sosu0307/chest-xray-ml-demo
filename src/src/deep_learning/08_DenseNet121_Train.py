from __future__ import annotations
import torch
import os
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import v2
from torchsummary import summary
from sklearn.metrics import accuracy_score, f1_score, recall_score

# =========================
# 1) Config
# =========================
DATA_ROOT = Path("/kaggle/input/data-bin/data_bin")
MODEL_SAVE_PATH = Path("/kaggle/working/models/densenet121_best.pt")

MODEL_NAME = "densenet121"
IMG_SIZE = 224
BATCH_SIZE = 16  # DenseNet is deeper but memory efficient
EPOCHS = 10
LR = 1e-4  # Stable learning rate for transfer learning
SEED = 42
NUM_WORKERS = 4
SOFT_BG_ALPHA = 0.15

# =========================
# 2) Dataset Class
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
        self.samples = []

        # Modern Augmentations (v2)
        self.augmentor = v2.Compose(
            [
                v2.RandomRotation(degrees=15),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
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

        if not self.samples:
            raise RuntimeError(f"No samples found in {split_dir}")

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
            state = torch.get_rng_state()
            img = self.augmentor(img)
            torch.set_rng_state(state)
            mask = v2.RandomRotation(degrees=15)(mask)

        img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        mask_np = (np.array(mask.resize((IMG_SIZE, IMG_SIZE))) / 255.0 > 0.5).astype(
            np.float32
        )

        masked = img_np * (self.alpha + (1.0 - self.alpha) * mask_np)
        masked_pil = Image.fromarray(np.clip(masked, 0, 255).astype(np.uint8)).convert(
            "RGB"
        )

        return self.img_transform(masked_pil), torch.tensor(
            s["label"], dtype=torch.long
        )


# =========================
# 3) Training Utils
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
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_pred.extend(model(x).argmax(1).cpu().numpy())
        y_true.extend(y.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    return acc, f1, rec


# =========================
# 4) Main
# =========================


def main():
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting DenseNet-121 Training on {device}")
    os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)

    train_ds = XrayMaskedBinaryDataset(DATA_ROOT / "train", augment=True)
    val_ds = XrayMaskedBinaryDataset(DATA_ROOT / "val", augment=False)

    labels = [s["label"] for s in train_ds.samples]
    weights = 1.0 / np.bincount(labels)
    sampler = WeightedRandomSampler(np.array([weights[l] for l in labels]), len(labels))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # LOAD DENSENET
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    # Target the .classifier for DenseNet
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        acc, f1 = evaluate(model, val_loader, device)
        val_acc, val_f1, val_rec = evaluate(model, val_loader, device)

    print(
        f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | "
        f"Val F1: {val_f1:.4f} | Acc: {val_acc:.4f} | Rec: {val_rec:.4f}"
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
