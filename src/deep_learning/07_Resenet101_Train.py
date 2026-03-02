from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from pathlib import Path
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score, f1_score, recall_score

# =========================
# 1) Config
# =========================
DATA_ROOT = Path("/kaggle/input/data-bin/data_bin")
MODEL_SAVE_PATH = Path("/kaggle/working/models/resnet101_best.pt")

MODEL_NAME = "resnet101"
IMG_SIZE = 224
BATCH_SIZE = 8  # Lower batch size for ResNet-101 (deep model)
EPOCHS = 10
LR = 1e-4
SEED = 42
NUM_WORKERS = 4
SOFT_BG_ALPHA = 0.15


class XrayMaskedBinaryDataset(Dataset):
    def __init__(self, split_dir: Path, augment: bool = False):
        self.samples = []
        self.alpha = SOFT_BG_ALPHA
        self.augment = augment
        self.augmentor = v2.Compose(
            [
                v2.RandomRotation(degrees=15),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
        for cls in ["covid", "noncovid"]:
            label = 1 if cls == "covid" else 0
            img_dir, msk_dir = split_dir / cls / "images", split_dir / cls / "masks"
            for p in sorted(img_dir.glob("*.png")):
                if (msk_dir / p.name).exists():
                    self.samples.append(
                        {"img": p, "mask": msk_dir / p.name, "label": label}
                    )
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img, msk = Image.open(s["img"]).convert("L"), Image.open(s["mask"]).convert("L")
        if self.augment:
            state = torch.get_rng_state()
            img = self.augmentor(img)
            torch.set_rng_state(state)
            msk = v2.RandomRotation(degrees=15)(msk)
        img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        msk_np = (np.array(msk.resize((IMG_SIZE, IMG_SIZE))) / 255.0 > 0.5).astype(
            np.float32
        )
        masked = img_np * (self.alpha + (1.0 - self.alpha) * msk_np)
        return self.transform(
            Image.fromarray(np.clip(masked, 0, 255).astype(np.uint8)).convert("RGB")
        ), torch.tensor(s["label"])


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


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)
    train_ds = XrayMaskedBinaryDataset(DATA_ROOT / "train", augment=True)
    val_ds = XrayMaskedBinaryDataset(DATA_ROOT / "val", augment=False)

    lbls = [s["label"] for s in train_ds.samples]
    weights = 1.0 / np.bincount(lbls)
    sampler = WeightedRandomSampler(np.array([weights[l] for l in lbls]), len(lbls))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    best_f1 = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0  # Loss-Tracking

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Optimization Step
            opt.zero_grad()
            outputs = model(x)
            loss = crit(outputs, y)
            loss.backward()
            opt.step()

            train_loss += loss.item()

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
