from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# =========================
# CONFIG
# =========================
DATA_ROOT = Path("data_bin")  # adjust if needed
OUT_DIR = Path("models/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUT_DIR / "pro_cnn_best.pt"
LAST_MODEL_PATH = OUT_DIR / "pro_cnn_last.pt"
HISTORY_CSV = OUT_DIR / "cnn_history.csv"

CONFIG = {
    "img_size": 224,
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 35,
    "alpha": 0.05,
    "patience": 6,
    "num_workers": 2,
    "threshold": 0.40,  # same as your validation rule
    "weight_decay": 1e-4,
    "seed": 42,
}


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# MODEL
# =========================
class ProCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(ProCNN, self).__init__()

        def conv_block(in_f, out_f, k):
            return nn.Sequential(
                nn.Conv2d(
                    in_f, out_f, kernel_size=k, stride=1, padding=k // 2, bias=False
                ),
                nn.BatchNorm2d(out_f),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32, k=5),
            conv_block(32, 64, k=5),
            conv_block(64, 128, k=3),
            conv_block(128, 256, k=3),
            conv_block(256, 512, k=3),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# =========================
# DATASET
# =========================
class XrayMaskedDataset(Dataset):
    def __init__(self, split_dir: Path, alpha=0.05, augment=False, img_size=224):
        self.alpha = alpha
        self.augment = augment
        self.img_size = img_size
        self.samples = []

        for cls, label in [("noncovid", 0), ("covid", 1)]:
            img_dir = split_dir / cls / "images"
            msk_dir = split_dir / cls / "masks"
            if img_dir.exists():
                for img_p in sorted(img_dir.glob("*.png")):
                    msk_p = msk_dir / img_p.name
                    if msk_p.exists():
                        self.samples.append((img_p, msk_p, label))

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, msk_p, y = self.samples[idx]

        img = Image.open(img_p).convert("L").resize((self.img_size, self.img_size))
        msk = (
            Image.open(msk_p)
            .convert("L")
            .resize((self.img_size, self.img_size), resample=Image.NEAREST)
        )

        if self.augment:
            if np.random.random() < 0.5:
                angle = np.random.uniform(-10, 10)
                img = img.rotate(angle)
                msk = msk.rotate(angle)

            if np.random.random() < 0.3:
                img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.8, 1.2))

            if np.random.random() < 0.3:
                img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.8, 1.2))

        x = np.array(img, dtype=np.float32)
        m = (np.array(msk) > 0).astype(np.float32)
        x_masked = x * (self.alpha + (1.0 - self.alpha) * m)

        rgb = Image.fromarray(np.clip(x_masked, 0, 255).astype(np.uint8)).convert("RGB")
        return self.transform(rgb), torch.tensor(y, dtype=torch.long)


# =========================
# METRICS
# =========================
def binary_metrics_from_counts(tp, fp, fn, tn):
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return recall, precision, f1, acc


# =========================
# CSV LOGGER
# =========================
def init_history_csv(path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "val_loss",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "tp",
                "fp",
                "fn",
                "tn",
            ]
        )


def append_history_row(path: Path, row: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["epoch"],
                row["lr"],
                row["train_loss"],
                row["val_loss"],
                row["accuracy"],
                row["precision"],
                row["recall"],
                row["f1"],
                row["tp"],
                row["fp"],
                row["fn"],
                row["tn"],
            ]
        )


# =========================
# TRAIN
# =========================
def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = XrayMaskedDataset(
        DATA_ROOT / "train",
        alpha=CONFIG["alpha"],
        augment=True,
        img_size=CONFIG["img_size"],
    )
    val_ds = XrayMaskedDataset(
        DATA_ROOT / "val",
        alpha=CONFIG["alpha"],
        augment=False,
        img_size=CONFIG["img_size"],
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            "Train/Val dataset is empty. Check DATA_ROOT path and folder structure."
        )

    # class balancing
    labels = [s[2] for s in train_ds.samples]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    model = ProCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=2
    )

    init_history_csv(HISTORY_CSV)

    best_f1 = 0.0
    early_stop_counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        # ---- TRAIN ----
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)

        # ---- VAL ----
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        tp = fp = fn = tn = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item()
                val_batches += 1

                probs_covid = torch.softmax(outputs, dim=1)[:, 1]
                preds = (probs_covid > CONFIG["threshold"]).long()

                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()

        val_loss = val_loss_sum / max(val_batches, 1)
        recall, precision, f1, acc = binary_metrics_from_counts(tp, fp, fn, tn)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | LR {current_lr:.6f} | "
            f"TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"Acc {acc:.3f} | Prec {precision:.3f} | Rec {recall:.3f} | F1 {f1:.3f}"
        )

        append_history_row(
            HISTORY_CSV,
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            },
        )

        # scheduler + early stopping on F1
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            early_stop_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"--> New best F1={best_f1:.4f}. Saved: {BEST_MODEL_PATH}")
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter}/{CONFIG['patience']}")

        if early_stop_counter >= CONFIG["patience"]:
            print(f"Stopping early at epoch {epoch}.")
            break

    torch.save(model.state_dict(), LAST_MODEL_PATH)
    print(f"Saved last model: {LAST_MODEL_PATH}")
    print(f"History CSV: {HISTORY_CSV}")


if __name__ == "__main__":
    main()
