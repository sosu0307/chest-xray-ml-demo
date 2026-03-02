from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import v2
from sklearn.metrics import confusion_matrix

# =========================
# 1) Config
# =========================
DATA_ROOT = Path("data_bin")

OUT_DIR = Path("models/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUT_DIR / "resnet50_best.pt"
LAST_MODEL_PATH = OUT_DIR / "resnet50_last.pt"
HISTORY_CSV = OUT_DIR / "resnet_history.csv"

MODEL_NAME = "resnet50"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
SEED = 42
NUM_WORKERS = 2
SOFT_BG_ALPHA = 0.15

# decision threshold for class-1 (covid) during validation metrics
VAL_THRESHOLD = 0.40

# optional training stability
PATIENCE = 4  # early stopping on F1
SCHED_PATIENCE = 2  # ReduceLROnPlateau patience
SCHED_FACTOR = 0.2


# =========================
# 2) CSV logging utils
# =========================
def init_history_csv(path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
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
        w = csv.writer(f)
        w.writerow(
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
# 3) Dataset Class
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
        self.img_size = img_size
        self.samples = []

        # Augmentations for image only (mask handled separately for rotation sync)
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

            if not img_dir.exists() or not mask_dir.exists():
                continue

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
            # apply same random rotation to image and mask
            angle = random.uniform(-15, 15)
            img = v2.functional.rotate(img, angle)
            mask = v2.functional.rotate(mask, angle)

            # other augmentations only on image
            img = v2.ColorJitter(brightness=0.2, contrast=0.2)(img)
            if random.random() < 0.5:
                img = v2.RandomAdjustSharpness(sharpness_factor=2, p=1.0)(img)

        img_np = np.array(img.resize((self.img_size, self.img_size)), dtype=np.float32)
        mask_np = (
            np.array(mask.resize((self.img_size, self.img_size)), dtype=np.float32)
            / 255.0
        )
        mask_np = (mask_np > 0.5).astype(np.float32)

        # soft background masking
        masked = img_np * (self.alpha + (1.0 - self.alpha) * mask_np)
        masked_pil = Image.fromarray(np.clip(masked, 0, 255).astype(np.uint8)).convert(
            "RGB"
        )

        return self.img_transform(masked_pil), torch.tensor(
            s["label"], dtype=torch.long
        )


# =========================
# 4) Training/Eval utils
# =========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, criterion, threshold=0.40):
    model.eval()

    val_loss_sum = 0.0
    val_batches = 0

    y_true_all = []
    y_pred_all = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        val_loss_sum += loss.item()
        val_batches += 1

        probs = torch.softmax(logits, dim=1)[:, 1]  # covid prob
        y_pred = (probs >= threshold).long()

        y_true_all.extend(y.cpu().numpy().tolist())
        y_pred_all.extend(y_pred.cpu().numpy().tolist())

    val_loss = val_loss_sum / max(val_batches, 1)

    # confusion matrix with fixed label order [0,1]
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "val_loss": float(val_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# =========================
# 5) Main
# =========================
def main():
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Training: {MODEL_NAME} on {device}")

    # Datasets
    train_ds = XrayMaskedBinaryDataset(
        DATA_ROOT / "train",
        img_size=IMG_SIZE,
        alpha=SOFT_BG_ALPHA,
        augment=True,
    )
    val_ds = XrayMaskedBinaryDataset(
        DATA_ROOT / "val",
        img_size=IMG_SIZE,
        alpha=SOFT_BG_ALPHA,
        augment=False,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Dataset empty. Check DATA_ROOT and folder structure.")

    # Sampler for class balancing
    labels = [s["label"] for s in train_ds.samples]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = np.array([class_weights[l] for l in labels], dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHED_FACTOR, patience=SCHED_PATIENCE
    )

    init_history_csv(HISTORY_CSV)

    best_f1 = 0.0
    early_stop_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)

        # ---- Validate ----
        eval_out = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            threshold=VAL_THRESHOLD,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print("PWD:", Path.cwd())
        print("BEST_MODEL_PATH:", BEST_MODEL_PATH.resolve())
        print("LAST_MODEL_PATH:", LAST_MODEL_PATH.resolve())
        print("HISTORY_CSV:", HISTORY_CSV.resolve())

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"TrainLoss: {train_loss:.4f} | ValLoss: {eval_out['val_loss']:.4f} | "
            f"F1: {eval_out['f1']:.4f} | Acc: {eval_out['accuracy']:.4f} | "
            f"Prec: {eval_out['precision']:.4f} | Rec: {eval_out['recall']:.4f} | "
            f"TP/FP/FN/TN: {eval_out['tp']}/{eval_out['fp']}/{eval_out['fn']}/{eval_out['tn']}"
        )

        # CSV row
        append_history_row(
            HISTORY_CSV,
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": float(train_loss),
                "val_loss": eval_out["val_loss"],
                "accuracy": eval_out["accuracy"],
                "precision": eval_out["precision"],
                "recall": eval_out["recall"],
                "f1": eval_out["f1"],
                "tp": eval_out["tp"],
                "fp": eval_out["fp"],
                "fn": eval_out["fn"],
                "tn": eval_out["tn"],
            },
        )

        # scheduler on val f1
        scheduler.step(eval_out["f1"])

        # checkpoint by best F1
        if eval_out["f1"] > best_f1:
            best_f1 = eval_out["f1"]
            early_stop_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"--> New best F1: {best_f1:.4f} | saved: {BEST_MODEL_PATH}")
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter}/{PATIENCE}")

        if early_stop_counter >= PATIENCE:
            print(f"Stopping early at epoch {epoch}.")
            break

    # always save last
    torch.save(model.state_dict(), LAST_MODEL_PATH)

    print("\nTraining finished.")
    print(f"Best Val F1: {best_f1:.4f}")
    print(f"Best model:  {BEST_MODEL_PATH}")
    print(f"Last model:  {LAST_MODEL_PATH}")
    print(f"History CSV: {HISTORY_CSV}")


if __name__ == "__main__":
    main()
