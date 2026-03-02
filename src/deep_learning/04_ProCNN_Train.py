import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
from pathlib import Path
from torchsummary import summary

# --- 1. CONFIGURATION ---
DATA_ROOT = Path("/kaggle/input/data-bin/data_bin")
MODEL_SAVE_PATH = Path("/kaggle/working/models/pro_cnn_best.pt")
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "img_size": 224,
    "batch_size": 32,
    "lr": 0.0001,
    "epochs": 35,  # Mehr Zeit für das größere Modell
    "alpha": 0.05,  # Scharfer Fokus auf das Lungeninnere
    "patience": 6,  # Etwas mehr Geduld beim Early Stopping
}


# --- 2. ARCHITECTURE (Power-Up: Verdoppelte Filter) ---
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

        # Architektur mit 32 -> 64 -> 128 -> 256 -> 512 Filtern
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


# --- 3. DATASET CLASS (Rotation, Brightness, Contrast) ---
class XrayMaskedDataset(Dataset):
    def __init__(self, split_dir, alpha=0.05, augment=False):
        self.alpha = alpha
        self.augment = augment
        self.samples = []
        for cls, label in [("noncovid", 0), ("covid", 1)]:
            img_dir, msk_dir = split_dir / cls / "images", split_dir / cls / "masks"
            if img_dir.exists():
                for img_p in sorted(img_dir.glob("*.png")):
                    msk_p = msk_dir / img_p.name
                    if msk_p.exists():
                        self.samples.append((img_p, msk_p, label))

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, msk_p, y = self.samples[idx]
        img = Image.open(img_p).convert("L").resize((224, 224))
        msk = Image.open(msk_p).convert("L").resize((224, 224), resample=Image.NEAREST)

        if self.augment:
            # Rotation
            if np.random.random() < 0.5:
                angle = np.random.uniform(-10, 10)
                img = img.rotate(angle)
                msk = msk.rotate(angle)

            # Helligkeit
            if np.random.random() < 0.3:
                img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.8, 1.2))

            # Kontrast
            if np.random.random() < 0.3:
                img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.8, 1.2))

        # Maskierung anwenden
        x = np.array(img, dtype=np.float32)
        m = (np.array(msk) > 0).astype(np.float32)
        x_masked = x * (self.alpha + (1.0 - self.alpha) * m)

        rgb = Image.fromarray(np.clip(x_masked, 0, 255).astype(np.uint8)).convert("RGB")
        return self.transform(rgb), torch.tensor(y, dtype=torch.long)


# --- 4. TRAINING & EVALUATION LOOP ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Datasets
    train_ds = XrayMaskedDataset(
        DATA_ROOT / "train", alpha=CONFIG["alpha"], augment=True
    )
    val_ds = XrayMaskedDataset(DATA_ROOT / "val", alpha=CONFIG["alpha"])

    # Sampler für Balancing
    labels = [s[2] for s in train_ds.samples]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], sampler=sampler
    )
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])

    model = ProCNN().to(device)
    summary(model, input_size=(3, 224, 224))

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=2
    )

    best_f1 = 0.0
    early_stop_counter = 0

    for epoch in range(CONFIG["epochs"]):
        # TRAINING
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # VALIDATION
        model.eval()
        tp, fp, fn, tn = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = (torch.softmax(outputs, dim=1)[:, 1] > 0.4).long()

                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()

        # Metrics
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        acc = (tp + tn) / (tp + tn + fp + fn)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:02d} | LR: {current_lr:.6f} | Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.2f} | Rec: {recall:.2f} | F1: {f1:.2f}"
        )

        # Scheduler & Early Stopping Logic
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> New Best F1: {f1:.4f} - Modell gespeichert!")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"EarlyStopping: {early_stop_counter}/{CONFIG['patience']}")

        if early_stop_counter >= CONFIG["patience"]:
            print(f"🛑 Abbruch nach {epoch+1} Epochen.")
            break


if __name__ == "__main__":
    main()
