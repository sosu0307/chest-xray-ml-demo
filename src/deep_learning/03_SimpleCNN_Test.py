from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# --- CONFIG --
# Project Path - Only when run locally
DATA_ROOT = Path("/content/data_bin")

# Model Path
MODEL_DIR = Path("/content/drive/MyDrive/nov25_bds_int_covid1_code/models")
MODEL_PATH = MODEL_DIR / "simple_cnn_masked_best.pt"

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 2
ALPHA = 0.15


# --- MODEL DEFINITION ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 112 -> 56
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 56 -> 28
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4: 28 -> 14
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --- HELPER & DATASET ---
def list_png(folder: Path):
    return sorted(folder.glob("*.png"))


def mask_to_01(mask: Image.Image) -> np.ndarray:
    arr = np.array(mask.convert("L"), dtype=np.uint8)
    return (arr > 0).astype(np.float32)


class XrayMaskedBinaryDataset(Dataset):
    def __init__(self, split_dir: Path, img_size: int = 224, alpha: float = 0.15):
        self.alpha = alpha
        self.samples = []
        for cls, label in [("noncovid", 0), ("covid", 1)]:
            img_dir = split_dir / cls / "images"
            msk_dir = split_dir / cls / "masks"
            for img_path in list_png(img_dir):
                msk_path = msk_dir / img_path.name
                if msk_path.exists():
                    self.samples.append((img_path, msk_path, label))

        self.img_resize = transforms.Resize((img_size, img_size))
        self.msk_resize = transforms.Resize(
            (img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST
        )
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, msk_path, y = self.samples[idx]
        img = self.img_resize(Image.open(img_path).convert("L"))
        msk = self.msk_resize(Image.open(msk_path))
        x = np.array(img, dtype=np.float32)
        m = mask_to_01(msk)
        x = x * (self.alpha + (1.0 - self.alpha) * m)
        x = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8), mode="L").convert(
            "RGB"
        )
        return self.norm(self.to_tensor(x)), torch.tensor(y, dtype=torch.long)


# --- MAIN EVALUATION ---
@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = XrayMaskedBinaryDataset(
        DATA_ROOT / "test", img_size=IMG_SIZE, alpha=ALPHA
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Initialize model
    model = SimpleCNN(num_classes=2)

    # Load CKPT
    if MODEL_PATH.exists():
        print(f"Loading model: {MODEL_PATH}")
        ckpt = torch.load(MODEL_PATH, map_location=device)

        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        print("WARNING: No model checkpoint found, evaluating untrained model.")

    model.to(device).eval()

    y_true, y_pred, y_prob = [], [], []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred)
        y_prob.extend(prob)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== TEST RESULTS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nConfusion Matrix [ [TN FP]\n  [FN TP] ]:")
    print(cm)


if __name__ == "__main__":
    main()
