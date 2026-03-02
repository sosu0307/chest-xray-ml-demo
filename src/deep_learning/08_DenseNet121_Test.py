import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# =========================
# 1) Config
# =========================
DATA_ROOT = Path("/kaggle/input/data-bin/data_bin")
MODEL_PATH = Path("/kaggle/working/models/densenet121_best.pt")

MODEL_NAME = "densenet121"

IMG_SIZE = 224
BATCH_SIZE = 16
ALPHA = 0.15

# =========================
# 2) Dataset Class
# =========================


class XrayMaskedBinaryDataset(Dataset):
    def __init__(self, split_dir: Path, img_size: int = 224, alpha: float = 0.15):
        self.split_dir = split_dir
        self.alpha = alpha
        self.samples = []

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
# 3) Main Test Loop
# =========================


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Architecture: {MODEL_NAME.upper()} on {device}")

    test_ds = XrayMaskedBinaryDataset(
        DATA_ROOT / "test", img_size=IMG_SIZE, alpha=ALPHA
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # INITIALIZE DENSENET
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    if not MODEL_PATH.exists():
        print(f"❌ Weights not found at {MODEL_PATH}")
        return

    print(f"Loading weights from: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    y_true, y_pred, y_prob = [], [], []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]

        y_true.extend(y.numpy())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    print("\n" + "=" * 35)
    print(f"   TEST RESULTS: {MODEL_NAME.upper()}")
    print("=" * 35)
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_true, y_prob):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
