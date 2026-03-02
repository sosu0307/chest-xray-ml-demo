import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

# --- 1. Path ---
DATA_ROOT = Path("/content/data_bin")
MODEL_DIR = Path("/content/drive/MyDrive/nov25_bds_int_covid1_code/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "simple_cnn_masked_best.pt"

IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15
ALPHA = 0.15
NUM_WORKERS = 2


# --- 2. Model ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --- 3. Dataset with  Data Augmentation ---
def list_png(folder: Path):
    return sorted(folder.glob("*.png"))


def mask_to_01(mask: Image.Image) -> np.ndarray:
    arr = np.array(mask.convert("L"), dtype=np.uint8)
    return (arr > 0).astype(np.float32)


class XrayMaskedBinaryDataset(torch.utils.data.Dataset):
    def __init__(
        self, split_dir: Path, img_size=224, alpha=0.15, augment: bool = False
    ):
        self.alpha = alpha
        self.augment = augment
        self.samples = []
        for cls, label in [("noncovid", 0), ("covid", 1)]:
            img_dir = split_dir / cls / "images"
            msk_dir = split_dir / cls / "masks"
            for img_path in list_png(img_dir):
                msk_path = msk_dir / img_path.name
                if msk_path.exists():
                    self.samples.append((img_path, msk_path, label))

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.msk_resize = transforms.Resize(
            (img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, msk_path, y = self.samples[idx]
        img = Image.open(img_path).convert("L").resize((224, 224))
        msk = self.msk_resize(Image.open(msk_path))

        # --- DATA AUGMENTATION BLOCK ---
        if self.augment:
            angle = np.random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BILINEAR)
            msk = msk.rotate(angle, resample=Image.NEAREST)

            brightness_factor = np.random.uniform(0.8, 1.2)
            img = transforms.functional.adjust_brightness(img, brightness_factor)

        x = np.array(img, dtype=np.float32)
        m = mask_to_01(msk)
        x = x * (self.alpha + (1.0 - self.alpha) * m)
        x_rgb = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8), mode="L").convert(
            "RGB"
        )
        return self.transform(x_rgb), torch.tensor(y, dtype=torch.long)


# --- 4. Main Training Loop ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training auf: {device} (mit Augmentation)")

    train_ds = XrayMaskedBinaryDataset(DATA_ROOT / "train", alpha=ALPHA, augment=True)
    val_ds = XrayMaskedBinaryDataset(DATA_ROOT / "val", alpha=ALPHA, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict()}, MODEL_PATH)
            print(f"--> Modell gespeichert ({val_acc:.2f}%)")


if __name__ == "__main__":
    main()
