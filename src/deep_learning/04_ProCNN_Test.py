import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# --- 1. KONFIGURATION ---
DATA_ROOT = Path("/kaggle/input/data-bin/data_bin")
MODEL_PATH = Path("/kaggle/working/models/pro_cnn_best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {"img_size": 224, "batch_size": 32, "alpha": 0.05, "default_threshold": 0.40}


# --- 2. ARCHITEKTUR ---
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


# --- 3. DATASET KLASSE ---
class XrayMaskedDataset(Dataset):
    def __init__(self, split_dir, alpha=0.05):
        self.alpha = alpha
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
                transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, msk_p, y = self.samples[idx]
        img = (
            Image.open(img_p)
            .convert("L")
            .resize((CONFIG["img_size"], CONFIG["img_size"]))
        )
        msk = (
            Image.open(msk_p)
            .convert("L")
            .resize((CONFIG["img_size"], CONFIG["img_size"]), resample=Image.NEAREST)
        )
        x, m = np.array(img, dtype=np.float32), (np.array(msk) > 0).astype(np.float32)
        x_masked = x * (self.alpha + (1.0 - self.alpha) * m)
        rgb = Image.fromarray(np.clip(x_masked, 0, 255).astype(np.uint8)).convert("RGB")
        return self.transform(rgb), torch.tensor(y, dtype=torch.long)


# --- 4. GRAD-CAM FUNKTIONEN ---
def generate_gradcam(model, img_tensor, target_class=None):
    model.eval()
    last_conv_layer = model.features[-1]
    gradients, activations = [], []

    def save_gradient(grad):
        gradients.append(grad)

    def save_activation(act):
        activations.append(act)

    h1 = last_conv_layer.register_forward_hook(lambda m, i, o: save_activation(o))
    h2 = last_conv_layer.register_full_backward_hook(
        lambda m, i, o: save_gradient(o[0])
    )

    output = model(img_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, target_class].backward()

    h1.remove()
    h2.remove()

    pooled_grads = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(activations[0].shape[1]):
        activations[0][:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(activations[0], dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    heatmap /= np.max(heatmap) + 1e-10
    return heatmap, target_class


def plot_gradcam(img_tensor, heatmap, title):
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.5 * heatmap_color + 0.5 * img

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original (Masked)")
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(title)
    plt.show()


# --- 5. HAUPT-AUSFÜHRUNG ---
def run_full_evaluation():
    print(f"Using device: {DEVICE}")
    test_ds = XrayMaskedDataset(DATA_ROOT / "test", alpha=CONFIG["alpha"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    model = ProCNN().to(DEVICE)
    if not MODEL_PATH.exists():
        print(f"❌ Modell unter {MODEL_PATH} nicht gefunden!")
        return

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    threshold = (
        float(ckpt.get("best_threshold", CONFIG["default_threshold"]))
        if isinstance(ckpt, dict)
        else CONFIG["default_threshold"]
    )
    model.eval()

    # --- Teil A: Metriken & CM ---
    y_true, y_probs = [], []
    for imgs, labels in test_loader:
        out = model(imgs.to(DEVICE))
        y_true.extend(labels.numpy())
        y_probs.extend(torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy())

    y_pred = (np.array(y_probs) > threshold).astype(int)
    print(
        "\n🚀 TEST RESULTS\n",
        classification_report(y_true, y_pred, target_names=["Normal", "COVID"]),
    )

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Normal", "COVID"],
        yticklabels=["Normal", "COVID"],
    )
    plt.title(f"Test Confusion Matrix (Thr: {threshold:.2f})")
    plt.show()

    # --- Teil B: Grad-CAM für ein COVID Bild ---
    print("\n🔎 Generiere Grad-CAM für ein COVID-Beispiel...")
    covid_indices = [i for i, l in enumerate(y_true) if l == 1]
    if covid_indices:
        idx = covid_indices[0]  # Erstes COVID Bild
        img, label = test_ds[idx]
        heatmap, pred_idx = generate_gradcam(model, img.unsqueeze(0).to(DEVICE))
        plot_gradcam(
            img,
            heatmap,
            f"Grad-CAM | Actual: COVID | Pred: {['Normal', 'COVID'][pred_idx]}",
        )
    else:
        print("Keine COVID Bilder im Testset gefunden.")


# Start
run_full_evaluation()
