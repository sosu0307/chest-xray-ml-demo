import random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =========================
# 1) Config
# =========================
DATA_ROOT = Path("/kaggle/input/data-bin/data_bin")
MODEL_PATH = Path("/kaggle/working/models/resnet50_best.pt")
OUT_DIR = Path("/kaggle/working/report_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
ALPHA = 0.15
SEED = 42


# =========================
# 2) Dataset
# =========================
class XrayMaskedBinaryDataset(Dataset):
    def __init__(self, split_dir: Path, img_size: int = 224, alpha: float = 0.15):
        self.samples = []
        self.img_size = img_size
        self.alpha = alpha

        for cls in ["covid", "noncovid"]:
            label = 1 if cls == "covid" else 0
            img_dir = split_dir / cls / "images"
            mask_dir = split_dir / cls / "masks"

            for img_path in sorted(img_dir.glob("*.png")):
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    self.samples.append(
                        {
                            "img": img_path,
                            "mask": mask_path,
                            "label": label,
                            "class_name": cls,
                        }
                    )

        if not self.samples:
            raise RuntimeError(f"No samples found in {split_dir}")

        self.tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["img"]).convert("L")
        mask = Image.open(s["mask"]).convert("L")

        img_np = np.array(img.resize((self.img_size, self.img_size)), dtype=np.float32)
        mask_np = (
            np.array(mask.resize((self.img_size, self.img_size)), dtype=np.float32)
            / 255.0
        )
        mask_np = (mask_np > 0.5).astype(np.float32)

        masked = img_np * (self.alpha + (1.0 - self.alpha) * mask_np)
        masked_rgb = Image.fromarray(np.clip(masked, 0, 255).astype(np.uint8)).convert(
            "RGB"
        )

        x = self.tf(masked_rgb)
        y = torch.tensor(s["label"], dtype=torch.long)

        meta = {
            "filename": s["img"].name,
            "class_name": s["class_name"],
            "masked_rgb_uint8": np.array(masked_rgb),
        }
        return x, y, meta


# =========================
# 3) Grad-CAM
# =========================
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.h1 = target_layer.register_forward_hook(self._fhook)
        self.h2 = target_layer.register_full_backward_hook(self._bhook)

    def _fhook(self, module, inp, out):
        self.activations = out.detach()

    def _bhook(self, module, gin, gout):
        self.gradients = gout[0].detach()

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def compute(self, x, class_idx):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx]
        score.backward()

        acts = self.activations[0]  # [C,H,W]
        grads = self.gradients[0]  # [C,H,W]
        w = grads.mean(dim=(1, 2))  # [C]

        cam = torch.zeros(acts.shape[1:], device=acts.device)
        for c, wc in enumerate(w):
            cam += wc * acts[c]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def overlay_cam(rgb_uint8, cam_2d, alpha=0.4):
    h, w, _ = rgb_uint8.shape
    cam_img = Image.fromarray((cam_2d * 255).astype(np.uint8)).resize(
        (w, h), Image.BILINEAR
    )
    cam_norm = np.array(cam_img) / 255.0
    heatmap = plt.get_cmap("jet")(cam_norm)[..., :3]
    base = rgb_uint8.astype(np.float32) / 255.0
    ov = (1 - alpha) * base + alpha * heatmap
    ov = np.clip(ov, 0, 1)
    return (ov * 255).astype(np.uint8)


# =========================
# 4) Main
# =========================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Data
    test_ds = XrayMaskedBinaryDataset(
        DATA_ROOT / "test", img_size=IMG_SIZE, alpha=ALPHA
    )

    def collate_fn(batch):
        xs, ys, metas = zip(*batch)
        return torch.stack(xs), torch.stack(ys), list(metas)

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    # Inference
    y_true, y_pred, y_prob = [], [], []
    all_metas = []
    all_x_cpu = []

    with torch.no_grad():
        for x, y, metas in test_loader:
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            y_true.extend(y.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            all_metas.extend(metas)
            all_x_cpu.extend([xi.clone() for xi in x])  # store normalized tensor on CPU

    # -------------------------
    # Figure 1: Confusion Matrix (Blue)
    # -------------------------
    cm = confusion_matrix(y_true, y_pred)

    blue_cmap = LinearSegmentedColormap.from_list(
        "custom_blue", ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap=blue_cmap)
    plt.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix - ResNet50 (Test)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-COVID", "COVID"])
    ax.set_yticklabels(["Non-COVID", "COVID"])

    thr = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i,j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thr else "black",
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_main_cm_resnet50.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # Figure 2: ROC Curve
    # -------------------------
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ResNet50 (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve - ResNet50 (Test)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_main_roc_topline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # Figure 3: One Grad-CAM Example
    # Priority pick: True Positive COVID
    # -------------------------
    idx_tp = [
        i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt == 1 and yp == 1
    ]
    idx_fp = [
        i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt == 0 and yp == 1
    ]
    idx_fn = [
        i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt == 1 and yp == 0
    ]
    idx_tn = [
        i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt == 0 and yp == 0
    ]

    if idx_tp:
        pick = idx_tp[0]
        case_name = "TP"
    elif idx_fp:
        pick = idx_fp[0]
        case_name = "FP"
    elif idx_fn:
        pick = idx_fn[0]
        case_name = "FN"
    else:
        pick = idx_tn[0]
        case_name = "TN"

    gradcam = GradCAM(model, model.layer4[-1].conv3)

    x_single = all_x_cpu[pick].unsqueeze(0).to(device)
    true_label = y_true[pick]
    pred_label = y_pred[pick]
    pred_prob = y_prob[pick]
    meta = all_metas[pick]

    cam = gradcam.compute(x_single, class_idx=pred_label)
    gradcam.remove()

    base_rgb = meta["masked_rgb_uint8"]
    overlay = overlay_cam(base_rgb, cam, alpha=0.4)

    label_map = {0: "Non-COVID", 1: "COVID"}

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(base_rgb)
    axes[0].set_title("Masked Input")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Overlay")
    axes[1].axis("off")

    fig.suptitle(
        f"Grad-CAM ({case_name}) | true={label_map[true_label]} | "
        f"pred={label_map[pred_label]} | p(COVID)={pred_prob:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "figure_main_gradcam_example.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    print("Saved report figures to:", OUT_DIR)
    print("- figure_main_cm_resnet50.png")
    print("- figure_main_roc_topline.png")
    print("- figure_main_gradcam_example.png")


if __name__ == "__main__":
    main()
