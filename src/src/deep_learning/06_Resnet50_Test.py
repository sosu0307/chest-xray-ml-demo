import os
from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    auc,
    roc_curve,
)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =========================
# 1) Config & Paths
# =========================
DATA_ROOT = Path("/kaggle/input/data-bin/data_bin")
MODEL_PATH = Path("/kaggle/working/models/resnet50_best.pt")
OUT_DIR = Path("/kaggle/working/eval_resnet50")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "resnet50"
IMG_SIZE = 224
BATCH_SIZE = 16
ALPHA = 0.15
NUM_GRADCAM_SAMPLES = 8  # how many Grad-CAM examples to save
SEED = 42


# =========================
# 2) Reproducibility
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 3) Dataset Class
# =========================
class XrayMaskedBinaryDataset(Dataset):
    """
    Returns:
        image_tensor: normalized tensor for model input
        label_tensor: 0/1
        meta: dict containing original paths for traceability
    """

    def __init__(self, split_dir: Path, img_size: int = 224, alpha: float = 0.15):
        self.split_dir = split_dir
        self.alpha = alpha
        self.img_size = img_size
        self.samples = []

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

        img_np = np.array(img.resize((self.img_size, self.img_size)), dtype=np.float32)
        mask_np = (
            np.array(mask.resize((self.img_size, self.img_size)), dtype=np.float32)
            / 255.0
        )
        mask_np = (mask_np > 0.5).astype(np.float32)

        # lung-focused masking
        masked = img_np * (self.alpha + (1.0 - self.alpha) * mask_np)
        masked_pil = Image.fromarray(np.clip(masked, 0, 255).astype(np.uint8)).convert(
            "RGB"
        )

        x = self.img_transform(masked_pil)
        y = torch.tensor(s["label"], dtype=torch.long)

        meta = {
            "img_path": str(s["img"]),
            "mask_path": str(s["mask"]),
            "class_name": s["class_name"],
            "filename": s["img"].name,
            "masked_rgb_uint8": np.array(masked_pil),  # for Grad-CAM overlay
        }
        return x, y, meta


# =========================
# 4) Grad-CAM utilities
# =========================
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: shape [1, 3, H, W]
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        score.backward()

        # activations/gradients shape: [1, C, h, w]
        grads = self.gradients[0]  # [C, h, w]
        acts = self.activations[0]  # [C, h, w]

        # channel weights: global avg pooling on grads
        weights = grads.mean(dim=(1, 2))  # [C]
        cam = torch.zeros(acts.shape[1:], device=acts.device)

        for c, w in enumerate(weights):
            cam += w * acts[c]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), class_idx


def overlay_heatmap_on_image(
    rgb_uint8: np.ndarray, cam_2d: np.ndarray, alpha: float = 0.4
):
    """
    rgb_uint8: HxWx3 uint8 image
    cam_2d: normalized [0,1] map in low/high resolution
    """
    h, w, _ = rgb_uint8.shape
    cam_img = Image.fromarray((cam_2d * 255).astype(np.uint8)).resize(
        (w, h), resample=Image.BILINEAR
    )
    cam_resized = np.array(cam_img) / 255.0

    # Use matplotlib colormap to colorize CAM (jet for visibility)
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam_resized)[..., :3]  # drop alpha channel
    base = rgb_uint8.astype(np.float32) / 255.0

    overlay = (1 - alpha) * base + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8), (heatmap * 255).astype(np.uint8)


# =========================
# 5) Plot functions
# =========================
def plot_confusion_matrix(cm, out_path: Path):
    # custom blue-ish colormap (explicitly not green)
    blue_cmap = LinearSegmentedColormap.from_list(
        "custom_blue", ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=blue_cmap)
    plt.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix - ResNet50 (Test)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-COVID", "COVID"])
    ax.set_yticklabels(["Non-COVID", "COVID"])

    # annotate values
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - ResNet50 (Test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true, y_prob, out_path: Path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - ResNet50 (Test)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================
# 6) Main Test + Eval + Grad-CAM
# =========================
def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Model: {MODEL_NAME} on {device}")

    # Dataset / loader
    test_ds = XrayMaskedBinaryDataset(
        DATA_ROOT / "test", img_size=IMG_SIZE, alpha=ALPHA
    )

    # collate meta safely
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
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Loading weights from: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=device)
    # support both plain state_dict and checkpoint dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)

    model.to(device).eval()

    y_true, y_pred, y_prob = [], [], []
    all_metas = []

    print(f"Processing {len(test_ds)} test images...")
    with torch.no_grad():
        for x, y, metas in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            all_metas.extend(metas)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 40)
    print(f"TEST RESULTS: {MODEL_NAME.upper()}")
    print("=" * 40)
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"F1-Score   : {f1:.4f}")
    print(f"ROC-AUC    : {auc_roc:.4f}")
    print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=["Non-COVID", "COVID"], digits=4, zero_division=0
    )
    print("\nClassification Report:\n")
    print(report)

    # Save metrics text
    metrics_txt = OUT_DIR / "metrics_report.txt"
    with open(metrics_txt, "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Test size: {len(test_ds)}\n\n")
        f.write(f"Accuracy  : {acc:.6f}\n")
        f.write(f"Precision : {prec:.6f}\n")
        f.write(f"Recall    : {rec:.6f}\n")
        f.write(f"F1-Score  : {f1:.6f}\n")
        f.write(f"ROC-AUC   : {auc_roc:.6f}\n\n")
        f.write("Confusion Matrix [ [TN, FP], [FN, TP] ]\n")
        f.write(f"{cm}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Save plots
    plot_confusion_matrix(cm, OUT_DIR / "confusion_matrix_resnet50.png")
    plot_roc_curve(y_true, y_prob, OUT_DIR / "roc_curve_resnet50.png")
    plot_pr_curve(y_true, y_prob, OUT_DIR / "pr_curve_resnet50.png")

    # ===== Grad-CAM =====
    gradcam_dir = OUT_DIR / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    gradcam = GradCAM(model=model, target_layer=model.layer4[-1].conv3)

    # Select candidates: TP/FP/FN priority, then random fill
    indices = list(range(len(y_true)))
    tp_idx = [i for i in indices if y_true[i] == 1 and y_pred[i] == 1]
    fp_idx = [i for i in indices if y_true[i] == 0 and y_pred[i] == 1]
    fn_idx = [i for i in indices if y_true[i] == 1 and y_pred[i] == 0]
    tn_idx = [i for i in indices if y_true[i] == 0 and y_pred[i] == 0]

    picked = []
    for group in [tp_idx[:3], fp_idx[:2], fn_idx[:2], tn_idx[:1]]:
        picked.extend(group)

    if len(picked) < NUM_GRADCAM_SAMPLES:
        remaining = [i for i in indices if i not in picked]
        random.shuffle(remaining)
        picked.extend(remaining[: NUM_GRADCAM_SAMPLES - len(picked)])

    picked = picked[:NUM_GRADCAM_SAMPLES]

    for rank, i in enumerate(picked, start=1):
        sample = test_ds[i]
        x_single = sample[0].unsqueeze(0).to(device)  # [1,3,H,W]
        true_label = int(sample[1].item())
        meta = sample[2]

        # predicted class
        with torch.no_grad():
            logits = model(x_single)
            probs = torch.softmax(logits, dim=1)
            pred_label = int(torch.argmax(probs, dim=1).item())
            pred_prob = float(probs[0, pred_label].item())

        # Grad-CAM for predicted class
        cam_map, _ = gradcam.generate(x_single, class_idx=pred_label)

        base_rgb = meta["masked_rgb_uint8"]  # already 224x224 RGB
        overlay, heatmap = overlay_heatmap_on_image(base_rgb, cam_map, alpha=0.40)

        # save side-by-side figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(base_rgb)
        axes[0].set_title("Masked Input")
        axes[0].axis("off")

        axes[1].imshow(heatmap)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        label_map = {0: "Non-COVID", 1: "COVID"}
        fig.suptitle(
            f"#{rank} | file={meta['filename']} | true={label_map[true_label]} | "
            f"pred={label_map[pred_label]} | conf={pred_prob:.3f}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(
            gradcam_dir / f"gradcam_{rank:02d}_{meta['filename']}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    gradcam.remove_hooks()

    print("\nSaved outputs to:", OUT_DIR)
    print(f"- {OUT_DIR / 'metrics_report.txt'}")
    print(f"- {OUT_DIR / 'confusion_matrix_resnet50.png'}")
    print(f"- {OUT_DIR / 'roc_curve_resnet50.png'}")
    print(f"- {OUT_DIR / 'pr_curve_resnet50.png'}")
    print(f"- {gradcam_dir} (Grad-CAM images)")


if __name__ == "__main__":
    main()
