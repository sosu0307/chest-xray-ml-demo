"""
Exploratory Data Analysis (EDA)
COVID-19 Chest X-Ray Radiography Dataset

Goal:
- Understand the dataset before Deep Learning
- Check class distribution
- Verify image format (grayscale / RGB)
- Inspect lung masks
- Visualize sample images with masks

Dataset structure (raw):

raw/
├── COVID/
│   ├── images/
│   └── masks/
├── NORMAL/
│   ├── images/
│   └── masks/
├── VIRAL_PNEUMONIA/
│   ├── images/
│   └── masks/
└── LUNG_OPACITY/
    ├── images/
    └── masks/

Run:
python src/DL/01_eda.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =========================
# 1. Dataset paths
# =========================

RAW_DIR = Path("raw")

CLASSES = {
    "covid": RAW_DIR / "COVID",
    "normal": RAW_DIR / "Normal",
    "viral_pneumonia": RAW_DIR / "Viral Pneumonia",
    "lung_opacity": RAW_DIR / "Lung_Opacity",
}


# =========================
# 2. Count images per class
# =========================

print("\n=== Class distribution ===")

class_counts = {}

for cls, cls_path in CLASSES.items():
    images_dir = cls_path / "images"
    n_images = len(list(images_dir.glob("*.png")))
    class_counts[cls] = n_images
    print(f"{cls}: {n_images} images")


# =========================
# 3. Plot class distribution
# =========================

plt.figure()
plt.bar(class_counts.keys(), class_counts.values())
plt.title("Class distribution")
plt.xticks(rotation=20)
plt.tight_layout()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/class_distribution.png", dpi=200)
plt.close()


# =========================
# 4. Inspect one image (channels, size)
# =========================

example_img_path = next((CLASSES["covid"] / "images").glob("*.png"))
img = Image.open(example_img_path)
img_array = np.array(img)

print("\n=== Image inspection ===")
print("Example image path:", example_img_path)
print("PIL mode:", img.mode)
print("Image shape:", img_array.shape)
print("Pixel min / max:", img_array.min(), img_array.max())


# =========================
# 5. Inspect one mask
# =========================

example_mask_path = next((CLASSES["covid"] / "masks").glob("*.png"))
mask = Image.open(example_mask_path)
mask_array = np.array(mask)

print("\n=== Mask inspection ===")
print("Example mask path:", example_mask_path)
print("Mask shape:", mask_array.shape)
print("Mask min / max:", mask_array.min(), mask_array.max())
print("Unique mask values:", np.unique(mask_array))


# =========================
# 6. Visualize image and mask overlay
# =========================

plt.figure(figsize=(5, 5))
plt.imshow(img, cmap="gray")
plt.imshow(mask_array, alpha=0.3)
plt.title("Chest X-ray with lung mask overlay")
plt.axis("off")
Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/Visualize image and mask overlay.png", dpi=200)
plt.close()


# =========================
# 7. Visualize random samples per class
# =========================

print("\n=== Visual samples ===")

plt.figure(figsize=(10, 8))
plot_index = 1

for cls, cls_path in CLASSES.items():
    images = list((cls_path / "images").glob("*.png"))
    masks = list((cls_path / "masks").glob("*.png"))

    if len(images) == 0:
        continue

    img_path = images[0]
    mask_path = masks[0]

    img = Image.open(img_path)
    mask = Image.open(mask_path)

    plt.subplot(2, 2, plot_index)
    plt.imshow(img, cmap="gray")
    plt.imshow(mask, alpha=0.3)
    plt.title(cls)
    plt.axis("off")

    plot_index += 1

plt.suptitle("Sample X-rays with lung masks")
plt.tight_layout()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/VisualizVisualize random samples per class.png", dpi=200)
plt.close()


print("\nEDA completed.")
