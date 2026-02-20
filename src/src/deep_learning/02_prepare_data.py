"""
Prepare Binary Dataset (COVID vs NON-COVID) with paired lung masks

Input dataset structure (raw):
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

Output dataset structure (data_bin):
data_bin/
  train/
    covid/
      images/
      masks/
    noncovid/
      images/
      masks/
  val/...
  test/...

What this script does:
- Pairs each image with its corresponding mask (same filename stem)
- Maps labels:
    covid -> covid
    normal/viral_pneumonia/lung_opacity -> noncovid
- Stratified split into train/val/test
- Copies files into output folders

Run:
python src/data/02_prepare_data_bin.py
"""

from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


# =========================
# 1) Config
# =========================

RAW_DIR = Path("raw")
OUT_DIR = Path("data_bin")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

IMG_EXTS = (".png", ".jpg", ".jpeg")


# 4-class -> binary mapping
CLASSES = {
    "covid": RAW_DIR / "COVID",
    "normal": RAW_DIR / "Normal",
    "viral_pneumonia": RAW_DIR / "Viral Pneumonia",
    "lung_opacity": RAW_DIR / "Lung_Opacity",
}


def to_binary_label(cls_name: str) -> str:
    return "covid" if cls_name == "covid" else "noncovid"


# =========================
# 2) Helpers
# =========================


def list_images(folder: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted([p for p in files if p.is_file()])


def build_mask_index(mask_dir: Path):
    """
    Map: stem -> mask_path
    Masks are often PNG. We accept png/jpg/jpeg just in case.
    """
    idx = {}
    for p in list_images(mask_dir):
        idx[p.stem] = p
    return idx


def make_output_dirs(out_root: Path):
    for split in ["train", "val", "test"]:
        for cls in ["covid", "noncovid"]:
            (out_root / split / cls / "images").mkdir(parents=True, exist_ok=True)
            (out_root / split / cls / "masks").mkdir(parents=True, exist_ok=True)


# =========================
# 3) Collect paired samples
# =========================

# Each sample: (img_path, mask_path, binary_label)
samples = []

missing_masks_total = 0

for cls_name, cls_path in CLASSES.items():
    images_dir = cls_path / "images"
    masks_dir = cls_path / "masks"

    imgs = list_images(images_dir)
    mask_index = build_mask_index(masks_dir)

    for img_path in imgs:
        mask_path = mask_index.get(img_path.stem)  # same filename stem

        if mask_path is None:
            missing_masks_total += 1
            continue  # skip samples without mask

        bin_label = to_binary_label(cls_name)
        samples.append((img_path, mask_path, bin_label))

print("\n=== Pairing summary ===")
print("Total paired samples:", len(samples))
print("Skipped (missing mask):", missing_masks_total)

if len(samples) == 0:
    raise RuntimeError("No paired samples found. Check raw folder structure.")


# =========================
# 4) Stratified split
# =========================

X = samples
y = [1 if s[2] == "covid" else 0 for s in samples]  # covid=1, noncovid=0

# train vs temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=(1.0 - TRAIN_RATIO), random_state=SEED, stratify=y
)

# temp -> val vs test
val_size_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=(1.0 - val_size_of_temp),
    random_state=SEED,
    stratify=y_temp,
)

print("\n=== Split sizes ===")
print("train:", len(X_train), "val:", len(X_val), "test:", len(X_test))


# =========================
# 5) Copy files into output structure
# =========================

make_output_dirs(OUT_DIR)


def copy_split(split_samples, split_name: str):
    counts = {"covid": 0, "noncovid": 0}

    for img_path, mask_path, label in split_samples:
        dst_img = OUT_DIR / split_name / label / "images" / img_path.name
        dst_msk = OUT_DIR / split_name / label / "masks" / mask_path.name

        shutil.copy2(img_path, dst_img)
        shutil.copy2(mask_path, dst_msk)

        counts[label] += 1

    return counts


counts_train = copy_split(X_train, "train")
counts_val = copy_split(X_val, "val")
counts_test = copy_split(X_test, "test")

print("\n=== Output counts (binary) ===")
print("train:", counts_train)
print("val  :", counts_val)
print("test :", counts_test)

print("\nDone. Output folder:")
print(OUT_DIR.resolve())
