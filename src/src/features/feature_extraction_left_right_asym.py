import sys
from pathlib import Path

# Make "src" importable when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd

from src.features.LAR_img import lung_area_ratio_img
from src.features.framing_img import bbox_area_ratio_img
from src.features.texture_img import gradient_magnitude_std_img, high_frequency_energy_fft_img
from src.features.shape_img import opacity_compactness_img, opacity_eccentricity_img


EXTS = {".png", ".jpg", ".jpeg"}


# -----------------------
# helpers
# -----------------------
def load_gray(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def asym_relative(l: float, r: float, eps: float = 1e-6) -> float:
    return (float(l) - float(r)) / (float(l) + float(r) + eps)


def feature_block(img, roi_threshold=0):
    """Compute all features for ONE isolated-lung image."""
    if img is None:
        return {
            "lung_area_ratio": 0.0,
            "bbox_area_ratio": 0.0,
            "grad_mag_std": 0.0,
            "fft_hf_energy": 0.0,
            "opacity_compactness": 0.0,
            "opacity_eccentricity": 0.0,
        }

    return {
        "lung_area_ratio": lung_area_ratio_img(img, threshold=roi_threshold),
        "bbox_area_ratio": bbox_area_ratio_img(img, threshold=roi_threshold),
        "grad_mag_std": gradient_magnitude_std_img(img, threshold=roi_threshold),
        "fft_hf_energy": high_frequency_energy_fft_img(img, threshold=roi_threshold, normalize=True),
        "opacity_compactness": opacity_compactness_img(img, threshold=roi_threshold),
        "opacity_eccentricity": opacity_eccentricity_img(img, threshold=roi_threshold),
    }


def compute_left_right_asym(left_img, right_img, roi_threshold=0):
    lf = feature_block(left_img, roi_threshold=roi_threshold)
    rf = feature_block(right_img, roi_threshold=roi_threshold)

    out = {}
    for k in lf.keys():
        out[f"{k}_left"] = lf[k]
        out[f"{k}_right"] = rf[k]
        out[f"{k}_asym_relative"] = asym_relative(lf[k], rf[k])
    return out


def strip_type_suffix(stem: str):
    """
    Remove '_left' or '_right' or '_both' from filename stem.
    Example:
      'Covid-1_left' -> 'Covid-1'
    """
    s = stem
    for suf in ["_left", "_right", "_both"]:
        if s.lower().endswith(suf):
            return s[: -len(suf)]
    return s


def build_pairs_in_folder(folder: Path):
    """
    Find left/right image pairs based on naming:
      <base>_left.<ext> and <base>_right.<ext>
    Returns list of (base, left_path, right_path)
    """
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS]

    # Map: (base_lower) -> dict(left=Path, right=Path)
    pairs = {}
    for p in files:
        stem = p.stem  # without extension
        stem_low = stem.lower()

        if stem_low.endswith("_left"):
            base = strip_type_suffix(stem)
            key = base.lower()
            pairs.setdefault(key, {"base": base, "left": None, "right": None})
            pairs[key]["left"] = p

        elif stem_low.endswith("_right"):
            base = strip_type_suffix(stem)
            key = base.lower()
            pairs.setdefault(key, {"base": base, "left": None, "right": None})
            pairs[key]["right"] = p

        # we ignore "_both" here for now

    out = []
    for key, d in pairs.items():
        if d["left"] is not None and d["right"] is not None:
            out.append((d["base"], d["left"], d["right"]))

    return out


# -----------------------
# main runner
# -----------------------
def run(data_root: Path, out_csv: Path, roi_threshold=0):
    """
    Expects:
      data_root/covid/*.png  (contains *_left, *_right, *_both)
      data_root/non_covid/... (may contain subfolders)
    We will scan recursively and treat each folder that contains *_left/_right pairs as a 'class folder'.
    """
    rows = []

    # scan all directories under data_root
    all_dirs = [p for p in data_root.rglob("*") if p.is_dir()]
    all_dirs.insert(0, data_root)

    for folder in all_dirs:
        pairs = build_pairs_in_folder(folder)
        if not pairs:
            continue

        # label = top-level folder name under data_root (e.g. covid / non_covid / normal / etc.)
        # We'll define label as the relative path from data_root.
        label = str(folder.relative_to(data_root)).replace("\\", "/")

        print(f"{label}: found {len(pairs)} left/right pairs")

        for idx, (base, left_path, right_path) in enumerate(pairs, start=1):
            left_img = load_gray(left_path)
            right_img = load_gray(right_path)

            if left_img is None or right_img is None:
                continue

            row = {
                #"image_name": base,          # e.g., Covid-1
                "image_name": f"{base}_both",
                #"left_file": left_path.name,
                #"right_file": right_path.name,
                "label": label,
            }
            row.update(compute_left_right_asym(left_img, right_img, roi_threshold=roi_threshold))
            rows.append(row)

            if idx % 1000 == 0:
                print(f"{label}: processed {idx}/{len(pairs)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved features to: {out_csv}")
    print(f"Total samples: {len(df)}")


if __name__ == "__main__":
    DATA_ROOT = Path(
        r"D:\Data_Sceience_AI_ML\Data_Scientest\Projects\My_Project\Local\data\processed\Isolated_lung_images"
    )
    OUT_CSV = DATA_ROOT / "features_vijay_left_right_asym.csv"

    # If background has tiny noise, raise threshold to 5 or 10
    run(DATA_ROOT, OUT_CSV, roi_threshold=0)
