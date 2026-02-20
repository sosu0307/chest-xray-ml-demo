# src/features/build_dataset.py
import os
from pathlib import Path
import pandas as pd

from .extract import extract_features

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def build_features_dataset(
    data_root: str,
    use_roi: bool = True,
    anatomical_swap: bool = True,
    only_both: bool = False,
    class_folders=None,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Dataset-specific pipeline for COVID-19 Radiography Dataset:

      RAW_ROOT/
        COVID/images/*.png
        COVID/masks/*.png
        Normal/images/*.png
        Normal/masks/*.png
        Lung_Opacity/images/*.png
        Lung_Opacity/masks/*.png
        Viral Pneumonia/images/*.png
        Viral Pneumonia/masks/*.png
    """
    data_root = Path(data_root)
    rows = []   # <-- THIS WAS BROKEN BEFORE

    # determine class folders
    if class_folders is None:
        class_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    else:
        lookup = {d.name.casefold(): d for d in data_root.iterdir() if d.is_dir()}
        class_dirs = [lookup[str(n).casefold()] for n in class_folders if str(n).casefold() in lookup]

    total_images = 0
    total_masks_missing = 0
    total_rows = 0

    for class_dir in class_dirs:
        label = class_dir.name
        images_dir = class_dir / "images"
        masks_dir = class_dir / "masks"

        if not images_dir.is_dir():
            print(f"WARNING: images folder not found: {images_dir}")
            continue
        if not masks_dir.is_dir():
            print(f"WARNING: masks folder not found: {masks_dir}")
            continue

        for img_path in images_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
                continue

            if only_both and "_both" not in img_path.stem.casefold():
                continue

            total_images += 1

            mask_path = masks_dir / img_path.name
            if not mask_path.is_file():
                total_masks_missing += 1
                continue

            row = extract_features(
                image_path=str(img_path),
                mask_path=str(mask_path),
                label=label,
                use_roi=use_roi,
                anatomical_swap=anatomical_swap,
            )

            if row is not None:
                rows.append(row)
                total_rows += 1

    if debug:
        print("DEBUG COUNTS")
        print("  total images scanned :", total_images)
        print("  masks missing       :", total_masks_missing)
        print("  rows created        :", total_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if df["image_name"].duplicated().any():
        dupes = df[df["image_name"].duplicated(keep=False)].sort_values("image_name")
        raise ValueError(f"Duplicate image_name found. Example rows:\n{dupes.head(20)}")

    meta = ["image_name", "label", "size_kb"]
    other_cols = [c for c in df.columns if c not in meta]
    df = df[meta + sorted(other_cols)].sort_values(["label", "image_name"]).reset_index(drop=True)

    return df


def save_features_dataset(df: pd.DataFrame, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
