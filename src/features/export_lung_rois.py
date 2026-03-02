import os
from pathlib import Path

import cv2
import numpy as np

from src.features.image_io import load_gray_image, load_mask_image
from src.features.binary_masking import (
    ensure_binary_mask,
    bbox_from_mask,
    split_left_right_mask,
)

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def crop_and_mask(img: np.ndarray, mask: np.ndarray, pad: int = 2) -> np.ndarray | None:
    """
    Crop to the bounding box of the mask (with optional padding),
    then set all pixels outside the mask to 0 inside that crop.
    """
    mask = ensure_binary_mask(mask)
    bb = bbox_from_mask(mask)
    if bb is None:
        return None

    x1, y1, x2, y2 = bb
    h, w = img.shape[:2]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    img_c = img[y1 : y2 + 1, x1 : x2 + 1]
    m_c = mask[y1 : y2 + 1, x1 : x2 + 1]

    return cv2.bitwise_and(img_c, img_c, mask=m_c)


def _norm_stem(stem: str) -> str:
    """
    Normalize file stems so that image and mask match more often.
    Example: 'covid1_mask' -> 'covid1'
    """
    s = stem.casefold()
    for suffix in [
        "_mask", "-mask", "mask",
        "_lung", "-lung", "_lungs", "-lungs",
        "_seg", "-seg", "_segmentation"
    ]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def build_mask_index(mask_dir: Path) -> dict:
    """
    Build dict: {normalized_stem: full_mask_path}
    """
    idx = {}
    for p in mask_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            idx[_norm_stem(p.stem)] = str(p)
    return idx


def export_rois_for_class(
    class_dir: Path,
    out_dir: Path,
    pad: int = 2,
    anatomical_swap: bool = False,
) -> tuple[int, int]:
    """
    Export 3 ROI images per input image:
      <id>_left.png
      <id>_right.png
      <id>_both.png

    LEFT/RIGHT default is image-coordinate:
      - left = left half of image mask
      - right = right half of image mask

    If anatomical_swap=True, left/right are swapped (useful if your dataset flips anatomy).
    """
    images_dir = class_dir / "images"
    masks_dir = class_dir / "maske"

    if not images_dir.is_dir():
        print(f"WARNING: images folder not found: {images_dir}")
        return (0, 0)
    if not masks_dir.is_dir():
        print(f"WARNING: mask folder not found: {masks_dir}")
        return (0, 0)

    mask_index = build_mask_index(masks_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    saved = 0

    for img_path in images_dir.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue

        total += 1
        img = load_gray_image(str(img_path))
        if img is None:
            continue

        base_id = _norm_stem(img_path.stem)
        mask_path = mask_index.get(base_id)
        if mask_path is None:
            # no matching mask -> skip
            continue

        mask = load_mask_image(mask_path)
        if mask is None:
            continue

        mask = ensure_binary_mask(mask)

        # both lungs ROI
        both_roi = crop_and_mask(img, mask, pad=pad)
        if both_roi is None or both_roi.size == 0:
            continue

        # split mask into left/right (image-coordinate)
        mL, mR = split_left_right_mask(mask)
        if mL is None or mR is None:
            continue

        # optional swap for anatomical convention
        if anatomical_swap:
            mL, mR = mR, mL

        left_roi = crop_and_mask(img, mL, pad=pad)
        right_roi = crop_and_mask(img, mR, pad=pad)

        if left_roi is None or right_roi is None or left_roi.size == 0 or right_roi.size == 0:
            continue

        # Save
        out_left = out_dir / f"{base_id}_left.png"
        out_right = out_dir / f"{base_id}_right.png"
        out_both = out_dir / f"{base_id}_both.png"

        cv2.imwrite(str(out_left), left_roi)
        cv2.imwrite(str(out_right), right_roi)
        cv2.imwrite(str(out_both), both_roi)

        saved += 1

    return total, saved


def export_all_classes(
    raw_root: str,
    out_root: str,
    classes=("covid", "normal", "viral pneumonia", "lung opacity"),
    pad: int = 2,
    anatomical_swap: bool = False,
):
    """
    Process folder structure:
      RAW_ROOT/<class>/images/*
      RAW_ROOT/<class>/maske/*

    Output:
      OUT_ROOT/<class>/<id>_left.png
      OUT_ROOT/<class>/<id>_right.png
      OUT_ROOT/<class>/<id>_both.png
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    for cls in classes:
        class_dir = raw_root / cls
        out_dir = out_root / cls

        total, saved = export_rois_for_class(
            class_dir=class_dir,
            out_dir=out_dir,
            pad=pad,
            anatomical_swap=anatomical_swap,
        )
        print(f"[{cls}] images found: {total}, exported triplets: {saved}")


if __name__ == "__main__":
    RAW_ROOT = r"D:\DS_ML\COVID-19_Radiography_Dataset\raw"
    OUT_ROOT = r"D:\DS_ML\COVID_Projekt\lung_rois_export"

    export_all_classes(
        raw_root=RAW_ROOT,
        out_root=OUT_ROOT,
        pad=2,
        anatomical_swap=False,  # set True if you need anatomical swap
    )
