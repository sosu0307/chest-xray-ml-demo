# src/features/extract.py
import os
import cv2
import numpy as np
from scipy.stats import skew, kurtosis

from .image_io import load_gray_image, load_mask_image
from .texture import entropy_gray, energy_gray, lbp_features
from .binary_masking import ensure_binary_mask, bbox_from_mask, split_left_right_mask

EPS = 1e-6


def asym_relative(L: float, R: float, eps: float = EPS) -> float:
    """Relative asymmetry: (L-R)/(L+R+eps)."""
    return (L - R) / (L + R + eps)


def crop_and_mask(img: np.ndarray, mask: np.ndarray, pad: int = 2) -> np.ndarray | None:
    """
    Crop to bounding box of mask (+padding) and zero-out outside mask inside the crop.
    Mask must be binary uint8 (0/255).
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

    img_c = img[y1:y2 + 1, x1:x2 + 1]
    m_c = mask[y1:y2 + 1, x1:x2 + 1]

    return cv2.bitwise_and(img_c, img_c, mask=m_c)


def _extract_features_single(img: np.ndarray) -> dict:
    """Compute features on ONE lung ROI (already cropped/masked)."""
    img_flat = img.flatten()
    pixels = img.astype(np.float32)

    mean_intensity = float(pixels.mean())
    rms_contrast = float(np.sqrt(np.mean((pixels - mean_intensity) ** 2)))

    dark_pixel_ratio = float((img < 50).mean())
    thr = np.percentile(img, 95)
    bright_pixel_ratio = float((img > thr).mean())

    laplacian_variance = float(cv2.Laplacian(img, cv2.CV_64F).var())
    img_skew = float(skew(img_flat))
    img_kurtosis = float(kurtosis(img_flat))

    ent = float(entropy_gray(img))
    en = float(energy_gray(img))

    # LBP expects uint8
    roi_lbp = img
    if roi_lbp.dtype != np.uint8:
        roi_lbp = np.clip(roi_lbp, 0, 255)
        if roi_lbp.max() <= 1.0:
            roi_lbp = roi_lbp * 255.0
        roi_lbp = roi_lbp.astype(np.uint8)

    lbp_mean, lbp_std, _ = lbp_features(roi_lbp)

    return {
        "mean_intensity": mean_intensity,
        "rms_contrast": rms_contrast,
        "dark_pixel_ratio": dark_pixel_ratio,
        "bright_pixel_ratio": bright_pixel_ratio,
        "laplacian_variance": laplacian_variance,
        "entropy": ent,
        "energy": en,
        "lbp_mean": float(lbp_mean),
        "lbp_std": float(lbp_std),
        "skew": img_skew,
        "kurtosis": img_kurtosis,
    }


def extract_features(
    image_path: str,
    mask_path: str,
    label: str,
    use_roi: bool = True,
    anatomical_swap: bool = True,
) -> dict | None:
    """
    Uses the provided mask_path (same filename as image in /masks).
    Splits mask into left/right halves. If anatomical_swap=True, swaps halves so:
      - output *_lunge-left  = anatomical LEFT
      - output *_lunge-right = anatomical RIGHT
    """
    img = load_gray_image(image_path)
    if img is None:
        return None

    mask = load_mask_image(mask_path)
    if mask is None:
        return None

    mask = ensure_binary_mask(mask)

    m_image_left, m_image_right = split_left_right_mask(mask)
    if m_image_left is None or m_image_right is None:
        return None

    # Typical CXR: image-left corresponds to anatomical RIGHT -> swap for anatomical labels
    if anatomical_swap:
        mL = m_image_right  # anatomical LEFT
        mR = m_image_left   # anatomical RIGHT
    else:
        mL = m_image_left
        mR = m_image_right

    if use_roi:
        imgL = crop_and_mask(img, mL, pad=2)
        imgR = crop_and_mask(img, mR, pad=2)
    else:
        imgL = cv2.bitwise_and(img, img, mask=mL)
        imgR = cv2.bitwise_and(img, img, mask=mR)

    if imgL is None or imgR is None or imgL.size == 0 or imgR.size == 0:
        return None

    fL = _extract_features_single(imgL)
    fR = _extract_features_single(imgR)

    row = {
        "image_name": os.path.basename(image_path),
        "label": label,
        "size_kb": float(os.path.getsize(image_path) / 1024.0),
    }

    for k, v in fL.items():
        row[f"{k}_lunge-left"] = float(v)
    for k, v in fR.items():
        row[f"{k}_lunge-right"] = float(v)

    for k in fL.keys():
        row[f"{k}_asym_relative"] = float(asym_relative(float(fL[k]), float(fR[k])))

    return row

