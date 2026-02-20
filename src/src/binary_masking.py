# src/features/binary_masking.py
import cv2
import numpy as np

def ensure_binary_mask(mask):
    """Convert mask to uint8 binary (0 or 255)."""
    if mask is None:
        return None
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return ((mask > 127).astype(np.uint8) * 255)

def bbox_from_mask(mask):
    """Return bbox (x1, y1, x2, y2) of white pixels in mask."""
    mask = ensure_binary_mask(mask)
    if mask is None:
        return None

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def split_left_right_mask(mask):
    """
    Split whole lung mask into left and right lung masks using connected components.
    Returns (left_mask, right_mask) as 0/255 uint8.
    """
    mask = ensure_binary_mask(mask)
    if mask is None:
        return None, None

    mask01 = (mask > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask01, connectivity=8)

    if n <= 1:
        return None, None

    comps = []
    for lab in range(1, n):
        area = stats[lab, cv2.CC_STAT_AREA]
        cx, _ = centroids[lab]
        comps.append((lab, area, cx))

    comps.sort(key=lambda x: x[1], reverse=True)
    comps = comps[:2]

    if len(comps) == 1:
        lab, _, _ = comps[0]
        return (labels == lab).astype(np.uint8) * 255, None

    (lab1, _, cx1), (lab2, _, cx2) = comps
    m1 = (labels == lab1).astype(np.uint8) * 255
    m2 = (labels == lab2).astype(np.uint8) * 255

    return (m1, m2) if cx1 < cx2 else (m2, m1)
