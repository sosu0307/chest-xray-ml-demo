import numpy as np


def roi_mask_from_isolated(img, threshold=0):
    """
    ROI mask from isolated lung image:
    outside lung is 0 (black), inside lung > threshold.
    Returns boolean mask.
    """
    if img is None:
        return None
    return img > threshold


def bbox_from_roi(roi_mask):
    """
    Bounding box from boolean ROI mask.
    Returns (x1, y1, x2, y2) or None.
    """
    if roi_mask is None:
        return None
    ys, xs = np.where(roi_mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_to_bbox(img, roi_mask, pad=2):
    """
    Crop image and ROI mask to bbox (+ padding).
    Returns (img_crop, mask_crop) or (None, None)
    """
    bbox = bbox_from_roi(roi_mask)
    if bbox is None:
        return None, None

    x1, y1, x2, y2 = bbox
    H, W = img.shape[:2]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W - 1, x2 + pad)
    y2 = min(H - 1, y2 + pad)

    return img[y1:y2 + 1, x1:x2 + 1], roi_mask[y1:y2 + 1, x1:x2 + 1]
