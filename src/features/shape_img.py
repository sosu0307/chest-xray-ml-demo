import cv2
import numpy as np
from .ROI_mask_img import roi_mask_from_isolated, crop_to_bbox


def _opacity_binary_inside_roi(img, threshold=0):
    """
    Otsu threshold on ROI crop, then keep only pixels within ROI.
    Returns binary image (uint8 0/255) inside ROI crop.
    """
    roi = roi_mask_from_isolated(img, threshold=threshold)
    if roi is None:
        return None, None

    img_c, roi_c = crop_to_bbox(img, roi, pad=2)
    if img_c is None:
        return None, None

    _, binary = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # keep only ROI
    binary_roi = binary.copy()
    binary_roi[~roi_c] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel)
    binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel)

    return binary_roi, roi_c


def opacity_compactness_img(img, threshold=0, min_area=50):
    """
    Compactness of largest opacity region inside ROI.
    compactness = 4*pi*Area / Perimeter^2
    """
    binary_roi, _ = _opacity_binary_inside_roi(img, threshold=threshold)
    if binary_roi is None:
        return 0.0

    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < min_area:
        return 0.0

    perim = float(cv2.arcLength(c, True))
    if perim <= 1e-6:
        return 0.0

    return float((4.0 * np.pi * area) / (perim * perim + 1e-6))


def opacity_eccentricity_img(img, threshold=0, min_area=50):
    """
    Eccentricity of largest opacity region inside ROI.
    Uses ellipse fit: ecc = sqrt(1 - (b^2/a^2))
    """
    binary_roi, _ = _opacity_binary_inside_roi(img, threshold=threshold)
    if binary_roi is None:
        return 0.0

    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < min_area or len(c) < 5:
        return 0.0

    (_, _), (major, minor), _ = cv2.fitEllipse(c)
    a = max(major, minor) / 2.0
    b = min(major, minor) / 2.0
    if a <= 1e-6:
        return 0.0

    ecc = np.sqrt(max(0.0, 1.0 - (b * b) / (a * a + 1e-6)))
    return float(ecc)
