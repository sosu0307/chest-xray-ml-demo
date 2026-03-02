import cv2
import numpy as np
from .ROI_mask_img import roi_mask_from_isolated, crop_to_bbox


def gradient_magnitude_std_img(img, threshold=0):
    """
    Std of gradient magnitude inside ROI only.
    """
    roi = roi_mask_from_isolated(img, threshold=threshold)
    if roi is None:
        return 0.0

    img_c, roi_c = crop_to_bbox(img, roi, pad=2)
    if img_c is None:
        return 0.0

    gx = cv2.Sobel(img_c, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_c, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    vals = grad[roi_c]
    return float(vals.std()) if vals.size else 0.0


def high_frequency_energy_fft_img(img, threshold=0, cutoff_ratio=0.1, normalize=True):
    """
    High-frequency FFT energy inside ROI.
    Steps:
      1) crop to ROI bbox
      2) set outside ROI to 0
      3) FFT
      4) remove low-frequency center
      5) sum energy
      6) optionally normalize by ROI pixel count
    """
    roi = roi_mask_from_isolated(img, threshold=threshold)
    if roi is None:
        return 0.0

    img_c, roi_c = crop_to_bbox(img, roi, pad=2)
    if img_c is None:
        return 0.0

    # mask outside ROI to 0
    img_m = img_c.copy()
    img_m[~roi_c] = 0

    fft = np.fft.fftshift(np.fft.fft2(img_m))
    h, w = img_m.shape
    c = int(min(h, w) * cutoff_ratio)

    fft[h // 2 - c:h // 2 + c, w // 2 - c:w // 2 + c] = 0
    energy = float(np.sum(np.abs(fft) ** 2))

    if not normalize:
        return energy

    denom = float(roi_c.sum())
    return energy / max(denom, 1.0)
