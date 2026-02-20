from .ROI_mask_img import roi_mask_from_isolated


def lung_area_ratio_img(img, threshold=0):
    """
    Lung area ratio = ROI pixels / total image pixels
    ROI pixels are defined by isolated image > threshold.
    """
    roi = roi_mask_from_isolated(img, threshold=threshold)
    if roi is None:
        return 0.0

    H, W = img.shape[:2]
    return float(roi.sum()) / float(H * W)
