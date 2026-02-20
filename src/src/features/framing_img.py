from .ROI_mask_img import roi_mask_from_isolated, bbox_from_roi


def bbox_area_ratio_img(img, threshold=0):
    """
    bbox area ratio = bbox_area(ROI) / image_area
    """
    roi = roi_mask_from_isolated(img, threshold=threshold)
    if roi is None:
        return 0.0

    bbox = bbox_from_roi(roi)
    if bbox is None:
        return 0.0

    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1 + 1) * (y2 - y1 + 1)

    H, W = img.shape[:2]
    return float(bbox_area) / float(H * W)
