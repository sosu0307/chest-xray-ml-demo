# src/features/image_io.py

import cv2
from .config import IMG_SIZE

def load_gray_image(path: str, size=IMG_SIZE):
    """Load grayscale uint8 image and resize to fixed size."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def load_mask_image(path: str, size=IMG_SIZE):
    """Load grayscale mask and resize with NEAREST, return uint8."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
