# src/features/texture.py

import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from .config import LBP_P, LBP_R, LBP_METHOD


def entropy_gray(img_u8, bins=256) -> float:
    """Shannon entropy from grayscale histogram."""
    hist = np.histogram(img_u8, bins=bins, range=(0, 256), density=True)[0]
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def energy_gray(img_u8, bins=256) -> float:
    """Energy = sum(p^2) of normalized grayscale histogram."""
    p = np.histogram(img_u8, bins=bins, range=(0, 256), density=True)[0]
    return float(np.sum(p ** 2))


def lbp_features(img_u8, P=LBP_P, R=LBP_R, method=LBP_METHOD):
    """LBP mean/std + normalized histogram."""
    img_f = img_u8.astype(np.float32) / 255.0
    lbp = local_binary_pattern(img_f, P, R, method=method)

    lbp_mean = float(lbp.mean())
    lbp_std = float(lbp.std())

    n_bins = P + 2 if method == "uniform" else int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist / hist.sum() if hist.sum() > 0 else hist.astype(np.float32)

    return lbp_mean, lbp_std, hist


def compute_glcm_features(roi_img):
    """
    Computes rotation-invariant GLCM features for a specific lung ROI.
    
    Args:
        roi_img: NumPy array containing a single isolated lung

    Returns:
        glcm_results: Dictionary containing contrast, homogeneity, energy, and correlation
    """
    # Return 0 if image is all black
    if np.max(roi_img) == 0:
        return {f"glcm_{p}": 0.0 for p in ['contrast', 'homogeneity', 'energy', 'correlation']}

    # Create matrix of GLCM values
    glcm = graycomatrix(roi_img, distances=[4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
                        
    # Define GLCM properties of interest
    properties = ['contrast', 'homogeneity', 'energy', 'correlation']

    return {f"glcm_{p}": np.mean(graycoprops(glcm, p), axis=1)[0] for p in properties}