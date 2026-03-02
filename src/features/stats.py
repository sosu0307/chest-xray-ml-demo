from scipy.stats import skew, kurtosis


def compute_skew(roi_img):
    """
    Calculates the skewness of pixel intensities within the lung area, 
    excluding the black background.

    Args:
        roi_img: NumPy array of an isolated lung image

    Returns:
        The calculated skewness (float)
    """
    pixels = roi_img[roi_img > 0]
    return skew(pixels) if len(pixels) > 0 else 0.0


def compute_kurtosis(roi_img):
    """
    Calculates the kurtosis of pixel intensities within the lung area, 
    excluding the black background.

    Args:
        roi_img: NumPy array of an isolated lung image

    Returns:
        The calculated kurtosis (float)
    """
    pixels = roi_img[roi_img > 0]
    return kurtosis(pixels) if len(pixels) > 0 else 0.0