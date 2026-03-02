import os
import cv2
import numpy as np
import pandas as pd
from skimage import img_as_ubyte
from .texture import compute_glcm_features
from .stats import compute_skew, compute_kurtosis


def prepare_image_for_extraction(image_path):
    """
    Ensures image is 224x224, and np.uint8.

    Args:
        image_path: filepath to image

    Returns:
        img: image that is grayscale, size 224x224, and type np.uint8
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    if img.dtype != np.uint8:
        if img.max() <= 1.0 and img.dtype.kind == 'f':
            img = img_as_ubyte(img)
        elif img.max() > 255:
            img = (img / (img.max() / 255.0)).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


def extract_lung_rois(img):
    """
    Identifies the two largest contours in a masked image and separates them 
    into individual images for the anatomical right and left lungs.

    Args:
        img: NumPy array of the masked grayscale image

    Returns:
        masks[0]: Image containing only the anatomical right lung (image left)
        masks[1]: Image containing only the anatomical left lung (image right)
    """
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area and take top 2 (assumed to be the two lung lobes)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    
    # Fallback if masking/segmentation failed to find two distinct areas
    if len(contours) < 2:
        return img, np.zeros_like(img)

    # Sort by X-coordinate. Anatomical Right Lung is on the left side of the image.
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    masks = []
    for cnt in contours:
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        # Isolate the lung pixels by applying the mask to the original image
        masks.append(cv2.bitwise_and(img, mask))
        
    return masks[0], masks[1]


def calculate_nrms_contrast(roi_img):
    """
    Calculates the Normalized Root Mean Square (NRMS) contrast for 
    non-zero pixels in the lung ROI.

    Args:
        roi_img: NumPy array of an isolated lung image

    Returns:
        nrms_val: Standard deviation divided by mean intensity (float)
    """
    pixels = roi_img[roi_img > 0]
    if len(pixels) == 0: 
        return 0.0
    mean, std = np.mean(pixels), np.std(pixels)

    return std / mean if mean > 0 else 0.0


def get_all_features(roi_img):
    """
    Aggregates all texture and distribution features for a single lung ROI.

    Args:
        roi_img: NumPy array of an isolated lung image

    Returns:
        feats: Dictionary containing all computed features for the ROI
    """
    feats = compute_glcm_features(roi_img)
    feats['skew'] = compute_skew(roi_img)
    feats['kurtosis'] = compute_kurtosis(roi_img)
    return feats


def extract_features(image_folders_list):
    """
    Iterates through folders to extract lung-specific features and 
    calculates asymmetry indices between left and right lungs.

    Args:
        image_folders_list: List of filepaths to image directories

    Returns:
        df: Pandas DataFrame with columns for left, right, and asymmetry features
    """
    features_data = []
    epsilon = 1e-6

    for image_folder in image_folders_list:
        if not os.path.exists(image_folder): 
            continue
        for image_name in os.listdir(image_folder):
            if image_name.lower().endswith(".png"):
                image_path = os.path.join(image_folder, image_name)
                img = prepare_image_for_extraction(image_path)

                # Isolate lungs
                right_roi, left_roi = extract_lung_rois(img)

                # Compute features per lung
                r_feats = get_all_features(right_roi)
                l_feats = get_all_features(left_roi)

                row = {"image_name": image_name}
                
                # Dynamic column generation for Left, Right, and Asymmetry
                for key in r_feats.keys():
                    L, R = l_feats[key], r_feats[key]
                    row[f"left_{key}"] = L
                    row[f"right_{key}"] = R
                    # (L - R) / (L + R + ε)
                    row[f"asym_{key}"] = (L - R) / (L + R + epsilon)

                features_data.append(row)

    return pd.DataFrame(features_data)

if __name__ == '__main__':
    image_folders_list = [
        '/Users/Ice/Repositories/Data Scientest/Project/Isolated_lung_images/covid',
        '/Users/Ice/Repositories/Data Scientest/Project/Isolated_lung_images/non_covid'
    ]

    df = extract_features(image_folders_list)
    df.to_csv("glcm_skew_kurtosis_features.csv", index=False)
    print(f"Extraction complete. Processed {len(df)} images.")
