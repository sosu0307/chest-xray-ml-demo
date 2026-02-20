import os
import cv2
import numpy as np
import random

# ============================
# SETTINGS 
# ============================
IMG_EXTS = (".png", ".jpg", ".jpeg")

FRAC = 0.30            # sample 30% of images to estimate blur threshold
MIN_SAMPLES = 300      # minimum sample for stable percentile
MAX_SAMPLES = 2000     # max sample for speed
SEED = 42

FINAL_SIZE = (224, 224)
CROP_MARGIN = 5
PRINT_EVERY = 2000      # print progress during blur threshold calc


# ============================
# 1) Load & Prepare Image (REFERENCE STYLE)
# ============================
def load_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, FINAL_SIZE)  # Uniform size → stable features (as reference)
    return img


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    mask = cv2.resize(mask, FINAL_SIZE, interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# ============================
# 2) FAST listing helper
# ============================
def list_image_files_fast(folder, exts=IMG_EXTS):
    files = []
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(exts):
                files.append(entry.path)
    return files


def choose_sample_size(n_total, frac=FRAC, min_samples=MIN_SAMPLES, max_samples=MAX_SAMPLES):
    n = int(n_total * frac)
    n = max(min_samples, n)
    n = min(max_samples, n)
    n = min(n_total, n)
    return n


# ============================
# 3) Automatic Blur Threshold per Class (REFERENCE STYLE, but SAMPLED)
# ============================
def calculate_blur_threshold_sampled(image_folder, percentile=20):
    """
    Same logic as your reference:
    - load_gray_image() -> 224x224
    - Laplacian variance
    But we do it on a percentage sample for speed.
    """
    all_files = list_image_files_fast(image_folder)
    if len(all_files) == 0:
        return 10.0

    n_sample = choose_sample_size(len(all_files))
    random.seed(SEED)
    files = random.sample(all_files, n_sample)

    print(f"  Blur threshold: sampling {n_sample} images out of {len(all_files)} total...")

    laplacian_vars = []
    for i, file in enumerate(files, start=1):
        img = load_gray_image(file)  # 224x224 (reference)
        if img is not None:
            lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
            laplacian_vars.append(lap_var)

        if i % PRINT_EVERY == 0 or i == n_sample:
            print(f"    processed {i}/{n_sample}")

    if len(laplacian_vars) == 0:
        return 10.0

    return float(np.percentile(laplacian_vars, percentile))


# ============================
# 4) Blur Detection (REFERENCE STYLE)
# ============================
def is_blurry(img_224, threshold):
    laplacian_var = cv2.Laplacian(img_224, cv2.CV_64F).var()
    return laplacian_var < threshold


# ============================
# 5) Under-/Overexposure (REFERENCE STYLE)
# ============================
def bad_exposure(img_224, min_mean=40, max_mean=220):
    mean_intensity = np.mean(img_224)
    return mean_intensity < min_mean or mean_intensity > max_mean


# ============================
# 6) Low Edge Content (REFERENCE STYLE)
# ============================
def low_edge_content(img_224, threshold=0.01):
    edges = cv2.Canny(img_224, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return edge_density < threshold


# ============================
# 7) Mask quality checks (simple + safe)
# ============================
def mask_too_empty(mask_224, min_foreground_ratio=0.002):
    fg_ratio = np.sum(mask_224 > 0) / mask_224.size
    return fg_ratio < min_foreground_ratio


def mask_bbox_too_small(mask_224, min_w=10, min_h=10):
    ys, xs = np.where(mask_224 > 0)
    if len(xs) == 0:
        return True
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    return w < min_w or h < min_h


# ============================
# 8) Combined Filter (Image + Mask)
# ============================
def is_usable_pair(img_224, mask_224, blur_threshold):
    if img_224 is None or mask_224 is None:
        return False

    # image checks (reference logic)
    if is_blurry(img_224, blur_threshold):
        return False
    if bad_exposure(img_224):
        return False
    if low_edge_content(img_224):
        return False

    # mask checks
    if mask_too_empty(mask_224):
        return False
    if mask_bbox_too_small(mask_224):
        return False

    return True


# ============================
# 9) Cropping using mask bbox (224 space)
# ============================
def compute_crop_box_from_mask(mask_224, margin=CROP_MARGIN):
    ys, xs = np.where(mask_224 > 0)
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(mask_224.shape[1], x2 + margin)
    y2 = min(mask_224.shape[0], y2 + margin)

    return (x1, y1, x2, y2)


def crop_image_and_mask(img_224, mask_224, crop_box):
    x1, y1, x2, y2 = crop_box
    img_crop = img_224[y1:y2, x1:x2]
    mask_crop = mask_224[y1:y2, x1:x2]

    # keep final size consistent
    img_crop = cv2.resize(img_crop, FINAL_SIZE)
    mask_crop = cv2.resize(mask_crop, FINAL_SIZE, interpolation=cv2.INTER_NEAREST)
    return img_crop, mask_crop


# ============================
# 10) Filter + Crop + Save (pairs)
# ============================
def filter_crop_and_save_pairs(images_dir, masks_dir, out_images_dir, out_masks_dir, blur_threshold):
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    image_files = list_image_files_fast(images_dir)

    saved = 0
    skipped = 0
    skipped_missing_mask = 0

    for i, img_path in enumerate(image_files, start=1):
        fname = os.path.basename(img_path)
        mask_path = os.path.join(masks_dir, fname)

        if not os.path.exists(mask_path):
            skipped_missing_mask += 1
            continue

        img_224 = load_gray_image(img_path)
        mask_224 = load_mask(mask_path)

        if not is_usable_pair(img_224, mask_224, blur_threshold):
            skipped += 1
            continue

        crop_box = compute_crop_box_from_mask(mask_224)
        if crop_box is None:
            skipped += 1
            continue

        img_crop, mask_crop = crop_image_and_mask(img_224, mask_224, crop_box)

        cv2.imwrite(os.path.join(out_images_dir, fname), img_crop)
        cv2.imwrite(os.path.join(out_masks_dir, fname), mask_crop)
        saved += 1

        if i % 500 == 0:
            print(f"  processed {i}/{len(image_files)} | saved {saved}")

    print("\n========== SUMMARY ==========")
    print(f"Images dir: {images_dir}")
    print(f"Saved: {saved}")
    print(f"Skipped (missing mask): {skipped_missing_mask}")
    print(f"Skipped (failed checks): {skipped}")
    print("=============================\n")


# ============================
# 11) RUN on dataset
# ============================
raw_base = r"D:\Data_Sceience_AI_ML\Data_Scientest\Projects\My_Project\Local\data\raw_2\COVID-19_Radiography_Dataset"
clean_base = r"D:\Data_Sceience_AI_ML\Data_Scientest\Projects\My_Project\Local\data\processed"

for subfolder in os.listdir(raw_base):
    class_dir = os.path.join(raw_base, subfolder)
    if not os.path.isdir(class_dir):
        continue

    images_dir = os.path.join(class_dir, "images")
    masks_dir = os.path.join(class_dir, "masks")

    if not (os.path.isdir(images_dir) and os.path.isdir(masks_dir)):
        print(f"Skipping '{subfolder}' (missing images/ or masks/)")
        continue

    # Determine class group
    if subfolder.lower() == "covid":
        blur_percentile = 5
        out_class = "covid"
    else:
        blur_percentile = 10
        out_class = "non_covid"

    out_images_dir = os.path.join(clean_base, out_class, "images")
    out_masks_dir = os.path.join(clean_base, out_class, "masks")

    print(f"\n[{subfolder}] Step 1: Calculate blur threshold (reference style, 224x224) with sampling frac={FRAC}")
    blur_threshold = calculate_blur_threshold_sampled(images_dir, percentile=blur_percentile)
    print(f"[{subfolder}] Using blur threshold = {blur_threshold:.2f}")

    print(f"[{subfolder}] Step 2: Filter + crop + save")
    filter_crop_and_save_pairs(
        images_dir=images_dir,
        masks_dir=masks_dir,
        out_images_dir=out_images_dir,
        out_masks_dir=out_masks_dir,
        blur_threshold=blur_threshold
    )
