import cv2
import os
import numpy as np

# Define folder paths
# Folder containing covid and non_covid images and masks
IMAGE_FOLDER = '/Users/Ice/Repositories/Data Scientest/Project/Cleaned_with_Mask(Filtered_and_Cropped)' 
# Folder to save isolated lung images
OUTPUT_FOLDER = '/Users/Ice/Repositories/Data Scientest/Project/Isolated_lung_images'

categories = ['covid', 'non_covid']

# Loop through covid and non_covid folders
for category in categories:
    img_dir = os.path.join(IMAGE_FOLDER, category, 'images')
    mask_dir = os.path.join(IMAGE_FOLDER, category, 'masks')
    
    # Create the specific covid or non_covid output folder if it does not exist
    category_output = os.path.join(OUTPUT_FOLDER, category)
    if not os.path.exists(category_output):
        os.makedirs(category_output)

    # Loop through images in the category/images folder
    for filename in os.listdir(img_dir):
        if filename.endswith(".png"):
            file_id = filename.split(".")[0]
            
            img_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            # Load x-ray image and mask
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 

            if img is None or mask is None:
                continue

            # Ensure the mask is exactly the same size as the image
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # --- ISOLATE BOTH LUNGS ---
            # This keeps image pixels only where the mask is white (255)
            both_lungs = cv2.bitwise_and(img, img, mask=mask)

            # Save into the category-specific folder
            cv2.imwrite(os.path.join(category_output, f"{file_id}_both.png"), both_lungs)
