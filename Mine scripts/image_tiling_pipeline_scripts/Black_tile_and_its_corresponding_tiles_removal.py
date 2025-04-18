import os
import cv2
import re
import numpy as np

def is_black_mask(mask_path, pixel_threshold=10, contour_area_threshold=350):
    """
    Check if a mask should be considered black based on foreground pixel count and contour area.
    A mask is considered black if the number of nonzero pixels is <= pixel_threshold
    or if all detected contours have an area <= contour_area_threshold.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Unable to load mask {mask_path}")
        return False
    
    # Check foreground pixel count
    if np.count_nonzero(mask) <= pixel_threshold:
        return True
    
    # Find contours and check contour area
    _,mask = cv2.threshold(mask,128,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > contour_area_threshold:
            return False
    
    return True

def clean_up_images_and_masks(image_dir, mask_dir):
    # Step 1: Get the list of subdirectories
    image_subfolders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
    mask_subfolders = [f for f in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, f))]

    # Step 2: Process each corresponding subdirectory
    for subfolder in image_subfolders:
        image_subfolder_path = os.path.join(image_dir, subfolder)
        mask_subfolder_path = os.path.join(mask_dir, subfolder)

        # Check if the corresponding mask subfolder exists
        if not os.path.exists(mask_subfolder_path):
            print(f"Error: Missing corresponding mask subfolder for {subfolder}")
            continue

        # Step 3: Iterate through files in the image subfolder
        for image_file in os.listdir(image_subfolder_path):
            image_file_path = os.path.join(image_subfolder_path, image_file)

            # Extract the numerical ID from the image filename
            image_id_match = re.search(r'_(\d+)\.jpg$', image_file)
            if not image_id_match:
                print(f"Warning: Skipping file with unexpected name format: {image_file}")
                continue
            image_id = image_id_match.group(1)

            # Construct the expected mask filename
            mask_file = f"{subfolder}_tile_{image_id}.jpg"
            mask_file_path = os.path.join(mask_subfolder_path, mask_file)

            # Check if the mask file exists
            if not os.path.exists(mask_file_path):
                print(f"Error: Missing corresponding mask file for {image_file}")
                continue

            # Step 4: Check if the mask should be considered black based on pixel count and contour area
            if is_black_mask(mask_file_path):
                # If the mask is considered black, delete both mask and corresponding image tile
                os.remove(mask_file_path)
                os.remove(image_file_path)
                print(f"Deleted black mask and corresponding image: {mask_file} and {image_file}")
