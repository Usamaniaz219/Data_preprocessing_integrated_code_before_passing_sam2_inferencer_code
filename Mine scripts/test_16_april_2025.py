
    
import os
import shutil

def extract_tiles(source_root, dest_root):
    """
    Extracts 'image_tiles' and 'mask_tiles' from 'step6_outputs' directories 
    under each result folder and copies them into the destination directory 
    with renamed folders to avoid conflicts.

    Args:
        source_root (str): Path to the root directory containing result folders.
        dest_root (str): Path to the directory where extracted tiles should be saved.
    """
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)

    # Loop through each 'result*' directory
    for folder in os.listdir(source_root):
        result_path = os.path.join(source_root, folder)
        step6_path = os.path.join(result_path, "outputs", "step_6_outputs")
        
        if os.path.isdir(step6_path):
            for sub_dir in ["image_tiles_", "mask_tiles_"]:
                src = os.path.join(step6_path, sub_dir)
                if os.path.exists(src):
                    # Destination directory with unique name
                    dst = os.path.join(dest_root, f"{folder}_{sub_dir}")
                    shutil.copytree(src, dst)
                    print(f"Copied {src} to {dst}")
                else:
                    print(f"{sub_dir} not found in {step6_path}")


source_merged_directory = "/media/usama/42F84FDFF84FCFB9/usama-dev/Meanshift_for_Zone_Segmentation/denoise_merged_mask_tiles_deom89/"
source_directory = "/media/usama/42F84FDFF84FCFB9/usama-dev/Meanshift_for_Zone_Segmentation/denoise_mask_tiles_demo89/"
destination_merged_directory = "Extracted_tiles_merged_demo89"
destination_directory = "Extracted_denoise_tiles_demo89"
extract_tiles(source_merged_directory, destination_merged_directory)
extract_tiles(source_directory, destination_directory)  # For comparison







# def load_image_and_points(image_path, merged_mask_path, denoise_mask_path):
#     # Dummy placeholder function. Replace with your actual implementation.
#     print(f"Processing: \nImage: {image_path}\nMerged Mask: {merged_mask_path}\nDenoise Mask: {denoise_mask_path}\n")

# def process_all_tiles(denoise_root, merged_root):
#     # Traverse subdirectories in the denoise root
#     for subfolder in os.listdir(denoise_root):
#         if subfolder.endswith('_image_tiles_'):
#             base_name = subfolder.replace('_image_tiles_', '')
#             denoise_image_dir = os.path.join(denoise_root, subfolder)
#             denoise_mask_dir = os.path.join(denoise_root, f"{base_name}_mask_tiles_")
#             merged_image_dir = os.path.join(merged_root, subfolder)
#             merged_mask_dir = os.path.join(merged_root, f"{base_name}_mask_tiles_")

#             if not os.path.isdir(denoise_mask_dir) or not os.path.isdir(merged_mask_dir):
#                 print(f"Skipping {base_name} due to missing mask directory.")
#                 continue

#             for image_file in os.listdir(merged_image_dir):
#                 if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     image_path = os.path.join(denoise_image_dir, image_file)
#                     merged_mask_path = os.path.join(merged_mask_dir, image_file)
#                     denoise_mask_path = os.path.join(denoise_mask_dir, image_file)

#                     if os.path.exists(merged_mask_path) and os.path.exists(denoise_mask_path):
#                         load_image_and_points(image_path, merged_mask_path, denoise_mask_path)
#                     else:
#                         print(f"Missing mask for: {image_file}")


# denoise_root_dir = 'Extracted_denoise_tiles'
# merged_root_dir = 'Extracted_tiles_merged'

# process_all_tiles(denoise_root_dir, merged_root_dir)

















