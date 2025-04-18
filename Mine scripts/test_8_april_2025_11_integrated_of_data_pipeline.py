
import image_tiling_pipeline_scripts.tiling_the_image_into_800_tiles_and_then_padded_to_make_it_1000
import image_tiling_pipeline_scripts.tiling_each_mask
import image_tiling_pipeline_scripts.copied_the_images_tiles_directory
import image_tiling_pipeline_scripts.Black_tile_and_its_corresponding_tiles_removal
import image_tiling_pipeline_scripts.renamed_the_mask_tiles_w_r_t_image_tiles
import image_tiling_pipeline_scripts.Make_a_paired_image_and_mask_tiles

from image_tiling_pipeline_scripts.tiling_the_image_into_800_tiles_and_then_padded_to_make_it_1000 import process_and_save_tiles, process_image
from image_tiling_pipeline_scripts.tiling_each_mask import process_mask_directory
from image_tiling_pipeline_scripts.copied_the_images_tiles_directory import duplicate_images_for_masks
from image_tiling_pipeline_scripts.Black_tile_and_its_corresponding_tiles_removal import clean_up_images_and_masks
from image_tiling_pipeline_scripts.renamed_the_mask_tiles_w_r_t_image_tiles import rename_images_and_masks
from image_tiling_pipeline_scripts.Make_a_paired_image_and_mask_tiles import copy_image_and_mask_tiles

import os
import cv2
import numpy as np
import os
import time
import logging
from sklearn.cluster import MeanShift
import shutil


logging.basicConfig(filename='im_process2.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Set up logging configuration 
def MeanShift_Zoning_Segmenter(image, output_subdir):
    pixels = image.reshape((-1, 3))
    # clustering = MeanShift(bandwidth=10, n_jobs=-1, bin_seeding=True, min_bin_freq=1, cluster_all=False).fit(pixels)
    clustering = MeanShift(bandwidth=8, bin_seeding=True).fit(pixels)
    # clustering = MeanShift(bandwidth=15, n_jobs=-1).fit(pixels)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    for label in unique_labels:  # Extract areas of interest based on unique labels using logical AND operation
        label_mask = (labels == label).reshape(image.shape[:2]).astype(np.uint8)
        area_of_interest = cv2.bitwise_and(image, image, mask=label_mask * 255)
        mask_name = f"{os.path.splitext(os.path.basename(output_subdir))[0]}_{label}.jpg"   # Output image naming includes the original image name and clustering label
        output_directory_path = os.path.join(output_subdir, mask_name)
        cv2.imwrite(output_directory_path, area_of_interest)

def process_image(image_path, output_dir):
    try:
       
        image = cv2.imread(image_path)   # Read the image
        start_time = time.time()       # Measure processing time
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)    # Resize and convert to RGB
        # Resized image using bicubic interpolation
        # image = cv2.resize(image, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print("image name", image_name)
        output_subdir = os.path.join(output_dir, image_name)
        os.makedirs(output_subdir, exist_ok=True)
        MeanShift_Zoning_Segmenter(image, output_subdir) # Apply the MeanShift_Zoning_Segmenter function
        logging.info(f"Processed image '{image_path}' - Resolution: {image.shape[1]}x{image.shape[0]}, Processing Time: {time.time() - start_time:.4f} seconds")  # Log image resolution and processing time

        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
        
    except Exception as e:
        logging.error(f"Error processing image '{image_path}': {str(e)}")



def find_mask_with_suffix(directory, suffix):
    """
    Find the first file ending with the given suffix in the directory.
    """
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            return os.path.join(directory, filename)
    return None

def apply_canny_edge_detector(image_path, output_path):
    """
    Step 1: Perform adaptive thresholding to detect edges.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image,(5,5),0)
    edges = cv2.Canny(image,100,200)

    # Apply adaptive threshold for edge detection
    # edges = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv2.THRESH_BINARY_INV, 21, 7)

    cv2.imwrite(output_path, edges)
    return edges

def combine_mask_with_edges(mask_path, edges):
    """
    Step 2: Combine zoning mask with detected edges.
    
    """
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure binary mask
    _, mask_image = cv2.threshold(mask_image, 50, 255, cv2.THRESH_BINARY)

    # Combine mask and edges using bitwise OR
    combined_mask = cv2.bitwise_or(mask_image, edges)

    # Morphological closing to remove small gaps
    kernel = np.ones((3, 3), np.uint8)
    # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # combined_mask = cv2.dilate(combined_mask,kernel,iterations=1)

    return combined_mask

def subtract_and_denoise(target_mask_path, combined_mask):
    """
    Step 3: Subtract edges from target cluster mask to denoise.
    """
    target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure binary mask
    _, target_mask = cv2.threshold(target_mask, 50, 255, cv2.THRESH_BINARY)
    # Subtract edges from the mask
    denoised_mask = cv2.subtract(target_mask, combined_mask)

    # Apply median blur for smoothing
    denoised_mask = cv2.medianBlur(denoised_mask, 5)

    return denoised_mask

def process_masks(image_path,ms_org_mask_directory,denoised_output_dir):
    """
    Main processing loop: find _0.jpg mask, then process all other masks.
    """

    # Ensure output directory exists
    # denoised_output_dir = os.path.join(mask_directory, "denoised_masks")
    # denoised_output_dir = os.path.join(mask_directory, "denoised_masks")
    # os.makedirs(denoised_output_dir, exist_ok=True)

    # Find the base _0 mask (this mask remains unchanged and used for all others)
    base_mask_path = find_mask_with_suffix(ms_org_mask_directory, "_0.jpg")
    if base_mask_path is None:
        print("No _0.jpg mask found in directory.")
        return

    print(f"Using base mask: {base_mask_path}")

    # Detect edges once from the main image
    # edge_output = os.path.join(mask_directory, "fl_edges.jpg")
    edge_output = "fl_edges.jpg"
    edges = apply_canny_edge_detector(image_path, edge_output)

    # Combine base mask with edges (this combined mask will be used for all images)
    combined_mask = combine_mask_with_edges(base_mask_path, edges)

    # Process each mask file (skip _0.jpg itself)
    for filename in os.listdir(ms_org_mask_directory):
        if filename.endswith(".jpg") and not filename.endswith("_0.jpg"):
            target_mask_path = os.path.join(ms_org_mask_directory, filename)

            # Apply denoising process
            denoised_mask = subtract_and_denoise(target_mask_path, combined_mask)
            kernel = np.ones((3,3),np.uint8)
            # Save denoised mask with the same name in "denoised_masks" folder
            denoised_output_path = os.path.join(denoised_output_dir, filename)
            # denoised_mask = cv2.erode(denoised_mask,kernel,iterations=1)
            cv2.imwrite(denoised_output_path, denoised_mask)
            print(f"Denoised mask saved: {denoised_output_path}")

def copy_dirs(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        print(f"Destination directory '{dst_dir}' already exists. Removing it.")
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    print(f"Copied directory from '{src_dir}' to '{dst_dir}'")
    # return dst_dir


def merge_mask_polygons(mask_dir):
    """
    Applies a sequence of morphological operations (dilation and erosion) 
    to all image files in the specified directory.

    Args:
        mask_dir (str): Path to the directory containing the mask images.
    """

    # List all image files in the directory
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Define the kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)

        # Read the image in grayscale
        image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            # Morphological operations: dilate -> erode -> dilate
            dilation = cv2.dilate(image, kernel, iterations=2)
            erosion = cv2.erode(dilation, kernel, iterations=5)
            final_output = cv2.dilate(erosion, kernel, iterations=3)

            # Save the processed image back to the same path
            cv2.imwrite(mask_path, final_output)
        else:
            print(f"Failed to read the image: {mask_file}")


def create_directory_structure(image_path, mask_dir,base_dir):
    # Extract image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Root directory
    # root_image_dir = image_name
    # base_dir = "/media/usama/42F84FDFF84FCFB9/usama-dev/Meanshift_for_Zone_Segmentation/denoise_mask_tiles/"
    root_image_dir  = os.path.join(base_dir,image_name)

    images_dir = os.path.join(root_image_dir, "images")
    masks_sub_dir = os.path.join(root_image_dir, "masks", image_name)
    outputs_dir = os.path.join(root_image_dir, "outputs")

    # mask_dir_1 = os.path.join(root_image_dir,"masks")

    # Create directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_sub_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # Copy the image
    shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))

    # Copy masks
    for mask in os.listdir(mask_dir):
        if mask == image_name:
            mask_paths = os.path.join(mask_dir,mask)
            for mask_file in os.listdir(mask_paths):
                mask_file_path = os.path.join(mask_paths,mask_file)
                shutil.copy(mask_file_path, masks_sub_dir)

  
    print(f"Directory structure created for '{image_name}'")

    return root_image_dir,outputs_dir



def process_full_pipeline(image_dir, output_base_dir):
    # Step 0: Create the directory structure
    # create_directory_structure(image_path, mask_dir)

    image_file_name = os.path.splitext(os.path.basename(image_dir))[0]
    # input_base_dir = os.path.join(os.getcwd(), image_file_name)  # or full path if needed

    # input_ori_image_dir = os.path.join(input_base_dir, "images")
    input_ori_image_dir = os.path.join(image_dir, "images")
    # mask_files_dir = os.path.join(input_base_dir, "masks")
    mask_files_dir = os.path.join(image_dir, "masks")
    step_1_output = os.path.join(output_base_dir, "step_1_outputs")
    step_2_output = os.path.join(output_base_dir, "step_2_outputs")
    step_3_output = os.path.join(output_base_dir,  "step_3_outputs")
    target_image_dir = os.path.join(output_base_dir,  "step_6_outputs/image_tiles_")
    target_mask_dir = os.path.join(output_base_dir,  "step_6_outputs/mask_tiles_")

    # Make directories
    os.makedirs(step_1_output, exist_ok=True)
    os.makedirs(step_2_output, exist_ok=True)
    os.makedirs(step_3_output, exist_ok=True)
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_mask_dir, exist_ok=True)

    # Pipeline steps
    process_and_save_tiles(input_ori_image_dir, step_1_output, process_image)
    process_mask_directory(mask_files_dir, step_2_output)
    duplicate_images_for_masks(
        os.path.join(step_1_output, f"{image_file_name}_tiles"),
        os.path.join(step_2_output, image_file_name),
        step_3_output
    )
    clean_up_images_and_masks(
        step_3_output,
        os.path.join(step_2_output, image_file_name)
    )
    rename_images_and_masks(
        step_3_output,
        os.path.join(step_2_output, image_file_name)
    )
    copy_image_and_mask_tiles(
        step_3_output,
        os.path.join(step_2_output, image_file_name),
        target_image_dir,
        target_mask_dir
    )




def Convert_all_images_and_masks_into_tiles(org_map_dir, merged_denoise_masks_outputs, merged_denoise_masks_outputs_after_tiling):
    for dir3 in os.listdir(merged_denoise_masks_outputs):
        map_image_path = os.path.join(org_map_dir, f"{dir3}.jpg")
        if os.path.exists(map_image_path):
            images_dir, outputs_dir_merged = create_directory_structure(
                map_image_path,
                merged_denoise_masks_outputs,
                merged_denoise_masks_outputs_after_tiling
            )
            process_full_pipeline(images_dir, outputs_dir_merged)

        

def process_images(org_map_dir, ms_org_masks_directory, denoise_masks_outputs, merged_denoise_masks_outputs,denoise_masks_outputs_after_tiling,merged_denoise_masks_outputs_after_tiling):
    os.makedirs(denoise_masks_outputs, exist_ok=True)

    os.makedirs(merged_denoise_masks_outputs, exist_ok=True)


    for filename in sorted(os.listdir(org_map_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(org_map_dir, filename)
            process_image(image_path, ms_org_masks_directory)

            image_name_wo_ext = os.path.splitext(filename)[0]

            # Loop through each mask subdirectory (e.g., dir1)
            for dir1 in os.listdir(ms_org_masks_directory):
                if dir1 == image_name_wo_ext:
                    dir1_path = os.path.join(ms_org_masks_directory, dir1)

                    if os.path.isdir(dir1_path):
                        # Create unique output directory for each dir1 and image
                        output_dir = os.path.join(denoise_masks_outputs, dir1)
                        os.makedirs(output_dir, exist_ok=True)

                        print(f"Processing: Image = {filename}, Mask Dir = {dir1}")
                        process_masks(image_path, dir1_path, output_dir)
                    
    for dir2 in os.listdir(denoise_masks_outputs):
        dir2_path = os.path.join(denoise_masks_outputs, dir2)
        merged_denoise_masks_outputs_path = os.path.join(merged_denoise_masks_outputs, dir2)
        os.makedirs(merged_denoise_masks_outputs_path, exist_ok=True)
        copy_dirs(dir2_path,merged_denoise_masks_outputs_path)
        dir2_path = os.path.join(denoise_masks_outputs, dir2)
        merge_mask_polygons(merged_denoise_masks_outputs_path)

   

    Convert_all_images_and_masks_into_tiles(org_map_dir, merged_denoise_masks_outputs, merged_denoise_masks_outputs_after_tiling)

    Convert_all_images_and_masks_into_tiles(org_map_dir, denoise_masks_outputs,denoise_masks_outputs_after_tiling)

    return denoise_masks_outputs_after_tiling,merged_denoise_masks_outputs_after_tiling


    
def main():
    
    # image_path = "/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/input__map_images/demo115.jpg"
    input_directory = "/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/demo_89_for_testing/"
    ms_org_masks_directory = "/media/usama/42F84FDFF84FCFB9/usama-dev/Meanshift_for_Zone_Segmentation/Experimented_results/ms_denoised_outputs_43_1/ms_org_outputs_demo89/"

    # Base directory where masks are stored
    denoise_masks_outputs = "/media/usama/42F84FDFF84FCFB9/usama-dev/Meanshift_for_Zone_Segmentation/Experimented_results/ms_denoised_outputs_43_1/ms_denoised_outputs_demo89/"
    merged_denoise_masks_outputs = "/media/usama/42F84FDFF84FCFB9/usama-dev/Meanshift_for_Zone_Segmentation/Experimented_results/ms_denoised_outputs_43_1/ms_merged_denoised_outputs_demo89/"

    denoise_masks_outputs_after_tiling =  "/media/usama/42F84FDFF84FCFB9/usama-dev/Meanshift_for_Zone_Segmentation/denoise_mask_tiles_demo89/"
    merged_denoise_masks_outputs_after_tiling = "/media/usama/42F84FDFF84FCFB9/usama-dev/Meanshift_for_Zone_Segmentation/denoise_merged_mask_tiles_deom89/"

    os.makedirs(denoise_masks_outputs_after_tiling,exist_ok=True)
    os.makedirs(merged_denoise_masks_outputs_after_tiling,exist_ok=True)

    outputs_dir_merged,denoise_outputs_dir = process_images(input_directory, ms_org_masks_directory,denoise_masks_outputs,merged_denoise_masks_outputs,denoise_masks_outputs_after_tiling,merged_denoise_masks_outputs_after_tiling)
    print("outputs dir merged",outputs_dir_merged)
    print("denoise outputs dir",denoise_outputs_dir)

    



if __name__ == "__main__":
    main()
















