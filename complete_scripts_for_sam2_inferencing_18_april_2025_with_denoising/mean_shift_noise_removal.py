import cv2
import os
import time
import numpy as np

def remove_noise_2(mask_directory,output_dir):
    filenames = []
    for filename in os.listdir(mask_directory):
        if filename.endswith(".jpg") and not filename.endswith("_0.jpg"):

            mask_img = cv2.imread(os.path.join(mask_directory, filename), cv2.IMREAD_GRAYSCALE)
            # cv2.imshow('mask', mask)
            dilation= cv2.dilate(mask_img, np.ones((5, 5), np.uint8), iterations=2)
            # cv2.imshow('dilate', dilation)

            erosion= cv2.erode(dilation, np.ones((5, 5), np.uint8), iterations=5)
            # cv2.imshow('erosion', erosion)

            dilation2= cv2.dilate(erosion, np.ones((5, 5), np.uint8), iterations=3)
            # cv2.imshow('dilate2', dilation2)
            _, binary_mask = cv2.threshold(dilation2, 20, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(binary_mask) == 0:
                continue
            cv2.imwrite(os.path.join(output_dir, filename), binary_mask)
            filenames.append(filename)
    return filenames
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 700, 700)
    # cv2.imshow('image', dilation2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    mask_dir = cv2.imread('/home/asfand/Work/sam2/testing/data/complete_test/wa_darrington/mean_tiles_dir/wa_darrington_3/denoised_masks_1')
    out_dir = cv2.imread('/home/asfand/Work/sam2/testing/data/complete_test/wa_darrington/mean_tiles_dir/wa_darrington_3/denoised_masks_2')
    remove_noise_2(mask_dir, out_dir)