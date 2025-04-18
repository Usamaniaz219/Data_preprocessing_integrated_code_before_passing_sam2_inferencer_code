import cv2
import os
import time
import numpy as np

def remove_noise(mask_img, filename,output_dir):
    # cv2.imshow('mask', mask)
    dilation= cv2.dilate(mask_img, np.ones((5, 5), np.uint8), iterations=2)
    # cv2.imshow('dilate', dilation)

    erosion= cv2.erode(dilation, np.ones((5, 5), np.uint8), iterations=5)
    # cv2.imshow('erosion', erosion)

    dilation2= cv2.dilate(erosion, np.ones((5, 5), np.uint8), iterations=3)
    # cv2.imshow('dilate2', dilation2)
    _, binary_mask = cv2.threshold(dilation2, 20, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(binary_mask) == 0:
        return
    cv2.imwrite(os.path.join(output_dir, filename), binary_mask)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 700, 700)
    # cv2.imshow('image', dilation2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    test_img = cv2.imread('/home/asfand/Work/sam2/testing/data/complete_test/wa_darrington/mean_tiles_dir/wa_darrington_3/wa_darrington_3_tile_13.jpg')
    remove_noise(test_img, 'test','.')