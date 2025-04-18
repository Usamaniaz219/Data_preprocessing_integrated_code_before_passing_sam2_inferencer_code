import os
import cv2

from testing.scripts.mean_shift_noise_removal import remove_noise


# import time
#
# start_time = time.time()

# Define your padding and tiling functions
def pad_image_to_tile_size(image, tile_size):
    h, w = image.shape[:2]
    pad_h = (tile_size - h % tile_size) if h % tile_size != 0 else 0
    pad_w = (tile_size - w % tile_size) if w % tile_size != 0 else 0
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def split_image_into_tiles(image, tile_size, pad_size):
    h, w = image.shape[:2]
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            print(tile.shape)
            top_pad = (pad_size - tile.shape[0]) // 2
            bottom_pad = pad_size - tile.shape[0] - top_pad
            left_pad = (pad_size - tile.shape[1]) // 2
            right_pad = pad_size - tile.shape[1] - left_pad
            padded_tile = cv2.copyMakeBorder(tile, top_pad, bottom_pad, left_pad, right_pad, 
                                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
            print(padded_tile.shape)
            tiles.append(padded_tile)
    return tiles



# Main processing function
def process_mask_directory(image_path, tile_dir, no_noise_dir):
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading mask: {image_path}")
        return

    # Process the mask image to get tiles
    tile_size = 800
    pad_size = 1024
    padded_image = pad_image_to_tile_size(image, tile_size)
    tiles = split_image_into_tiles(padded_image, tile_size, pad_size)

    # Save each tile
    for idx, tile in enumerate(tiles):
        if cv2.countNonZero(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)) == 0:
            continue
        tile_name = f"{os.path.splitext(image_name)[0]}_tile_{idx}.jpg"
        tile_output_path = os.path.join(tile_dir, tile_name)
        cv2.imwrite(tile_output_path, tile)
        if no_noise_dir !="NAN":
            remove_noise(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY), tile_name,no_noise_dir)
        # print(f"Tile {idx} saved at: {tile_output_path}")


if __name__ == "__main__":
    root_mask_dir = '/home/asfand/Work/sam2/testing/data/complete_test/wa_darrington/wa_darrington.jpg'  # Root directory containing subdirectories of masks
    output_dir = '/home/asfand/Work/sam2/testing/data/complete_test/wa_darrington//masks_tiles/'  # Output directory for saving the tiles
    process_mask_directory(root_mask_dir, output_dir)

# end_time = time.time()
#
# print("Time taken " + str(end_time-start_time))