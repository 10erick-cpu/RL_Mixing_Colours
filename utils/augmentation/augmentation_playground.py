import glob
import os
import random

import Augmentor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from natsort import natsorted

OUT_FOLDER = "./test/out"
IN_FOLDER_IMGS = "./test/in/imgs/"
IN_FOLDER_MASKS = "./test/in/masks"
IN_FOLDER_PROD = "./test/in/"


if __name__ == "__main__":

    img_path = os.path.join(IN_FOLDER_PROD, "imgs/*.png")
    mask_path = os.path.join(IN_FOLDER_PROD, "masks/*.png")
    print(img_path, mask_path)
    # Reading and sorting the image paths from the directories
    ground_truth_images = natsorted(glob.glob(img_path))
    segmentation_mask_images = natsorted(glob.glob(mask_path))
    print(len(ground_truth_images), len(segmentation_mask_images))
    for i in range(0, len(ground_truth_images)):
        print("%s: Ground: %s | Mask 1: %s " %
              (i + 1, os.path.basename(ground_truth_images[i]),
               os.path.basename(segmentation_mask_images[i])))

    collated_images_and_masks = list(zip(ground_truth_images,
                                         segmentation_mask_images))

    images = [[np.asarray(Image.open(y)) for y in x] for x in collated_images_and_masks]

    p = Augmentor.DataPipeline(images)

    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    augmented_images = p.sample(10)
    r_index = random.randint(0, len(augmented_images) - 1)
    f, axarr = plt.subplots(1, 3, figsize=(20, 15))

    axarr[0].imshow(images[0][0])
    axarr[1].imshow(augmented_images[r_index][0])
    axarr[2].imshow(augmented_images[r_index][1], cmap="gray")
    plt.show()
