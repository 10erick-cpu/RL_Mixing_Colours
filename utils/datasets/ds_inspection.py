import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from datasets.dataset_info import JsonDatasetInfo
from datasets.multi_channel_dataset import SingleChannelDataset, ThreeChannelDataset
from helper_functions.img_utils import draw_fluorescence_overlays, remove_blobs
from models.filesystem_utils import Folder

KEY_NEXT = ord('k')
KEY_PREV = ord('j')
KEY_CANCEL = ord('c')
KEY_REMOVE_BLOB = ord('q')
KEY_DEL = 8


def show_and_wait_for_action(input_image, display_scale):
    cv2.imshow("",
               cv2.resize(input_image, (int(overlay.shape[1] * display_scale), int(overlay.shape[0] * display_scale))))

    key = cv2.waitKeyEx()
    return key


def move_files_deleted(del_dir, img_file, mask_files):
    del_dir.make_dir()
    del_dir_mask = del_dir.make_sub_folder("del_masks")
    del_dir_mask.make_dir()
    if isinstance(img_file, str):
        img_file = [img_file]

    for f in img_file:
        file_name = os.path.split(f)[1]
        os.rename(f, del_dir.get_file_path(file_name))

    for f in mask_files:
        file_name = os.path.split(f)[1]
        os.rename(f, del_dir_mask.get_file_path(file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify fluorescence input')
    parser.add_argument('folder')
    args = parser.parse_args()
    root_folder = Folder(args.folder)
    infos = JsonDatasetInfo.init_from_root_dir(args.folder)
    if len(infos) == 0:
        print("No dataset info files found in folder", args.folder)

    for ds_info in infos:
        print(ds_info)

        output_shape = (1024, 1344)
        if ds_info.image_types is not None and len(ds_info.image_types) > 1:

            dataset = ThreeChannelDataset.from_json_config(ds_info, output_shape, reduce_masks=False)
        else:
            dataset = SingleChannelDataset.from_json_config(ds_info, output_shape, None, reduce_masks=True)

        dataset.set_augmentation_enabled(False)

        item_idx = 0
        max_items = len(dataset)
        del_items_idx = []
        while 0 <= item_idx < max_items:
            img, mask = dataset[item_idx]
            img = img.numpy()
            if img.shape[0] == 1:
                img = img[0, :, :]
            else:
                img = np.transpose(img, (1, 2, 0))
                img = img[:, :, 0]
            img = (img * 255).astype(np.uint8)
            mask = mask.numpy()
            if len(mask.shape) == 2:
                mask_prim = (mask * 255).astype(np.uint8)
                mask_sec = None
            else:
                mask_prim = (mask[0, :, :] * 255).astype(np.uint8)
                mask_sec = (mask[1, :, :] * 255).astype(np.uint8)

            overlay = draw_fluorescence_overlays(img, green=mask_prim, red=mask_sec, overlay_weight=0.2)
            action = show_and_wait_for_action(overlay, 0.8)

            if action == KEY_CANCEL:
                break
            elif action == KEY_NEXT:
                item_idx += 1
                while item_idx in del_items_idx:
                    item_idx += 1
            elif action == KEY_PREV:
                item_idx -= 1 if item_idx - 1 >= 0 else 0
                while item_idx in del_items_idx:
                    item_idx -= 1
            elif action == KEY_DEL:
                img_info = dataset._get_image_info(item_idx)
                mask_info = dataset._get_mask_info(item_idx)
                print(img_info)
                print(mask_info)
                move_files_deleted(root_folder.make_sub_folder("0_removed"), img_info, mask_info)
                del_items_idx.append(item_idx)
                item_idx += 1
            elif action == KEY_REMOVE_BLOB:
                test = remove_blobs(mask[0, :, :].astype(np.uint8), 150)
                plt.imshow(test * 255, cmap="gray")
                plt.imshow(mask[0, :, :].astype(np.uint8), alpha=0.3)
                plt.show()
