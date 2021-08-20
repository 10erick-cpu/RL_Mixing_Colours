import argparse
import os

import cv2
import numpy as np
from PIL import Image
from helper_functions.img_utils import image_to_8bit, draw_fluorescence_overlays, apply_CLAHE_with_grayscale_image, \
    create_threshold
from models.filesystem_utils import Folder

EXT_DIC = "--DIC"
EXT_HOECHST = "--Hoechst"
EXT_TXRED = "--TxRed"
PRIMARY_EXTS = ['--Hoechst', '--Calcein']
SECONDARY_EXTS = ['--TxOrange', '--TxRed', '--Propidium', '--Blank', '--GFP']
EXT_CT_ORANGE = "--CellTracker orange"


def detect_extension(file, ext_set):
    result = [ext for ext in ext_set if ext in file]

    if len(result) == 0:
        return None

    if len(result) > 1:
        print(result)
        raise ValueError("Multiple extensions for file {} detected".format(file))
    return result[0]


def detect_primary_secondary(file_list):
    primary_dist = set()
    second_dist = set()
    for file in file_list:
        if EXT_DIC in file:
            continue
        prim = detect_extension(file, PRIMARY_EXTS)
        if prim and prim not in primary_dist:
            primary_dist.add(prim)

        second = detect_extension(file, SECONDARY_EXTS)
        if second and second not in second_dist:
            second_dist.add(second)

    print("Extensions detected: ")
    print("primary", primary_dist)
    print("secondary", second_dist)
    if len(primary_dist) == 1 and len(second_dist) == 1:
        return list(primary_dist)[0], list(second_dist)[0]
    elif len(primary_dist) >= 1 and len(second_dist) >= 1:
        print("Warn: multiple extensions detected using random")
        return list(primary_dist)[0], list(second_dist)[0]
    else:

        raise ValueError("Extension detection failed")


def remove_imageset(source_folder, rm_folder, img_set):
    for im_dict in img_set:
        source_file = im_dict['path']

        source_path = source_folder.get_file_path(source_file)
        target_path = rm_folder.get_file_path(source_file)
        print("Moving {} to {}\n".format(source_path, target_path))
        os.rename(source_path, target_path)


def show_and_wait_for_action(input_image):
    cv2.imshow("", cv2.resize(input_image, display_shape))

    key = cv2.waitKeyEx()
    return key


def apply_clahe(imgs, blocksize=128, max_slope=1):
    result = []
    for im in imgs:
        result.append(apply_CLAHE_with_grayscale_image(im, blocksize=blocksize, max_slope=max_slope))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify fluorescence input')
    parser.add_argument('folder')
    args = parser.parse_args()

    data_folder = Folder(args.folder)

    remove_folder = data_folder.make_sub_folder("00_removed")
    remove_folder.make_dir()

    show_image_overlay = False

    files = sorted(data_folder.get_files(abs_path=False))
    if ".DS_Store" in files:
        files.remove(".DS_Store")

    if len(files) % 3 != 0:
        print("suspicious file count: ", len(files))

    dics = list(filter(lambda x: EXT_DIC in x, files))

    if len(files) != len(dics) * 3:
        print("suspicious number of DIC images (# files: {} | dic images: {}, expected count: {})".format(len(files), len(dics),
                                                                                                          len(dics) * 3))

    primary_ext, secondary_ext = detect_primary_secondary(files)

    print("Detected primary / secondary images: {} / {}".format(primary_ext, secondary_ext))

    expected_files = [(dic, dic.replace(EXT_DIC, primary_ext), dic.replace(EXT_DIC, secondary_ext)) for dic in dics]

    error = []

    for dic_fn, h_fn, tx_fn in expected_files:
        if not data_folder.exists_path_in_folder(h_fn):
            error.append(("File does not exist", h_fn))

        if not data_folder.exists_path_in_folder(tx_fn):
            error.append(("File does not exist", tx_fn))
            if not data_folder.exists_path_in_folder(dic_fn):
                error.append(("File does not exist", dic_fn))

        if h_fn in files:
            files.remove(h_fn)
        if tx_fn in files:
            files.remove(tx_fn)
        if dic_fn in files:
            files.remove(dic_fn)

        imgs = [{'image': np.asarray(Image.open(data_folder.get_file_path(path))), 'path': path} for path in [dic_fn, h_fn, tx_fn]]

        if not imgs[0]['image'].shape == imgs[1]['image'].shape == imgs[2]['image'].shape:
            error.append(("Images do not have the same shape: bf: {}, hst: {}, tx: {}".format(str(imgs[0].shape), str(imgs[1].shape),
                                                                                              str(imgs[2].shape)), dic_fn, h_fn, tx_fn))
        else:
            if show_image_overlay:
                if imgs[0]['image'].dtype == np.uint16:
                    for im_dict in imgs:
                        im_dict['image'] = image_to_8bit(im_dict['image'])
                dic = imgs[0]['image']
                primary = imgs[1]['image']
                secondary = imgs[2]['image']

                overlay = draw_fluorescence_overlays(dic, green=primary, red=secondary)
                scale = 0.8
                display_shape = (int(overlay.shape[1] * scale), int(overlay.shape[0] * scale))

                print("Displaying images ")
                print("ch gray:", imgs[0]['path'])
                print("ch g:", imgs[1]['path'])
                print("ch r:", imgs[2]['path'])

                key = show_and_wait_for_action(overlay)
                print(key)

                if key == ord('l'):
                    dic, primary, secondary = apply_clahe([dic, primary, secondary])
                    overlay = draw_fluorescence_overlays(dic, green=primary, red=secondary)
                    key = show_and_wait_for_action(overlay)

                if key == ord('t'):
                    prim_th, sec_th = create_threshold([primary, secondary],513, -0.0019)
                    mask = np.bitwise_or(prim_th, sec_th)
                    dic_masked = dic.copy()
                    dic_masked[mask == 0] = 0
                    overlay = draw_fluorescence_overlays(dic_masked, green=prim_th, red=sec_th, overlay_weight=0.1)
                    key = show_and_wait_for_action(overlay)
                    toggle_th = True
                    while key == ord('t'):
                        if toggle_th:
                            overlay = draw_fluorescence_overlays(dic, green=primary, red=secondary)
                            toggle_th = False
                        else:
                            overlay = draw_fluorescence_overlays(dic_masked, green=prim_th, red=sec_th, overlay_weight=0.2)
                            toggle_th = True
                        key = show_and_wait_for_action(overlay)

                if key == ord('f'):
                    show_image_overlay = False
                elif key == ord('c'):
                    break
                elif key == 8:
                    # backspace, delete
                    print("Moving images to", remove_folder)
                    remove_imageset(data_folder, remove_folder, imgs)

    cv2.destroyAllWindows()
    print()
    print("{} errors detected".format(len(error)))
    if len(error) > 0:
        for err in error:
            print()
            print(err)
    if len(files) > 0:
        print("{} orphan files detected".format(len(files)))
        print(files)

    print("{} image sets in total".format(len(expected_files)))
