import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pandas import Series

from utils.helper_functions.img_utils import load_image, save_image, draw_fluorescence_overlays

from utils.models.dot_dict import DotDict


class ImageSet(object):
    OBJECT_MASK_IDENTIFIER = "obj-mask"
    MULTIMASK_IDENTIFIER = "multimask"

    def __init__(self, src_strategy, grayscale=True, force_8bit=True, lock=False):
        super().__init__()
        self.strategy = src_strategy
        self.channels = DotDict()
        self.masks = DotDict()
        self.images = DotDict()
        self._grayscale = grayscale
        self._8bit = force_8bit
        self.locked = lock

    def __getattr__(self, key):
        if key.startswith("__"):
            return super(ImageSet, self).__getattr__(key)
        key = key.lower()
        if key not in self.channels:
            raise AttributeError(key)
        if key not in self.images:
            self.images[key] = self._load(key)
        return self.images[key]

    def get_mask(self, key, do_cache=True):
        key = key.lower()
        if key not in self.masks:
            raise AttributeError(key)
        if do_cache and key not in self.images:
            self.images[key] = self._load(key, is_mask=True)
        else:
            return self._load(key, is_mask=True)
        return self.images[key]

    def image(self, channel, do_cache=True):
        channel = channel.lower()
        if do_cache:
            return self.__getattr__(channel)
        else:
            return self._load(channel, is_mask=False)

    def pd(self):
        return self.channels

    def _load(self, key, is_mask=False, force_8bit=None):
        if force_8bit is None:
            force_8bit = self._8bit
        if is_mask:
            return np.asarray(Image.open(self.masks[key]['base.file_path'].item()))

        return load_image(self.channels[key]['base.file_path'].item(), force_grayscale=self._grayscale,
                          force_8bit=force_8bit)

    def add_channel(self, chan_key, data):
        chan_key = chan_key.lower()
        if chan_key in self.channels:
            self.check_locked()
        if not self.process_mask_channel(chan_key, data):
            self.channels[chan_key.lower()] = data

    def process_mask_channel(self, chan_key, data):
        if not self._is_mask_id(chan_key):
            return False
        if self.MULTIMASK_IDENTIFIER == chan_key:
            return self.on_add_multimask(chan_key, data)
        if self.OBJECT_MASK_IDENTIFIER in chan_key:
            return self.on_add_object_mask(chan_key, data)
        return False

    def _is_mask_id(self, chan_key):
        return self.MULTIMASK_IDENTIFIER == chan_key or self.OBJECT_MASK_IDENTIFIER in chan_key

    def on_add_object_mask(self, chan_key, data):
        try:
            mask_id = chan_key.split("#")[1]
            self.masks[self.OBJECT_MASK_IDENTIFIER + "#" + mask_id] = data
            return True
        except:
            print("error processing object mask name ", chan_key)
            return False

    def on_add_multimask(self, chan_key, data):
        if self.MULTIMASK_IDENTIFIER in self.channels:
            print("Overriding multimask")
        self.masks[self.MULTIMASK_IDENTIFIER] = data
        return True

    def check_locked(self):
        if self.locked:
            raise ValueError("Imageset is locked")

    def assign_channel(self, chan_key, chan_image, chan_metadata):
        self.check_locked()
        self.channels[chan_key.lower()] = chan_metadata
        self.images[chan_key] = chan_image

    def __repr__(self):
        return "Imageset {} with {} channels: {} and {} masks".format(self.channels['dic']['base.file_path'].item(),
                                                                      len(self.channels), self.channels.keys(),
                                                                      len(self.masks))

    def _save_image(self, image, target_path, dry_run=False):
        if dry_run:
            print(target_path)
        else:
            save_image(target_path, image, self._grayscale)

    def _clone_channel(self, chan_key, is_mask=False):
        if is_mask:
            return self.masks[chan_key].copy(deep=True)

        return self.channels[chan_key].copy(deep=True)

    def set_quality(self, quality):
        for chan_key in self.channels.keys():
            chan_dict = self.channels[chan_key]
            chan_dict['qlvl'] = quality

    def quality(self):
        qlvl = self.channels['dic'].get("qlvl", '0')

        return int(qlvl)

    def persist(self, file_name_strategy, target_folder, make_parent_folder=False, dry_run=False, allow_override=False):
        names = file_name_strategy.get_file_names(self)
        result_set = ImageSet(file_name_strategy)
        for channel_name in self.channels.keys():
            channel = self.channels[channel_name]
            target_dir = target_folder
            if make_parent_folder:
                target_dir = target_folder.make_sub_folder(channel['base.folder_name'].item())
            name = names[channel_name]
            path = target_dir.get_file_path(name)
            chan_image = self.__getattr__(channel_name).copy()
            if os.path.exists(path):
                if allow_override:
                    warnings.warn("Overriding existing file " + path)
                else:
                    raise ValueError("File already exists", path)

            self._save_image(chan_image, path, dry_run)

            meta = self._clone_channel(channel_name)
            file_name_strategy.parse_base_attributes(target_dir.path_abs, name, meta)
            result_set.assign_channel(channel_name, chan_image, meta)

        for mask_item in self.masks.keys():
            channel = self.masks[mask_item]
            target_dir = target_folder
            if make_parent_folder:
                target_dir = target_folder.make_sub_folder(channel['base.folder_name'].item())
            name = names[mask_item]
            path = target_dir.get_file_path(name)
            chan_image = self.get_mask(mask_item).copy()
            if os.path.exists(path):
                warnings.warn("Overriding existing file " + path)

            self._save_image(chan_image, path, dry_run)

            meta = self._clone_channel(mask_item, is_mask=True)
            file_name_strategy.parse_base_attributes(target_dir.path_abs, name, meta)
            result_set.assign_channel(mask_item, chan_image, meta)

        return result_set, target_dir

    def align(self, target_key, method="ncc"):
        from utils.helper_functions.alignments import Alignment, M_MUTUAL_INFORMATION, M_CROSS_CORRELATION
        align_ncc = Alignment(method=M_CROSS_CORRELATION)
        align_mutual = Alignment(method=M_MUTUAL_INFORMATION)
        img_keys = sorted(self.channels.keys())
        img_keys.remove('dic')
        img_keys.remove(target_key)

        target_align = align_ncc if method == "ncc" else align_mutual

        try:
            self.images['dic'], self.images[target_key], others = target_align.get_aligned_images(
                self.__getattr__("dic"),
                self.__getattr__(
                    target_key),
                [self.__getattr__(key)
                 for
                 key in
                 img_keys],
                apply_clahe=False)
        except ValueError:
            if method == "ncc":
                print("NCC align failed, fallback to mutual information")
                self.images['dic'], self.images[target_key], others = align_mutual.get_aligned_images(
                    self.__getattr__("dic"),
                    self.__getattr__(target_key),
                    [self.__getattr__(key) for
                     key in
                     img_keys], apply_clahe=False)

        for idx, key in enumerate(img_keys):
            self.images[key] = others[idx]

        return self.images['dic'], self.images[target_key]

    def __len__(self):
        return len(self.channels.keys())

    def to_patches(self, num_patches):
        raise NotImplementedError()
        chan_idx = self.channels.keys()
        stack = np.stack([self.__getattr__(chan) for chan in chan_idx], axis=-1)
        patches = img_to_patches(stack, num_patches)
        result_sets = []
        for patch_idx, patch in enumerate(patches):
            new_set = ImageSet(self.strategy, self._grayscale, self._8bit)
            for idx, chan_name in enumerate(chan_idx):
                chan_img = patch[:, :, idx]
                meta = self._clone_channel(chan_name)
                meta['patch_id'] = Series(str(patch_idx), index=self.channels[chan_name].index)
                new_set.assign_channel(chan_name, chan_img, meta)

            result_sets.append(new_set)

        return result_sets

    def display_dic_multimask(self):
        f, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 6))
        dic = self.image("dic")
        multim = self.get_mask(self.MULTIMASK_IDENTIFIER)

        ax[0].imshow(dic, cmap="gray")
        ax[1].imshow(multim, cmap="gray")

        ax[2].imshow(dic, cmap="gray")
        ax[2].imshow(multim, cmap="gray", alpha=0.4)
        plt.show()

    def empty_cache(self):
        del self.images
        self.images = DotDict()

    def delete(self, dry_run=True):
        print("Deleting image set " + str(self))
        print("Images")
        item_count = 0
        for key in self.channels.keys():
            path = self.channels[key]['base.file_path'].item()
            print(key, path)
            if not dry_run:
                os.remove(path)
            item_count += 1

        print("Masks")
        for key in self.masks:
            path = self.channels[key]['base.file_path'].item()
            print(key, path)

            if not dry_run:
                os.remove(path)

            item_count += 1

        print("{} items deleted".format(item_count))

    def display(self, time_out=None):

        keys = list(self.channels.keys())
        keys.remove("dic")
        keys = sorted(keys)
        imgs = [self.__getattr__(chan_name) for chan_name in ["dic"] + keys]
        plot_count = len(imgs)
        overlay = None
        if len(imgs) == 3:
            overlay = draw_fluorescence_overlays(imgs[0], blue=imgs[1], red=imgs[2])
            plot_count += 1

        f, ax = plt.subplots(1, plot_count, sharex=True, sharey=True, figsize=(12, 6))
        for idx, img in enumerate(imgs):
            ax[idx].imshow(img, cmap="gray")

        if overlay is not None:
            ax[plot_count - 1].imshow(overlay)
        plt.tight_layout()

        if not time_out:
            plt.show()
        else:
            plt.pause(time_out)
            plt.close()
