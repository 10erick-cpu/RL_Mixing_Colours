import cv2
import numpy as np
import torch
from file_management.naming_strategies import CsvNamingStrategy, MetadataParser
from models.filesystem_utils import Folder


class ImageSetDataset(object):

    def __init__(self, inp_shape, directory_path, ds_mean=0, ds_std=None, target_channel="hoechst",
                 naming_strategy=CsvNamingStrategy(), mean_std_norm=False, return_class_ids=True):
        # super().__init__(inp_shape, normalize_fn=NormalizeToDataset(ds_mean, None), force_integer_mask=False)
        if directory_path is not None:
            self.data_dir = Folder(directory_path)

        self.data = None
        self.batch_shape = inp_shape
        self.naming_strategy = naming_strategy
        self.target_channel = target_channel.lower()
        self.mask_sanity_check_enabled = False
        self.ds_mean = ds_mean
        self.ds_std = ds_std
        # self.set_augmentation_enabled(False)
        self.augment = True
        self.mean_std_normalization = mean_std_norm
        self.return_class_ids=True

    def init(self):
        parser = MetadataParser(self.naming_strategy)
        self.data = self.naming_strategy.identify_image_sets(parser.extract_from(self.data_dir.make_file_provider()))

    def _get_item_count(self):
        return len(self.data)

    def _get_image_info(self, index):
        item = self.data[index]
        return item.channels['dic']["base.file_path"].item()

    def _get_mask_info(self, index):
        item = self.data[index]
        return item.channels[self.target_channel]["base.file_path"].item()

    def _mask_to_image(self, mask_data):
        return super(ImageSetDataset, self)._mask_to_image(mask_data), np.asarray([1])

    def __len__(self):
        if self.data is None:
            raise RuntimeError("ImageDataset not initialized")
        return len(self.data)

    def __add__(self, other):
        result = ImageSetDataset(self.batch_shape, None)
        result.data = self.data + other.data
        return result

    def split_test_train(self, test_ratio=0.1):
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(self.data, test_size=int(len(self) * test_ratio))
        train_set = ImageSetDataset(self.batch_shape, None, self.ds_mean, target_channel=self.target_channel,
                                    naming_strategy=self.naming_strategy)
        test_set = ImageSetDataset(self.batch_shape, None, self.ds_mean, target_channel=self.target_channel,
                                   naming_strategy=self.naming_strategy)
        train_set.data = train
        train_set.data_dir = self.data_dir
        test_set.data = test
        test_set.data_dir = self.data_dir
        return train_set, test_set

    def __getitem__(self, item):
        ims = self.data[item]
        #ims.display()
        input_img = ims._load('dic').astype(np.float32)

        target = ims._load(self.target_channel)
        input_img -= np.round(self.ds_mean * 255).astype(np.uint8)
        if self.batch_shape is not None:
            input_img = cv2.resize(input_img, (self.batch_shape[1], self.batch_shape[0]), interpolation=cv2.INTER_AREA)
            target = cv2.resize(target, (self.batch_shape[1], self.batch_shape[0]), interpolation=cv2.INTER_AREA)

        input_img = cv2.normalize(input_img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        target = cv2.normalize(target, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # target = (target - target.mean()) / (target.std() + 1e-8)
        target *= 255

        if self.augment:
            if np.random.uniform() > 0.5:
                input_img = np.flipud(input_img).copy()
                target = np.flipud(target).copy()
            if np.random.uniform() > 0.5:
                input_img = np.fliplr(input_img).copy()
                target = np.fliplr(target).copy()
            if np.random.uniform() > 0.7:
                # gamma adjustments (img/255)^gamma *255
                gamma = np.random.uniform(0.7, 1.5)
                input_img = input_img ** gamma

        if self.mean_std_normalization:
            # input_img = input_img.astype(np.float32)

            input_img = (input_img - self.ds_mean) / (self.ds_std + 1e-8)
            # target = (target - target.mean()) / (target.std() + 1e-8)
        # plt.imshow(input_img.squeeze() * 255, cmap="gray", vmin=0, vmax=255)
        # plt.show()
        # plt.imshow(target.squeeze(), cmap="gray", vmin=0, vmax=255)
        # plt.show()
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize(self.ds_mean, self.ds_std)
        #      ])

        input_img = torch.as_tensor(input_img[None, :, :])
        target = torch.as_tensor(target[None, :, :]).round().long()
        if self.return_class_ids:
            return input_img, target, np.asarray([1])
        else:
            return input_img, target

    # def _data_to_image(self, data):
    #    return data.dic.copy()

    # def _mask_to_image(self, mask_data):
    #    return image_to_float(mask_data.image(self.target_channel)), np.asarray([1]).copy()

    def compute_ds_mean(self):
        mean = None
        for idx in range(len(self)):
            img_nfo = self.data[idx]
            if mean is None:
                mean = img_nfo.image('dic').mean()
            else:
                mean += img_nfo.image('dic').mean()
                mean /= 2
        return mean

    def find_invalid_sets(self, min_target_intensity=30, delete=False, display=False):
        invalid = []
        for idx, (img, target, _) in enumerate(self):
            print("\r{}/{}".format(idx + 1, len(self)), end="")
            if target.max() < min_target_intensity:
                ims = self.data[idx]
                invalid.append((target.max().item(), ims))

        invalid = sorted(invalid, key=lambda x: x[0], reverse=display)

        print("Invalid data: ", len(invalid))
        for max_val, iv in invalid:
            print(iv)
            print(max_val)
            if display:
                iv.display()
            iv.delete(dry_run=(not delete))


if __name__ == '__main__':
    ds_dir = "/mnt/unix_data/datastorage/ztz_datasets/00_clean_ds_patches_10x/10x/jmt1-hoechst-txred"
    dataset = ImageSetDataset((300, 300), ds_dir, ds_mean=105.86375, target_channel='hoechst')
    dataset.init()

    import matplotlib.pyplot as plt

    print(dataset.compute_ds_mean())
    for img, mask, clz in dataset:
        print(img.shape)
        print(mask.shape)
        plt.imshow(img[0, :, :] * 255, cmap="gray")
        # plt.imshow(mask, alpha=0.4, cmap="jet")
        plt.show()
