import cv2
import numpy as np

from Augmentor.Operations import Distort
from PIL import Image
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import torch

from utils.file_management.naming_strategies import CsvNamingStrategy
from utils.helper_functions.img_utils import load_image
from utils.models.folder import Folder


class ImageSetObjectDetectionDataset(Dataset):
    def __init__(self, root, max_gt_masks, naming_strategy, file_extensions=['tif', 'tiff', 'png'],
                 include_subdirs=False, transforms=None):

        if isinstance(root, str):
            root = Folder(root)
        self.root = root
        self.transforms = transforms
        self.num_gt_masks = max_gt_masks
        self.extensions = file_extensions
        self.include_subdirs = include_subdirs
        self.naming_strategy = naming_strategy
        self.masks = None
        self.imgs = None
        self.elastic_transform = Distort(1, 32, 32, 8)
        # not working with masks atm
        self.enable_elastic_transform = False

    def prepare(self):

        self.imgs = []
        self.masks = []

        image_sets = self.naming_strategy.find_image_sets(self.root, self.extensions,
                                                          self.include_subdirs)

        if len(image_sets) == 0:
            raise ValueError("Dataset is empty")

        for idx, ims in enumerate(image_sets):
            mask_dict = dict(ims.masks)
            if "multimask" in mask_dict.keys():
                del mask_dict["multimask"]

            mask_arr = []

            for mask in mask_dict.values():
                mask_arr.append(mask['base.file_path'].item())
            self.imgs.append(ims.channels['dic']['base.file_path'].item())
            self.masks.append(mask_arr)

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        img = load_image(img_path, force_8bit=True, force_grayscale=True)
        mask_paths = self.masks[idx]

        train_masks = min(len(mask_paths), self.num_gt_masks)
        selected_masks = np.random.choice(mask_paths, train_masks)

        mask_array = []
        for mask_path in selected_masks:
            mask = load_image(mask_path, force_8bit=True, force_grayscale=True)
            mask_array.append(mask.astype(np.uint8))

        if self.enable_elastic_transform and np.random.uniform() <= 1.15:
            mask_array = [Image.fromarray(m) for m in mask_array]
            transform_input = [Image.fromarray(img)]+mask_array
            result = self.elastic_transform.perform_operation(transform_input)
            img = result[0]
            mask_array = result[1:]
            img = np.asarray(img)
            masks = np.stack(mask_array, axis=-1)
        else:
            masks = np.stack(mask_array, axis=-1)

        if np.random.uniform() >= 0.5:
            img = np.fliplr(img).copy()
            masks = np.fliplr(masks).copy()

        if np.random.uniform() >= 0.5:
            img = np.flipud(img).copy()
            masks = np.flipud(masks).copy()

        # get bounding box coordinates for each mask
        num_objs = len(selected_masks)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[:, :, i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = masks.transpose((2, 0, 1))
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    root = "/Volumes/thesis_data/test_dataset_od/test"
    ds = ImageSetObjectDetectionDataset(root, 50, CsvNamingStrategy(), ["png"], include_subdirs=True)
    ds.prepare()

    for i in range(len(ds)):
        img, target = ds[i]
        for box in target['boxes']:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        plt.ion()
        plt.imshow(img)
        plt.pause(1.5)
