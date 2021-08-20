import cv2
import numpy as np
import torch


from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

from utils.datasets.ds_utils import DS_ROOT_DIR
from utils.file_management.naming_strategies import CsvNamingStrategy
from utils.helper_functions.filesystem_utils import split_filename_extension
from utils.helper_functions.img_utils import load_image
from utils.models.folder import Folder

DATASET_ROOT = DS_ROOT_DIR


def process_multi_mask(targets):
    # instances are encoded as different colors
    obj_ids = np.unique(targets)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    num_keep = min(len(obj_ids), 30)
    # keep_objs = np.random.choice(obj_ids, num_keep)

    # obj_ids = keep_objs

    # split the color-encoded mask into a set
    # of binary masks
    masks = targets == obj_ids[:, None, None]

    # get bounding box coordinates for each mask
    num_objs = len(obj_ids)
    if num_objs == 0:
        raise IndexError("No masks available")

    boxes = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

    return masks, boxes, num_objs


class FlObjectDetectionDs(Dataset):
    def __init__(self, image_folder, mask_folder, transforms=None, augment=False, max_gt_instances=1000,
                 center_input=True):
        if isinstance(image_folder, str):
            image_folder = Folder(image_folder)

        if isinstance(mask_folder, str):
            mask_folder = Folder(mask_folder)

        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transforms = transforms
        self.augment = augment
        self.data = []
        self.max_gt_instances = max_gt_instances
        self.center_input = center_input

        self._build_dataset(image_folder, mask_folder, self.data)

    def _build_dataset(self, image_folder, mask_folder, dataset):
        images = sorted(image_folder.get_files(abs_path=False))
        masks = sorted(mask_folder.get_files(abs_path=False))

        for image in images:
            image_id, ext = split_filename_extension(image)
            for mask in masks:
                mask_id, ext = split_filename_extension(mask)
                mask_id = mask_id.replace("_multimask", "")
                if image_id == mask_id:
                    dataset.append((image, mask))
                    break

    def __len__(self):
        return len(self.data)

    def check_unsqueeze(self, tensor, unsqueeze_dim=0):
        if len(tensor.shape) <= 1:
            return tensor.unsqueeze(unsqueeze_dim)
        return tensor

    def __getitem__(self, idx):
        # load images ad masks
        image_name, mask_name = self.data[idx]
        img_path = self.image_folder.get_file_path(image_name)
        mask_path = self.mask_folder.get_file_path(mask_name)
        # print(img_path, mask_path)

        img = load_image(img_path, force_grayscale=True, force_8bit=True)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = load_image(mask_path, force_grayscale=True, force_8bit=False)

        mask = np.array(mask)

        if self.augment:
            if np.random.uniform() >= 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

            if np.random.uniform() >= 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

        masks, boxes, num_objs = process_multi_mask(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        min_area = 0

        valid_ids = (area > min_area).nonzero()

        num_keep = min(valid_ids.nelement(), self.max_gt_instances)

        if num_keep < valid_ids.nelement():
            keep_objs = np.random.choice(valid_ids.numpy().squeeze(), num_keep)

            valid_ids = keep_objs

        image_id = torch.tensor([idx])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes[valid_ids].squeeze()
        target["labels"] = labels[valid_ids].squeeze()
        target["masks"] = masks[valid_ids].squeeze()
        target["image_id"] = image_id
        target["area"] = area[valid_ids].squeeze()
        target["iscrowd"] = iscrowd[valid_ids].squeeze()

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if num_objs == 1:
            target["boxes"] = target["boxes"].unsqueeze(0)
            target["masks"] = target["masks"].unsqueeze(0)
            target["area"] = target["area"].unsqueeze(0)
            target["labels"] = target["labels"].unsqueeze(0)
            target["iscrowd"] = target['iscrowd'].unsqueeze(0)

        img = img.float()
        img = (img - img.min()) / (img.max() - img.min())
        if self.center_input:
            img = (img - img.mean()) / img.std()

        return img, target


class GeneratedImagesetDataset(Dataset):
    def __init__(self, image_sets, transforms=None, augment=False, max_gt_instances=1000, input_image="dic",
                 center_input=True):
        self.image_sets = image_sets
        self.transforms = transforms
        self.augment = augment
        self.max_gt_instances = max_gt_instances
        self.input_image = input_image
        self.center_input = center_input
        assert self.input_image in ['dic', 'hoechst', 'calcein']

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, idx):
        ims = self.image_sets[idx]

        # load images ad masks
        img = ims._load(self.input_image, force_8bit=True)
        mask = ims._load('multimask', force_8bit=False, is_mask=True)
        assert mask.dtype in [np.uint16, np.int32], mask.dtype

        if self.augment:
            if np.random.uniform() >= 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

            if np.random.uniform() >= 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

        masks, boxes, num_objs = process_multi_mask(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        min_area = 0

        valid_ids = (area > min_area).nonzero()

        num_keep = min(valid_ids.nelement(), self.max_gt_instances)

        if num_keep < valid_ids.nelement():
            keep_objs = np.random.choice(valid_ids.numpy().squeeze(), num_keep)

            valid_ids = keep_objs

        image_id = torch.tensor([idx])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes[valid_ids].squeeze()
        target["labels"] = labels[valid_ids].squeeze()
        target["masks"] = masks[valid_ids].squeeze()
        target["image_id"] = image_id
        target["area"] = area[valid_ids].squeeze()
        target["iscrowd"] = iscrowd[valid_ids].squeeze()

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if num_objs == 1:
            target["boxes"] = target["boxes"].unsqueeze(0)
            target["masks"] = target["masks"].unsqueeze(0)
            target["area"] = target["area"].unsqueeze(0)
            target["labels"] = target["labels"].unsqueeze(0)
            target["iscrowd"] = target['iscrowd'].unsqueeze(0)

        img = img.float()
        img = (img - img.min()) / (img.max() - img.min())
        if self.center_input:
            img = (img - img.mean()) / img.std()

        return img, target


class GroundTruthDatasetSparse(GeneratedImagesetDataset):
    def __init__(self, transforms, center_input=True):
        path_gt = "/mnt/unix_data/datastorage/ztz_datasets/01_clean_ds_v2/annotated_ground_truth/calcein/test_converted_sparse"
        ims = CsvNamingStrategy().find_image_sets(Folder(path_gt), subdirs=True)
        super().__init__(ims, transforms, augment=False, max_gt_instances=10000, center_input=center_input)


class GroundTruthDatasetDense(GeneratedImagesetDataset):
    def __init__(self, transforms, center_input=True):
        path_gt = "/mnt/unix_data/datastorage/ztz_datasets/01_clean_ds_v2/annotated_ground_truth/calcein/test_converted_dense"
        ims = CsvNamingStrategy().find_image_sets(Folder(path_gt), subdirs=True)
        super().__init__(ims, transforms, augment=False, max_gt_instances=10000, center_input=center_input)


class KaggleDs(FlObjectDetectionDs):
    def __init__(self, transforms=None, max_gt_instances=1000):
        base = DATASET_ROOT.get_file_path("external_datasets/fl_object_dataset/kaggle/")
        super().__init__(base + "images", base + "masks", transforms, max_gt_instances=max_gt_instances)


class BBBC039(FlObjectDetectionDs):
    def __init__(self, transforms=None,
                 max_gt_instances=1000):
        base = DATASET_ROOT.get_file_path("external_datasets/fl_object_dataset/bbbc039/")
        super().__init__(base + "images", base + "masks", transforms, max_gt_instances=max_gt_instances)


class BBBC006(FlObjectDetectionDs):
    def __init__(self, transforms=None, max_gt_instances=1000):

        base = DATASET_ROOT.get_file_path("external_datasets/fl_object_dataset/bbbc006/")

        super().__init__(base + "BBBC006_v1_images_z_16", base + "BBBC006_v1_labels",
                         transforms, max_gt_instances=max_gt_instances)

    def _build_dataset(self, image_folder, mask_folder, dataset):
        images = sorted(image_folder.get_files(abs_path=False))
        masks = sorted(mask_folder.get_files(abs_path=False))

        for image in images:
            if "_w2" in image:
                continue
            image_id, ext = split_filename_extension(image)
            image_id = image_id.split("_w1")[0]
            for mask in masks:
                if mask.startswith(image_id):
                    dataset.append((image, mask))
                    break


def draw_bbox(img, targets):
    boxes = targets['boxes']

    for coords in boxes:
        x1, y1, x2, y2 = coords.int().numpy().tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)

    plt.imshow(img, cmap="gray")
    plt.show()


def test_bbox_extraction():
    import matplotlib.pyplot as plt
    ds = BBBC006()

    # for idx, (img, target) in enumerate(ds):
    #     boxes = target['boxes']
    #     area = target['area']
    #     img_name, mask_name = ds.data[idx]
    #     print(img_name)
    #     print(boxes.shape)
    #     if boxes.shape[0] <= 1:
    #         for box, area in zip(boxes, area):
    #             print("a: ", area, "b", box)
    #         raise idx

    target_image = "mcf-z-stacks-03212011_i18_s2_w106ad4d95-6c95-4d70-9f10-7582af2ed02e.tif"
    target_idx = None
    for idx, (img_name, mask_name) in enumerate(ds.data):
        if img_name == target_image:
            target_idx = idx
            break

    # masks = load_image(
    # "/mnt/unix_data/datastorage/external_datasets/fl_object_dataset/kaggle/masks/kgds_item_36_multimask.tif")
    # inflated_masks, boxes, num_objs = KaggleDs.process_multi_mask(masks)

    img, target = ds[target_idx]
    boxes = target['boxes']
    area = target['area']
    for box, area in zip(boxes, area):
        print("a: ", area, "b", box)

    for coords in boxes:
        x1, y1, x2, y2 = coords
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)

    # for i in range(inflated_masks.shape[0]):
    #     plt.imshow(inflated_masks[i, :, :])
    #     plt.show()

    plt.imshow(img, cmap="gray")
    plt.show()
