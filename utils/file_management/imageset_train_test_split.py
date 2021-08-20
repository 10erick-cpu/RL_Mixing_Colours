from typing import List

import numpy as np
from utils.file_management.image_set import ImageSet
from utils.file_management.naming_strategies import CsvNamingStrategy
from utils.models.folder import Folder
from sklearn.model_selection import train_test_split

IMSArray = List[ImageSet]


def make_train_test_split(imageset_array: IMSArray, output_dir: Folder, test_ratio=0.15):
    num_items = len(imageset_array)
    idxes = np.arange(num_items)
    train_idx, test_idx = train_test_split(idxes, test_size=test_ratio)

    train_folder = output_dir.make_sub_folder("train")
    test_folder = output_dir.make_sub_folder("test")
    count = 0
    for i, idx in enumerate(train_idx):
        print("Save train {}/{}".format(i, len(train_idx)))
        ims = imageset_array[idx]
        ims.persist(CsvNamingStrategy(), train_folder, dry_run=False, make_parent_folder=True)
        count += 1
        ims.empty_cache()

    count = 0
    for i, idx in enumerate(test_idx):
        print("Save test {}/{}".format(i, len(test_idx)))
        ims = imageset_array[idx]
        ims.persist(CsvNamingStrategy(), test_folder, dry_run=False, make_parent_folder=True)
        count += 1
        ims.empty_cache()


base_folder = "/mnt/unix_data/datastorage/thesis_experiment_results/fl_detection/dataset_set1_2"
base_folder = Folder(base_folder)
sets = CsvNamingStrategy().find_image_sets(base_folder, subdirs=True)

output_dir = Folder("/mnt/unix_data/datastorage/thesis_experiment_results/fl_detection/ds_set1_2_train_test_split")
output_dir.make_dir()

make_train_test_split(sets, output_dir)
