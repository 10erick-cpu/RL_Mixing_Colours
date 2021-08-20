import os

from models.filesystem_utils import Folder

DS_V2_BASE_DIR = Folder(os.getenv("DS_V1_BASE_DIR", "/mnt/unix_data/datastorage/ztz_datasets/01_clean_ds_v2"))

DS_ROOT_DIR = Folder(os.getenv("DS_ROOT_DIR", "/mnt/unix_data/datastorage/"))

assert DS_ROOT_DIR.exists(), "ds root dir not found"
assert DS_V2_BASE_DIR.exists(), "ds_v2 base dir not found"
