import os

from utils.models.dot_dict import DotDict


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def split_filename_extension(file_path):
    name, ext = os.path.splitext(file_path)
    return os.path.basename(name), ext


def is_folder(path):
    return os.path.isdir(path)


def get_image_info(file_path):
    import imageio
    img = imageio.imread(file_path)
    img_info = DotDict()
    img_info.shape = DotDict()
    img_info.shape.width = img.shape[1]
    img_info.shape.height = img.shape[0]
    img_info.shape.depth = img.shape[2]
    img_info.path = file_path
    img_info.filename = file_path.split("/")[-1].split(".")[0]
    return img_info
