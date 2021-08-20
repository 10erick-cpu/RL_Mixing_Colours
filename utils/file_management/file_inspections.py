from utils.file_management.naming_strategies import MetadataParser
from utils.file_management.naming_strategies import ZtzRawDataStrategy
from utils.models.folder import Folder


def inspect_ztz_raw_info_files(base_dir, include_subdirs=True):
    image_sets = find_image_sets(base_dir, ZtzRawDataStrategy(), include_subdirs)

    for ims in image_sets:
        ims.display()


def find_image_sets(base_dir, naming_strategy, include_subdirs=False, extensions=['tif', 'tiff', 'png']):
    input_folder = Folder(base_dir)

    parser = MetadataParser(naming_strategy)
    metadata = parser.extract_from(
        input_folder.make_file_provider(extensions=extensions, include_subdirs=include_subdirs))

    return naming_strategy.identify_image_sets(metadata)


if __name__ == '__main__':
    path = "/mnt/unix_data/datastorage/raw_input_data/1_input_data/190320/20x/saturated/20190320_IBIDI_JIMT1_Celltracker_orange_Hoechst_001_20x"
    inspect_ztz_raw_info_files(path)
