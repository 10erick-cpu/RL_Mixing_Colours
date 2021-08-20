from utils.file_management.naming_strategies import MetadataParser
from file_management.naming_strategies import ZtzRawDataStrategy, CsvNamingStrategy
from models.filesystem_utils import Folder


def import_raw_data_to_image_sets(src_folder, src_naming_strategy, out_folder, out_naming_strategy, num_patches=1,
                                  make_subfolders=True, align=False):
    input_folder = Folder(src_folder)
    output_dir = Folder(out_folder)
    parser = MetadataParser(src_naming_strategy)
    metadata = parser.extract_from(
        input_folder.make_file_provider(extensions=['tif', 'tiff', 'png'], include_subdirs=True))

    image_sets = src_naming_strategy.identify_image_sets(metadata)
    new_is = []
    for is_idx, ims in enumerate(image_sets):
        print("\rProcessing image set {}/{}".format(is_idx + 1, len(image_sets)), end="")
        if align:
            ims.align('hoechst')
        if num_patches != 1:
            patches = ims.to_patches(num_patches)
            for patch_ims in patches:
                new_is.append(patch_ims.persist(out_naming_strategy, output_dir, make_parent_folder=make_subfolders))
        else:
            new_is.append(ims.persist(out_naming_strategy, output_dir, make_parent_folder=make_subfolders))


def ztz_csv_images_to_csv_naming(src_folder, out_folder):
    input_folder = Folder(src_folder)
    output_dir = Folder(out_folder)

    src_naming = CsvNamingStrategy()
    dest_naming = CsvNamingStrategy()

    parser = MetadataParser(src_naming)
    metadata = parser.extract_from(
        input_folder.make_file_provider(extensions=['tif', 'tiff', 'png'], include_subdirs=True))

    image_sets = src_naming.identify_image_sets(metadata)

    for idx, ims in enumerate(image_sets):
        print("\rProcessing image set {}/{}".format(idx + 1, len(image_sets)), end="")
        # ims.align('outline')
        ims.persist(dest_naming, output_dir, make_parent_folder=False, dry_run=False)


if __name__ == '__main__':
    src_dir = "/mnt/unix_data/datastorage/raw_input_data/1_input_data/190320/20x/saturated/20190319_IBIDI_JIMT1_CT_HOE_SAT"
    target_dir = "/mnt/unix_data/datastorage/ztz_datasets/00_clean_ds_patches/20x/20190320_IBIDI_JIMT1_CT_HOE_SAT"
    import_raw_data_to_image_sets(src_dir, ZtzRawDataStrategy(), target_dir, CsvNamingStrategy(), num_patches=16,
                                  make_subfolders=False, align=True)

    # ztz_csv_images_to_csv_naming(src_dir, target_dir)
