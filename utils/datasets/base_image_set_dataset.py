from utils.file_management.naming_strategies import CsvNamingStrategy, MetadataParser
from utils.models.dot_dict import DotDict
from utils.models.folder import Folder
from torch.utils.data import Dataset


class ImageSetDatasetBase(Dataset):

    def __init__(self, directory_path, target_channel="hoechst",
                 naming_strategy=CsvNamingStrategy(), search_subdirs=False):
        if directory_path is not None:
            self.data_dir = Folder(directory_path)

        self.data = None
        self.naming_strategy = naming_strategy
        self.target_channel = target_channel.lower()
        self.return_class_ids = True
        self.search_subdirs = search_subdirs

    def init(self):
        parser = MetadataParser(self.naming_strategy)
        self.data = self.naming_strategy.identify_image_sets(parser.extract_from(self.data_dir.make_file_provider(include_subdirs=self.search_subdirs)))

    def __len__(self):
        if self.data is None:
            raise RuntimeError("ImageDataset not initialized")
        if len(self.data) == 0:
            raise RuntimeError("ImageDataset does not contain any data")
        return len(self.data)

    def __getitem__(self, item):
        ims = self.data[item]
        input_img = ims._load('dic')
        target = ims._load(self.target_channel)
        return input_img, target, DotDict({'ims': ims.channels['dic']['base.file_path'].item()})

    def find_invalid_sets(self, min_target_intensity=30, delete=False, display=False):
        invalid = []
        for idx in range(len(self)):
            img, target, _ = self[idx]
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
   pass