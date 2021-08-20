from torch.utils.data import Dataset

from utils.helper_functions.img_utils import load_image
from utils.models.folder import FilteredFileProvider
from vision.detection.model_setup.model_setup import get_transform


class PredictionDataSet(Dataset):
    def __init__(self, fp: FilteredFileProvider):
        self.data = list(fp)
        self.transforms = get_transform(train=False)

    def __getitem__(self, idx):
        path = self.data[idx]

        img = load_image(path, force_grayscale=True, force_8bit=True)

        if self.transforms is not None:
            img, _ = self.transforms(img, target=None)
        return img, dict(), path

    def __len__(self):
        return len(self.data)
