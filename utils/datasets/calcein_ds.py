import torchvision

from utils.datasets.ds_utils import DS_V2_BASE_DIR
from utils.datasets.hoechst_ds import HoechstCropsTest, HoechstCropsTrain, HoechstCropsVal
from utils.datasets.base_image_set_dataset import ImageSetDatasetBase
from torchvision.transforms import ToTensor, Normalize, Resize, ToPILImage

base_path_calcein_crops = DS_V2_BASE_DIR.get_file_path("calcein/train_test_split/{}/crops")

COMBINED_STATS_CALC_HOE_TRAIN = {'img': {'mean': 0.524397582207289, 'std': 0.053815067373785296}}
COMBINED_STATS_CALC_HOE_TEST = {'img': {'mean': 0.5063164788049956, 'std': 0.053409717894149555}}
COMBINED_STATS_CALC_HOE_VAL = {'img': {'mean': 0.5300467579981143, 'std': 0.05569917475614189}}


def get_path(crop_type):
    return base_path_calcein_crops.format(crop_type)


def compute_weighted_mean_std(ds_1, ds_2):
    weighted_mean_mean = ds_1.MEAN_STD['img']['mean'] * len(ds_1) + ds_2.MEAN_STD['img']['mean'] * len(ds_2)
    weighted_mean_std = ds_1.MEAN_STD['img']['std'] * len(ds_1) + ds_2.MEAN_STD['img']['std'] * len(ds_2)
    weighted_mean_mean /= len(ds_1) + len(ds_2)
    weighted_mean_std /= len(ds_1) + len(ds_2)

    return weighted_mean_mean, weighted_mean_std


def calc_mean_std(ims_ds_base):
    img_mean = 0
    img_std = 0
    target_mean = 0
    target_std = 0
    count = 0

    for idx, (img, target, debug) in enumerate(ims_ds_base):
        print("\r{}/{}".format(idx + 1, len(ims_ds_base)), end="")
        img_mean += img.mean().item()
        img_std += img.std().item()
        target_mean += target.mean().item()
        target_std += target.std().item()
        count += 1
    result = {
        'img': {'mean': (img_mean / count), 'std': img_std / count},
        'target': {'mean': target_mean / count, 'std': target_std / count}
    }
    print(result)
    return result


def _build_transforms(mean_dict=None, resize_w=256, resize_h=195):
    img_transforms = [
        ToPILImage(),
        Resize((resize_h, resize_w)),
        # Resize((resize_w, resize_w)),
        ToTensor()
    ]

    target_transforms = [
        ToPILImage(),
        Resize((resize_h, resize_w)),
        # Resize((resize_w, resize_w)),
        ToTensor()
    ]

    if mean_dict is not None:
        img_transforms.append(
            Normalize(mean=[mean_dict['img']['mean']], std=[mean_dict['img']['std']], inplace=False))
        # target_transforms.append(
        #   Normalize(mean=[mean_dict['target']['mean']], std=[mean_dict['target']['std']], inplace=True))

    return torchvision.transforms.Compose(img_transforms), torchvision.transforms.Compose(target_transforms)


class CalceinDenseCrops(ImageSetDatasetBase):

    def __init__(self, ds_id, directory_path, target_channel, mean_dict=None, auto_init=False):
        super().__init__(directory_path, target_channel=target_channel, search_subdirs=True)
        self.ds_id = ds_id
        self.img_transforms, self.target_transforms = _build_transforms(mean_dict)
        if auto_init:
            self.init()

    def calculate_ds_stats(self):
        img_trans, target_trans = self.img_transforms, self.target_transforms
        self.img_transforms, self.target_transforms = _build_transforms(None)
        result = calc_mean_std(self)
        self.img_transforms, self.target_transforms = img_trans, target_trans
        return result

    def __getitem__(self, item):
        img, target, debug = super(CalceinDenseCrops, self).__getitem__(item)
        return self.img_transforms(img), self.target_transforms(target), debug

    def update_transforms(self, mean_dict):
        self.img_transforms, self.target_transforms = _build_transforms(mean_dict)


class CalceinCropsTrain(CalceinDenseCrops):
    # recalculate if needed

    MEAN_STD = {'img': {'mean': 0.45641849079538904, 'std': 0.03691042623549168},
                'target': {'mean': 0.12406172517007069, 'std': 0.13945796314320186}}

    def __init__(self, auto_init=False):
        super().__init__("calcein_train", get_path("train"), 'calcein', self.MEAN_STD, auto_init=auto_init)


class CalceinCropsVal(CalceinDenseCrops):
    # recalculate if needed

    MEAN_STD = {'img': {'mean': 0.465847204019626, 'std': 0.045660721708089116},
                'target': {'mean': 0.13003192598248522, 'std': 0.1284187944174434}}

    def __init__(self, auto_init=False):
        super().__init__("calcein_val", get_path("val"), 'calcein', self.MEAN_STD, auto_init=auto_init)


class CalceinCropsTest(CalceinDenseCrops):
    # recalculate if needed

    MEAN_STD = {'img': {'mean': 0.4375254809194141, 'std': 0.03810342778969142},
                'target': {'mean': 0.1443533836491406, 'std': 0.1328970885124161}}

    def __init__(self, auto_init=False):
        super().__init__("calcein_test", get_path("test"), 'calcein', self.MEAN_STD, auto_init=auto_init)


if __name__ == '__main__':
    val = compute_weighted_mean_std(CalceinCropsVal(auto_init=True), HoechstCropsVal(auto_init=True))
    print("val")
    print(val)
    print()

    train = compute_weighted_mean_std(CalceinCropsTrain(auto_init=True), HoechstCropsTrain(auto_init=True))
    print("train")
    print(train)
    print()
    test = compute_weighted_mean_std(CalceinCropsTest(auto_init=True), HoechstCropsTest(auto_init=True))
    print("test")
    print(test)
    print()
