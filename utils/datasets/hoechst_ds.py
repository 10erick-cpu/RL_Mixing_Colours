import torchvision

from datasets.clean_ds_v1.ds_utils import DS_V2_BASE_DIR
from datasets.imagesets.base_image_set_dataset import ImageSetDatasetBase
from helper_functions.img_utils import show_subplot
from torchvision.transforms import ToTensor, Normalize, Resize, ToPILImage

base_path_hoechst_crops = DS_V2_BASE_DIR.get_file_path("190305_IBIDI_JIMT1_CT_HOE/train_test_split/{}/crops")


def get_path(crop_type):
    return base_path_hoechst_crops.format(crop_type)


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
        'img': {'mean': img_mean / count, 'std': img_std / count},
        'target': {'mean': target_mean / count, 'std': target_std / count}
    }
    print(result)
    return result


def _build_transforms(mean_dict=None, resize_w=256, resize_h=195):
    img_transforms = [
        ToPILImage(),
        Resize((resize_h, resize_w)),
        ToTensor()
    ]

    target_transforms = [
        ToPILImage(),
        Resize((resize_h, resize_w)),
        ToTensor()
    ]

    if mean_dict is not None:
        img_transforms.append(
            Normalize(mean=[mean_dict['img']['mean']], std=[mean_dict['img']['std']], inplace=False))
        # target_transforms.append(
        #   Normalize(mean=[mean_dict['target']['mean']], std=[mean_dict['target']['std']], inplace=True))

    return torchvision.transforms.Compose(img_transforms), torchvision.transforms.Compose(target_transforms)


class HoechstCropsBase(ImageSetDatasetBase):

    def __init__(self, ds_id, directory_path, target_channel, mean_dict, auto_init=False):
        super().__init__(directory_path, target_channel=target_channel, search_subdirs=True)
        self.ds_id = ds_id

        self.img_transforms, self.target_transforms = _build_transforms(mean_dict)

        if auto_init:
            self.init()

    def __getitem__(self, item):
        img, target, debug = super(HoechstCropsBase, self).__getitem__(item)
        return self.img_transforms(img), self.target_transforms(target), debug

    def calculate_ds_stats(self):
        img_trans, target_trans = self.img_transforms, self.target_transforms
        self.img_transforms, self.target_transforms = _build_transforms(None)
        result = calc_mean_std(self)
        self.img_transforms, self.target_transforms = img_trans, target_trans
        return result

    def update_transforms(self, mean_dict):
        self.img_transforms, self.target_transforms = _build_transforms(mean_dict)


class HoechstCropsTrain(HoechstCropsBase):
    MEAN_STD = {'img': {'mean': 0.5610017083521582, 'std': 0.06291756644825108},
                'target': {'mean': 0.218353591193531, 'std': 0.23812354927154708}}

    def __init__(self, auto_init=False):
        super().__init__("jmt1-hoechst-txred", get_path("train"), 'hoechst', self.MEAN_STD, auto_init=auto_init)


class HoechstCropsVal(HoechstCropsBase):
    MEAN_STD = {'img': {'mean': 0.5632534238490565, 'std': 0.06089147805685884},
                'target': {'mean': 0.21623920587127815, 'std': 0.23526287864403927}}

    def __init__(self, auto_init=False):
        super().__init__("jmt1-hoechst-txred", get_path("val"), 'hoechst', self.MEAN_STD, auto_init=auto_init)


class HoechstCropsTest(HoechstCropsBase):
    MEAN_STD = {'img': {'mean': 0.5475910775363445, 'std': 0.06259349195682444},
                'target': {'mean': 0.24735561767973802, 'std': 0.24924054996498549}}

    def __init__(self, auto_init=False):
        super().__init__("jmt1-hoechst-txred", get_path("test"), 'hoechst', self.MEAN_STD, auto_init=auto_init)


if __name__ == '__main__':
    ds = HoechstCropsTest()
    ds.init()

    print(ds.calculate_ds_stats())
