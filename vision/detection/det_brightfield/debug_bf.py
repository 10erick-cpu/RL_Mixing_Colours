import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from utils.file_management.naming_strategies import CsvNamingStrategy
from utils.helper_functions.img_utils import load_image
from utils.helper_functions.model_persistence import CheckpointFolder
from utils.helper_functions.filesystem_utils import FilteredFileProvider, Folder
from utils.nn.cocoapi.custom import coco_utils
from vision.detection.detection_datasets.cell_outline_dataset import ImageSetObjectDetectionDataset
from vision.detection.model_setup.model_setup import get_transform


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, fp: FilteredFileProvider):
        self.data = list(fp)
        self.transforms = get_transform(train=False)

    def __getitem__(self, idx):
        path = self.data[idx]

        img = load_image(path, force_grayscale=True, force_8bit=True)

        if self.transforms is not None:
            img, _ = self.transforms(img, target=None)
        return img, dict()

    def __len__(self):
        return len(self.data)


def get_instance_segmentation_model(num_classes, pretrained=True):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained, box_detections_per_img=10000,
                                                               box_nms_thresh=0.25)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# use our dataset and defined transformations
root_train = "/mnt/unix_data/datastorage/thesis_experiment_results/fl_detection/ds_set1_2_train_test_split/train"

dataset = ImageSetObjectDetectionDataset(root_train, 50, CsvNamingStrategy(), ["png"], include_subdirs=True,
                                         transforms=get_transform(train=True))
# dataset.prepare()

root_test = "/mnt/unix_data/datastorage/thesis_experiment_results/fl_detection/ds_set1_2_train_test_split/test"
dataset_test = ImageSetObjectDetectionDataset(root_test, 50, CsvNamingStrategy(), ["png"], include_subdirs=True,
                                              transforms=get_transform(train=False))
dataset_test.prepare()

# split the dataset in train and test set
torch.manual_seed(1)
# indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.misc.data.Subset(dataset, indices[:-50])
# dataset_test = torch.misc.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
# data_loader = torch.misc.data.DataLoader(
#    dataset, batch_size=2, shuffle=True, num_workers=4,
#    collate_fn=misc.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=coco_utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes, pretrained=False)
# move model to the right device
model.to(device)
model_id = "mrcnntv_default"
# model_id = "mrcnntv_fl_detection"

model_checkpoint_folder = CheckpointFolder("../checkpoints")

cp = model_checkpoint_folder.load_last_training_checkpoint(model_id, strict=False)
if cp is not None:
    model.load_state_dict(cp['model_state_dict'])

metric_logger = coco_utils.MetricLogger(delimiter="  ")

test_path = "/mnt/unix_data/datastorage/raw_input_data/1_input_data/190320/20x/20190319_IBIDI_JIMT1_Celltracker_red&Hoechst_001_contam_20x"
# test_path = "/mnt/unix_data/datastorage/raw_input_data/1_input_data/190327/10x/live_dead_20190123_CTgreen_DRAQ_004"
test_path = "/mnt/unix_data/datastorage/ztz_analysis/dario_31.07/JIMT-1 Inc/T8"
# test_path ="/mnt/unix_data/datastorage/raw_input_data/1_input_data/190314_fl_hoechst_ct/190305_IBIDI_JIMT1_Celltracker_Hoechst/set_1/original_fl"
ds_imgs = SimpleDataset(
    Folder(test_path).make_file_provider(extensions=['tif']))

data_loader_raw_imgs = torch.utils.data.DataLoader(
    ds_imgs, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=coco_utils.collate_fn)

model.eval()
for idx, (images, targets) in enumerate(metric_logger.log_every(data_loader_raw_imgs, 1, "header")):
    images = list(image.to(device) for image in images)

    src_image = images[0].cpu().numpy().transpose((1, 2, 0))
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    with torch.no_grad():
        loss_dict_arr = model(images)
    loss_dict = loss_dict_arr[0]
    boxes = loss_dict['boxes']
    # masks = targets[0]['masks']
    masks = loss_dict['masks']

    print(masks.shape)

    print(masks[0].max(), masks[0].min())

    image = src_image

    image = (image * 255).astype(np.uint8).squeeze()
    for coords in boxes:
        coords = coords.cpu().numpy()

        cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 2)

    multi_mask = None
    for mask in masks:
        if multi_mask is None:
            multi_mask = mask.cpu().numpy()
        else:
            # mask = torch.nn.functional.sigmoid(mask)
            mask = mask.cpu().numpy()
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            # mask = (mask - mask.min()) / (mask.max() - mask.min())
            # mask[mask >= 0.5] = 1

            # plt.imshow(mask.squeeze())
            # plt.show()
            multi_mask[mask == 1] = 1

    # cv2.rectangle(image, (50, 50), (100, 100), (255, 255, 255), 3)
    f, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(src_image.squeeze(), cmap="gray")
    ax[1].imshow(image, cmap="gray")
    ax[1].imshow(multi_mask.squeeze() * 128, cmap="jet", alpha=0.1)

    out_folder = Folder("./pred_output")
    out_folder.make_dir()

    f.savefig(out_folder.get_file_path(f"f8_out#{idx}"))
    plt.close(f)

    print(loss_dict)
