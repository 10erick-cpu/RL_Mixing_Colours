import matplotlib.pyplot as plt
import numpy as np
import torch

# split the dataset in train and test set
from utils.helper_functions.model_persistence import CheckpointFolder
from utils.nn.cocoapi.custom import coco_utils
from utils.nn.cocoapi.custom.engine import evaluate
from vision.detection.detection_datasets.fl_multimask_dataset import KaggleDs
from vision.detection.model_setup.detection_utils import draw_bboxes, detections_to_multimask
from vision.detection.model_setup.model_setup import get_transform, get_instance_segmentation_model

torch.manual_seed(1)

dataset_test = KaggleDs(get_transform(train=False))

indices = torch.randperm(len(dataset_test)).tolist()

dataset_test = torch.utils.data.Subset(dataset_test, indices)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=coco_utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and cell
# get the model using our helper function
model = get_instance_segmentation_model(num_classes=2, pretrained=True,
                                        box_detections_per_img=10000,
                                        box_nms_thresh=0.25)
# move model to the right device
model.to(device)
model_id = "mrcnntv_fl_detection"

model_checkpoint_folder = CheckpointFolder("../checkpoints")

cp = model_checkpoint_folder.load_last_training_checkpoint(model_id, strict=False)
if cp is not None:
    model.load_state_dict(cp['model_state_dict'])

metric_logger = coco_utils.MetricLogger(delimiter="  ")

model.eval()

evaluate(model, data_loader_test, device)

for images, targets in metric_logger.log_every(data_loader_test, 1, "header"):
    images = list(image.to(device) for image in images)

    src_image = images[0].cpu().numpy().transpose((1, 2, 0))
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    with torch.no_grad():
        loss_dict_arr = model(images)
    loss_dict = loss_dict_arr[0]
    boxes = loss_dict['boxes']
    target_masks = targets[0]['masks']
    masks = loss_dict['masks']

    image = src_image

    image = (image * 255).astype(np.uint8).squeeze()

    image = draw_bboxes(image, boxes)

    det_multi_mask = detections_to_multimask(masks)
    target_multi_mask = detections_to_multimask(target_masks)

    diff = target_multi_mask - det_multi_mask

    missed = np.count_nonzero(diff)
    print(diff.shape)
    print(missed, "{0:.2f}".format(missed / (diff.shape[1] * diff.shape[2])))

    plt.imshow(diff.squeeze(), cmap="jet")
    plt.imshow(image, cmap="gray", alpha=0.5)
    # plt.imshow(target_multi_mask.squeeze(), cmap="jet", alpha=0.5)
    plt.show()

    print(loss_dict)
