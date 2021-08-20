import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def load_checkpoint(cp_folder, model_id, model):
    cp = cp_folder.load_last_training_checkpoint(model_id, strict=False)
    epoch = 0
    if cp is not None:
        model.load_state_dict(cp['model_state_dict'])
        epoch = cp['epoch']
        print(f"Checkpoint of epoch {epoch} loaded")
    else:
        print("No checkpoint found in ", cp_folder)

    return model, epoch


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # transforms.append(T.RandomHorizontalFlip(0.5))
        pass
    return T.Compose(transforms)


def get_instance_segmentation_model(num_classes, pretrained=True,
                                    # transform parameters
                                    min_size=800, max_size=1333,
                                    image_mean=None, image_std=None,
                                    # RPN
                                    rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                                    rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                                    rpn_nms_thresh=0.7,
                                    rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                                    rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                                    # Box parameters
                                    box_roi_pool=None, box_head=None, box_predictor=None,
                                    box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                                    box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                                    box_batch_size_per_image=512, box_positive_fraction=0.25,
                                    bbox_reg_weights=None):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained,
                                                               # transform parameters
                                                               min_size=min_size, max_size=max_size,
                                                               image_mean=image_mean, image_std=image_std,

                                                               rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                                                               rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                                                               rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                                                               rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
                                                               rpn_nms_thresh=rpn_nms_thresh,
                                                               rpn_fg_iou_thresh=rpn_fg_iou_thresh,
                                                               rpn_bg_iou_thresh=rpn_bg_iou_thresh,
                                                               rpn_batch_size_per_image=rpn_batch_size_per_image,
                                                               rpn_positive_fraction=rpn_batch_size_per_image,
                                                               # Box parameters
                                                               box_score_thresh=box_score_thresh,
                                                               box_nms_thresh=box_nms_thresh,
                                                               box_detections_per_img=box_detections_per_img,
                                                               box_fg_iou_thresh=box_fg_iou_thresh,
                                                               box_bg_iou_thresh=box_bg_iou_thresh,
                                                               box_batch_size_per_image=box_batch_size_per_image,
                                                               box_positive_fraction=box_positive_fraction,
                                                               bbox_reg_weights=bbox_reg_weights)

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
