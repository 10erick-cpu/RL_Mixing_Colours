import numpy as np
import torch
from torch.autograd.grad_mode import no_grad

from utils.helper_functions.torch_utils import training_device
from utils.models.dot_dict import DotDict
from vision.detection.model_setup.detection_utils import threshold_mask, draw_detection
from vision.detection.model_setup.model_setup import get_instance_segmentation_model, load_checkpoint

DEFAULT_COLORS = [16, 32, 64, 96, 128, 160, 192, 220, 255]


class ModelConfig(object):
    def __init__(self):
        pass

    NUM_CLASSES = 2
    MAX_DETECTIONS = 3000
    BOX_HEAD_NMS = 0.7
    BOX_HEAD_OBJ_TH = 0.8
    LOAD_PRE_TRAINED = False
    RPN_NMS_THRESH = 0.7
    RPN_POST_NMS_KEEP_INFERENCE = 3000


class MRCNNDetector(object):

    def setup_model(self):
        # our dataset has two classes only - background and cell
        # get the model using our helper function
        model = get_instance_segmentation_model(num_classes=self.config.NUM_CLASSES,
                                                pretrained=self.config.LOAD_PRE_TRAINED,
                                                rpn_pre_nms_top_n_test=3500,
                                                rpn_post_nms_top_n_test=self.config.RPN_POST_NMS_KEEP_INFERENCE,
                                                box_detections_per_img=self.config.MAX_DETECTIONS,
                                                box_nms_thresh=self.config.BOX_HEAD_NMS,

                                                rpn_nms_thresh=self.config.RPN_NMS_THRESH,
                                                box_score_thresh=self.config.BOX_HEAD_OBJ_TH
                                                )
        # move model to the right device
        # model.to(training_device())
        return load_checkpoint(self.cp_folder, self.model_id, model)

    def __init__(self, model_id, cp_folder=None):
        self.model_id = model_id
        self.cp_folder = cp_folder
        self.model = None
        self.epoch = -1
        self.config = None

    def load(self, model_config=ModelConfig()):
        self.config = model_config
        self.model, self.epoch = self.setup_model()
        self.model.eval()
        return self

    def detect_and_build_mask(self, input_tensor, mask_threshold=0.5):
        b, c, h, w = input_tensor.shape
        assert b == 1, "no batch support"
        scores, boxes, masks = self.detect(input_tensor[0].float())
        scores = scores.to("cpu")
        boxes = boxes.to("cpu")
        masks = masks.to("cpu")

        multimask_out = torch.zeros_like(input_tensor.squeeze()).int()
        masks = masks.squeeze()
        assert len(multimask_out.shape) == 2
        for i in range(masks.shape[0]):
            mask = masks[i].squeeze()
            mask = threshold_mask(mask, mask_threshold)
            masks[i] = mask
            multimask_out[mask != 0] = i + 1

        return scores, boxes, masks, multimask_out

    def detect_and_create_overlay(self, input_tensor, colors=DEFAULT_COLORS, confidence_th=0.5, mask_th=0.5):
        det_count = 0
        scores, boxes, masks, multimask_out = self.detect_and_build_mask(input_tensor, mask_threshold=mask_th)

        color_overlay = np.zeros((*input_tensor.squeeze().shape, 3), dtype=np.uint8)
        for idx, mask in enumerate(masks):

            if scores[idx] < confidence_th:
                continue
            det_count += 1
            mask = threshold_mask(mask, mask_th)
            color_overlay = draw_detection(color_overlay, mask, boxes[idx],
                                           color=np.random.choice(colors, size=3, replace=False))

        result_dict = DotDict({'output': (scores, boxes, masks, multimask_out),
                               'overlay': color_overlay,
                               'total_det': len(boxes), 'valid_det': det_count})
        return result_dict

    @no_grad()
    def detect(self, input_tensor, do_zip=False, plain_result=False):
        input_tensor = input_tensor.to(training_device())

        result = self.model.to(training_device())([input_tensor])[0]
        if plain_result:
            return result
        if do_zip:
            return zip(result['scores'], result['boxes'], result['masks'])
        return result['scores'], result['boxes'], result['masks']

    def detect_numpy(self, input_image):
        return self.detect(torch.from_numpy(input_image))
