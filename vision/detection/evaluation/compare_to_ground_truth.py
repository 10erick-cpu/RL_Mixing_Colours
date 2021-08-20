from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import torch

import utils.nn.cocoapi.custom.custom_coco_eval as cce
from utils.helper_functions.torch_utils import training_device
from utils.helper_functions.filesystem_utils import Folder
from utils.nn.cocoapi.custom import coco_utils
from utils.nn.cocoapi.custom.coco_results_handler import CocoResultsHandler
from utils.nn.cocoapi.custom.engine import evaluate
from vision.detection.detection_datasets.fl_multimask_dataset import GroundTruthDatasetSparse, GroundTruthDatasetDense
from vision.detection.model_setup.detection_utils import threshold_mask, draw_detection
from vision.detection.model_setup.model_setup import get_transform
from vision.prediction.settings.model_configurations import Sigmoid, Raw, Softmax

batch_size = 3
COLORS = [16, 32, 64, 96, 128, 160, 192, 220, 255]


class PredictorDetectorWrapper(torch.nn.Module):

    def inspect_detection(self, input, detection_target):
        scores, boxes, masks, multimask_out = self.detector.detect_and_build_mask(detection_target, mask_threshold=0.5)

        color_overlay = np.zeros((*detection_target.squeeze().shape, 3), dtype=np.uint8)

        det_count = 0
        confidence_th = 0.5
        for idx, mask in enumerate(masks):

            if scores[idx] < confidence_th:
                continue
            det_count += 1
            mask = threshold_mask(mask, 0.5)
            color_overlay = draw_detection(color_overlay, mask, boxes[idx],
                                           color=np.random.choice(COLORS, size=3, replace=False))
        print(f"{det_count}/{len(boxes)} valid detections")
        f, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 8))
        input = (input - input.min()) / (input.max() - input.min())
        dic = input.squeeze().cpu().numpy()
        result = detection_target.squeeze().cpu().numpy()
        ax[0][0].imshow(dic, cmap="gray")
        ax[0][1].imshow(result, cmap="gray")

        ax[1][0].imshow(dic, cmap="gray")
        ax[1][0].imshow(color_overlay, alpha=0.3)

        ax[1][1].imshow(result, cmap="gray")
        ax[1][1].imshow(color_overlay, alpha=0.3)
        plt.tight_layout()
        import seaborn as sns
        sns.despine(f, bottom=True, left=True, trim=True)
        plt.show()

    def __init__(self, predictor, detector):
        super().__init__()
        self.predictor = predictor
        self.detector = detector
        self.predictor.eval()
        self.detector.model.eval()

    def forward(self, input):
        if isinstance(input, list):
            assert len(input) == 1
            input = input[0]
            input = input.unsqueeze(0)
            print(input.shape)

        input = input.to(training_device())

        detection_target = self.predictor(input)

        if len(detection_target.shape) == 2:
            detection_target = detection_target.unsqueeze(0).unsqueeze(0)
        elif len(detection_target.shape) == 3:
            detection_target = detection_target.unsqueeze(0)

        detection_target = detection_target.float()
        # detection_target = (detection_target - detection_target.mean()) / detection_target.std()
        detection_target = (detection_target - detection_target.min()) \
                           / (detection_target.max() - detection_target.min())

        detection_target = (detection_target - detection_target.mean()) / detection_target.std()

        # self.inspect_detection(input, detection_target)
        result = self.detector.detect(detection_target[0], plain_result=True)

        return [result]


def ds_to_dataloader(ds, is_train, num_workers=cpu_count()):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
        collate_fn=coco_utils.collate_fn)


gt_sparse = GroundTruthDatasetSparse(get_transform(False))
gt_dense = GroundTruthDatasetDense(get_transform(False))

eval_model = "fl"

if eval_model == "fl":

    batch_size = 1

    from vision.detection.det_fluorescence.train_fl_detection import model_id, model_checkpoint_folder
    from vision.detection.det_fluorescence.nuclei_pred_to_dic_masks import get_detector, get_predictor

    configs = [
        (Sigmoid.Batch.bce(), "Sigmoid_Batch_bce"),
        (Sigmoid.Instance.bce(), "Sigmoid_Instance_bce"),
        (Sigmoid.Instance.l1(), "Sigmoid_Instance_l1"),
        (Raw.Instance.l1(), "Raw_Instance_l1"),
        (Raw.Batch.l2(), "Raw_Batch_l1"),
        (Raw.Instance.l2(), "Raw_Instance_l2"),
        (Raw.Instance.smooth_l1(), "Raw_Instance_smoothl1"),
        (Softmax.Instance.cross_entropy(), "Softmax_Instance_crossentropy")
    ]

    cce.DETECTIONS = [400, 400, 400]
    for cfg, name in configs:
        out_folder = Folder(f"./gt_evaluations/fl_{name}", create=True)
        predictor = get_predictor(cfg)
        detector = get_detector(model_id, model_checkpoint_folder)

        model = PredictorDetectorWrapper(predictor, detector)
        _, results = evaluate(model, ds_to_dataloader(gt_sparse, is_train=False), device=training_device())
        CocoResultsHandler(results, "calcein", "test_sparse", detections=cce.DETECTIONS).save(out_folder)

        _, results = evaluate(model, ds_to_dataloader(gt_dense, is_train=False), device=training_device())
        CocoResultsHandler(results, "calcein", "test_dense", detections=cce.DETECTIONS).save(out_folder)


elif eval_model == "bf":
    from vision.detection.det_brightfield.train_generated_dataset import model, model_id

    out_folder = Folder(f"./gt_evaluations/bf_{model_id}", create=True)

    cce.DETECTIONS = [400, 400, 400]

    _, results = evaluate(model, ds_to_dataloader(gt_sparse, is_train=False), device=training_device())
    CocoResultsHandler(results, "calcein", "test_sparse", detections=cce.DETECTIONS).save(out_folder)

    _, results = evaluate(model, ds_to_dataloader(gt_dense, is_train=False), device=training_device())
    CocoResultsHandler(results, "calcein", "test_dense", detections=cce.DETECTIONS).save(out_folder)
