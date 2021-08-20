import os

from utils.helper_functions.img_utils import load_image
from utils.nn.cocoapi.custom import coco_utils
from vision.detection.model_setup.detection_utils import do_single_detection
import matplotlib.pyplot as plt
colors = [32, 64, 128, 184, 255]
metric_logger = coco_utils.MetricLogger(delimiter="  ")
for images, targets, paths in metric_logger.log_every(None, 1, "header"):
    input_image = images[0].clone()

    dic_path = paths[0]
    target_dir = os.path.dirname(dic_path)
    fname = os.path.basename(dic_path).replace("Hoechst", "DIC")

    dic_path = target_dir + "/" + fname

    print(dic_path)
    dic = load_image(dic_path, True, True)

    c_overlay = do_single_detection(input_image, 0.55, mask_th=0.5)

    plt.imshow(dic, cmap="gray")
    plt.imshow(c_overlay, alpha=0.3)

    plt.show()
