import matplotlib.pyplot as plt
import torch

from utils.helper_functions.img_utils import load_image
from utils.helper_functions.torch_utils import training_device
from utils.models.folder import Folder

from vision.detection.model_setup.mrcnn_wrapper import ModelConfig, MRCNNDetector


def get_detector(model_id, checkpoint_folder):
    detection_config = ModelConfig()

    detection_config.BOX_HEAD_NMS = 0.25
    detection_config.BOX_HEAD_OBJ_TH = 0.5
    detection_config.RPN_NMS_THRESH = 0.7

    detector = MRCNNDetector(model_id, cp_folder=checkpoint_folder).load(detection_config)
    detector.model = detector.model.to(training_device())
    detector.model.eval()
    return detector


from vision.detection.det_brightfield.train_generated_dataset import model_id, model_checkpoint_folder

detector = get_detector(model_id, model_checkpoint_folder)

target_dir = "/mnt/unix_data/datastorage/dario/Image Analysis"

target_dir = Folder(target_dir)

files = list(target_dir.make_file_provider(extensions=["tif"], include_subdirs=True))
print(len(files))
print(files)

data = []

for idx, f in enumerate(files):
    print(f"{idx + 1}/{len(files)}")

    img = load_image(f, force_grayscale=True, force_8bit=True)
    display_img = img.copy()
    img = torch.from_numpy(img).float()
    img = (img - img.mean()) / img.std()
    result_dict = detector.detect_and_create_overlay(img.unsqueeze(0).unsqueeze(0))
    overlay = result_dict['overlay']
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))

    ax[0].imshow(display_img, cmap="gray")
    ax[1].imshow(display_img, cmap="gray")
    ax[1].imshow(overlay, alpha=0.2)
    multimask = result_dict['output'][3]
    multimask[multimask > 0] = 1
    ax[2].imshow(multimask, cmap="gray")
    fig.savefig(f.replace(".tif", "_analyzed.png"), dpi=150)
    plt.close(fig)
    # plt.show()

    assert multimask.max() == 1, multimask.min() == 0
    area_cell = multimask.numpy().sum()
    print(multimask.shape)
    area_total = multimask.shape[0] * multimask.shape[1]
    area_cell_total = area_cell / area_total
    area_cell_perc = area_cell_total * 100
    print(area_cell, area_total, area_cell / area_total)

    data.append({'file': f, 'img_name': f.split("/")[-1], "pix_cell": area_cell, "pix_total": area_total,
                 'area_cell': area_cell_total, 'area_cell_perc': area_cell_perc,
                 "cell_count": result_dict['valid_det']})
    print(data[-1])

import pandas as pd

data = pd.DataFrame(data)

data.to_csv(target_dir.get_file_path("analysis_result.csv"))
print(data)
