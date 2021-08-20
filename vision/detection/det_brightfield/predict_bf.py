import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from utils.helper_functions.img_utils import load_image
from utils.models.folder import Folder
from vision.detection.det_fluorescence.nuclei_pred_to_dic_masks import get_detector
from vision.detection.model_setup.mrcnn_wrapper import MRCNNDetector


def detect_path(detector, path):
    img = load_image(path, force_8bit=True, force_grayscale=True)

    img = torch.from_numpy(img).float().squeeze()

    img = (img - img.min()) / (img.max() - img.min())
    img = (img - img.mean()) / img.std()

    result = detector.detect_and_create_overlay(img.unsqueeze(0).unsqueeze(0))
    output = result.output
    overlay = result.overlay

    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(img.squeeze().numpy(), cmap="gray")
    ax[1].imshow(img.squeeze().numpy(), cmap="gray")
    ax[1].imshow(overlay.squeeze(), alpha=0.3)
    path = Folder("./bf_det_results", create=True).get_file_path(f"{np.random.randint(0, 1000)}.png")
    plt.tight_layout()
    sns.despine(f, left=True, bottom=True)
    f.savefig(path, dpi=200)
    plt.close()


def detect_folder(detector, path, extensions=None, contains=None, subdirs=True):
    out_folder = Folder("./pipeline_results")
    out_folder = out_folder.make_sub_folder(Folder(path).name)

    for idx, target in enumerate(
            Folder(path).make_file_provider(extensions=extensions, contains=contains, include_subdirs=subdirs)):
        detect_path(detector, target)


def detect_ds(detector: MRCNNDetector, ds, max_dets=10):
    count = 0
    for img, target in reversed(ds):
        count += 1
        result = detector.detect_and_create_overlay(img.unsqueeze(0))
        output = result.output
        overlay = result.overlay

        f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(img.squeeze().numpy(), cmap="gray")
        ax[1].imshow(img.squeeze().numpy(), cmap="gray")
        ax[1].imshow(overlay.squeeze(), alpha=0.3)
        path = Folder("./bf_det_results", create=True).get_file_path(f"{np.random.randint(0, 10000)}.png")
        plt.tight_layout()
        sns.despine(f, left=True, bottom=True)

        f.savefig(path, dpi=200)
        plt.close()
        if count > max_dets:
            break


from vision.detection.det_brightfield.train_generated_dataset import model_id, model_checkpoint_folder, all_ds

detector = get_detector(model_id, checkpoint_folder=model_checkpoint_folder)
detect_ds(detector, all_ds['calcein']['test'])

# detect_folder("/mnt/unix_data/datastorage/ztz_datasets/00_clean_ds/190305_IBIDI_JIMT1_CT_HOE", contains=['DIC'])

# detect_folder("/mnt/unix_data/datastorage/ztz_datasets/00_clean_ds/190319_JIMT1_CT_HOE_contam", contains=['DIC'])
# detect_folder(detector, "/mnt/unix_data/datastorage/ztz_analysis/dario_31.07/JIMT-1 Tubb")
# detect_folder("/mnt/unix_data/datastorage/ztz_datasets/00_clean_ds/190221_JIMT1_CT_HOE/20x", contains=['DIC'])
