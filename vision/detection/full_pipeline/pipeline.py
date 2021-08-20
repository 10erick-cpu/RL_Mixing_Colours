import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd.grad_mode import no_grad

from utils.helper_functions.img_utils import gray2rgb, load_image
from utils.helper_functions.misc_utils import timestamp_now_str
from utils.helper_functions.patch_extractor import PatchAnalyzer
from utils.helper_functions.torch_utils import training_device
from utils.models.folder import Folder

from vision.detection.model_setup.detection_utils import do_multi_detection, do_single_detection
from vision.detection.model_setup.mrcnn_wrapper import ModelConfig, MRCNNDetector
from vision.prediction.settings.model_configurations import Sigmoid

torch.manual_seed(1)

detection_config = ModelConfig()

detection_config.BOX_HEAD_NMS = 0.25
detection_config.BOX_HEAD_OBJ_TH = 0.01
detection_config.RPN_NMS_THRESH = 0.25

detector = MRCNNDetector("mrcnntv_fl_detection").load(detection_config)

cache = dict()

params = Sigmoid.Instance.bce()
unet_model = params.get_model()
unet_model.load(params)
unet_model.eval()


def draw_result(img, fl_out, detection_overlay, interactive=False, save_path=None):
    if 'draw' not in cache:
        f, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 8))
        cache['draw'] = f, ax

    f, ax = cache['draw']
    plt.cla()
    ax[0][0].imshow(img, cmap="gray")
    ax[0][1].imshow(img, cmap="gray")
    ax[0][1].imshow(detection_overlay, alpha=0.3)

    ax[1][0].imshow(fl_out, cmap="gray")
    ax[1][1].imshow(fl_out, cmap="gray")
    ax[1][1].imshow(detection_overlay, alpha=0.5)
    plt.tight_layout()
    plt.title(params.get_tensor_board_id())

    if save_path:
        f.savefig(save_path, dpi=300)
    if interactive:
        plt.ion()
        plt.pause(1)
    else:
        plt.show()
    plt.close(f)
    del cache['draw']


@no_grad()
def predict_fluorescence(brightfield_in, bg_th=0.0, do_tile=False, std_norm=False, batch_size=20, show_batch=False):
    if do_tile:

        data = torch.from_numpy(brightfield_in.squeeze())[None, None, :, :].float()

        # data = (data - data.mean()) / data.std()

        def net_fn(inp):

            result = unet_model.eval().to(training_device())(inp.to(training_device())).detach().cpu()
            if show_batch:
                f, ax = plt.subplots(batch_size, 2)
                for i in range(batch_size):
                    ax[i][0].imshow(inp[i].squeeze().numpy(), cmap="gray")
                    ax[i][1].imshow(result[i].squeeze().numpy(), cmap="gray")
                plt.show()
            return result

        analyzer = PatchAnalyzer(batch_size=batch_size, patch_h=196, patch_w=256, stride_h=196,
                                 stride_w=256)
        fl_out = analyzer(data, net=net_fn)
        fl_out = (fl_out - fl_out.min()) / (fl_out.max() - fl_out.min())
        fl_out = fl_out.numpy()

        return fl_out


    else:
        # fl_out = unet_model.to(training_device()).predict_raw_image(brightfield_in, return_tensor=False,
        #                                                             std_norm=std_norm)

        inp = torch.from_numpy(brightfield_in.squeeze()).float()[None, None]

        print("bf vals", inp.max(), inp.min())

        # inp = MinMaxTargetPreprocessor.normalize_mean(inp)
        fl_out = unet_model.to(training_device())(inp.to(training_device())).detach()
        # plt.imshow(fl_out, cmap="gray")
        # plt.show()

        # fl_out[fl_out < bg_th] = 0
        fl_out = fl_out.cpu().numpy().squeeze()

        fl_out = (fl_out - fl_out.min()) / (fl_out.max() - fl_out.min())
        # fl_out = torch.nn.Sigmoid()(fl_out)
        # fl_out[fl_out >= 0.7] = 1
        print(fl_out.min(), fl_out.max())
        fl_out -= fl_out.mean()
        return fl_out


def detect_fluorescence(fl_in):
    return detector.detect_numpy(fl_in)


def detect_path(path, use_multi=False):
    unet_model.eval()

    img = load_image(path, force_8bit=True, force_grayscale=True)

    # img = histogram_equalization(img)

    fl_out = predict_fluorescence(img, do_tile=False)

    # detections = detect_fluorescence(fl)

    fl_in = gray2rgb(fl_out).transpose((2, 0, 1))

    fl_in = torch.from_numpy(fl_in)

    # fl_in = (fl_in - fl_in.min()) / (fl_in.max() - fl_in.min())

    # fl_in = (fl_in - fl_in.min()) / (fl_in.std())

    # fl_in[fl_in>0.6] = 0.6
    if use_multi:
        c_overlay = do_multi_detection(fl_in, detector, confidence_th=0.66, mask_th=0.5)
    else:
        c_overlay = do_single_detection(fl_in, detector, confidence_th=0.5, mask_th=0.5)

    return img, fl_out, c_overlay


def detect_folder(path, extensions=None, contains=None, subdirs=True, save=False, max_detections=10):
    out_folder = Folder("./pipeline_results", create=True)
    out_folder = out_folder.make_sub_folder(params.get_tensor_board_id())
    if save:
        out_folder = out_folder.make_sub_folder(timestamp_now_str() + "_" + Folder(path).name)
    else:
        out_folder = None
    files = Folder(path).make_file_provider(extensions=extensions, contains=contains, include_subdirs=subdirs)

    if max_detections:
        files = np.random.choice(files, size=max_detections)

    for idx, target in enumerate(files):
        if "type=Calcein" in target or "type=Hoechst" in target or "type=hoechst" in target:
            continue
        out_path = out_folder.get_file_path(f"result{idx}.png") if out_folder else None
        draw_result(*detect_path(target),
                    interactive=False, save_path=out_path)


if __name__ == '__main__':
    # detect_folder("/mnt/unix_data/datastorage/ztz_datasets/00_clean_ds/190305_IBIDI_JIMT1_CT_HOE", contains=['DIC'])

    # detect_folder("/mnt/unix_data/datastorage/ztz_datasets/00_clean_ds/190319_JIMT1_CT_HOE_contam", contains=['DIC'])

    # detect_folder("/mnt/unix_data/datastorage/ztz_datasets/00_clean_ds/calcein_dense/cleaned_data", contains=['DIC'])
    # detect_folder("/mnt/unix_data/datastorage/ztz_datasets/00_clean_ds/190221_JIMT1_CT_HOE/20x", contains=['DIC'])
    # detect_folder("/mnt/unix_data/datastorage/ztz_datasets/01_clean_ds_v2/190305_IBIDI_JIMT1_CT_HOE/train_test_split/test/fullres", contains=['DIC'])
    # detect_folder("/mnt/unix_data/datastorage/ztz_datasets/01_clean_ds_v2/calcein/train_test_split/test/fullres",
    #              contains=['DIC'], save=False, max_detections=None)

    detect_folder("/mnt/unix_data/datastorage/ztz_analysis/dario_31.07/JIMT-1 Tubb", save=False, max_detections=None)

    # detect_folder("/mnt/unix_data/datastorage/ztz_datasets/01_clean_ds_v2/calcein/train_test_split/test/crops",
    #              contains=['DIC'], save=True, max_detections=30)
