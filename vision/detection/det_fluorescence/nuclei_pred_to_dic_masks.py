from collections import deque

import matplotlib.pyplot as plt
import torch
from torch import no_grad

import numpy as np

from utils.file_management.naming_strategies import MetadataParser, CsvNamingStrategy
from utils.helper_functions.img_utils import rgb2gray, gray2rgb
from utils.helper_functions.torch_utils import training_device
from utils.helper_functions.filesystem_utils import Folder
from vision.detection.detection_datasets.fl_multimask_dataset import GeneratedImagesetDataset
from vision.detection.model_setup.detection_utils import draw_detection, threshold_mask
from vision.detection.model_setup.model_setup import get_transform
from vision.detection.model_setup.mrcnn_wrapper import ModelConfig, MRCNNDetector
from vision.prediction.settings.model_configurations import Raw, Sigmoid
from vision.prediction.settings.training_configuration import TrainParams

COLORS = [16, 32, 64, 96, 128, 160, 192, 220, 255]


def get_detector(model_id, checkpoint_folder):
    detection_config = ModelConfig()

    detection_config.BOX_HEAD_NMS = 0.25
    detection_config.BOX_HEAD_OBJ_TH = 0.5
    detection_config.RPN_NMS_THRESH = 0.7

    detector = MRCNNDetector(model_id, cp_folder=checkpoint_folder).load(detection_config)
    detector.model = detector.model.to(training_device())
    detector.model.eval()
    return detector


def get_predictor(predictor_params: TrainParams):
    predictor = predictor_params.get_model()
    predictor.load(predictor_params)
    predictor.eval()
    return predictor.to(training_device())


def find_image_sets(base_folder, naming_strategy, subdirs=False):
    parser = MetadataParser(naming_strategy)
    metadata = parser.extract_from(
        base_folder.make_file_provider(extensions=['tif', 'tiff', 'png'], include_subdirs=subdirs))

    image_sets = naming_strategy.identify_image_sets(metadata)

    return image_sets


def test(image_set):
    raw_instance_l1 = get_predictor(Raw.Instance.l1())
    sm_instance_l1 = get_predictor(Sigmoid.Instance.l1())
    sm_batch_l1 = get_predictor(Sigmoid.Batch.l1())
    sm_instance_bce = get_predictor(Sigmoid.Instance.bce())

    for im in image_set:
        img = torch.from_numpy(rgb2gray(im.dic)).float().unsqueeze(0).unsqueeze(0).to(training_device())

        r_inst_l1 = raw_instance_l1.to(training_device())(img).cpu().numpy().squeeze()
        sm_inst_l1 = sm_instance_l1.to(training_device())(img).to("cpu").cpu().numpy().squeeze()
        sm_btch_l1 = sm_batch_l1.to(training_device())(img).to("cpu").cpu().numpy().squeeze()
        sm_inst_bce = sm_instance_bce.to(training_device())(img).to("cpu").cpu().numpy().squeeze()

        f, ax = plt.subplots(1, 6, sharex=True, sharey=True, figsize=(30, 20))

        ax[0].imshow(im.dic, cmap="gray")
        ax[0].set_title("brightfield")


        ax[1].imshow(im.calcein, cmap="gray")
        ax[1].set_title("ground truth")
        ax[2].imshow(sm_inst_l1, cmap="gray")
        ax[2].set_title("sigmoid_instance_l1")
        ax[3].imshow(sm_inst_bce, cmap="gray")
        ax[3].set_title("sigmoid_instance_bce")
        ax[4].imshow(sm_btch_l1, cmap="gray")
        ax[4].set_title("sigmoid_batch_l1")
        ax[5].imshow(r_inst_l1, cmap="gray")
        ax[5].set_title("raw_instance_l1")


        for i in range(6):
            ax[i].axis("off")

        import seaborn as sns

        sns.despine(f)
        plt.tight_layout()


        out = Folder("/home/mrd/Desktop/comparison_predictors/sidebyside", create=True)
        f.savefig(out.get_file_path(f"{np.random.randint(0, 10000)}.png"), bbox_inches='tight', dpi=200)
        plt.close(f)






@no_grad()
def nuclei_prediction_to_masks(detector, image_sets, predictor=None, target_channel_name="hoechst", output_folder=None,
                               override=False, do_output=False, center_mrcnn_input=True, max_dets=10):
    # output_folder = None
    #test(image_sets)

    #raise EOFError
    image_sets = deque(image_sets)
    total_count = len(image_sets)
    count = 0
    for idx in range(total_count):
        count += 1
        ims = image_sets.popleft()
        print("Processing {}/{}".format(idx + 1, total_count))
        dic_meta = ims.channels["dic"]

        if output_folder and do_output:
            folder_name = dic_meta['base.fn'].item().split(".")[0].replace("_type=DIC", "")
            target_dir = output_folder.make_sub_folder(folder_name, create=False)

            if target_dir.exists():
                if not override:
                    print("skip folder", target_dir)
                    continue
            else:
                target_dir.make_dir()

        else:
            target_dir = None
        dic = ims.dic
        dic = gray2rgb(dic)

        if predictor is not None:
            predictor_in = torch.from_numpy(rgb2gray(dic)).float().unsqueeze(0).unsqueeze(0).to(training_device())

            detection_target = predictor(predictor_in).to("cpu")
        else:
            detection_target = ims.image(target_channel_name)
            detection_target = rgb2gray(detection_target)
            detection_target = torch.from_numpy(detection_target)

        if len(detection_target.shape) == 2:
            detection_target = detection_target.unsqueeze(0).unsqueeze(0)
        elif len(detection_target.shape) == 3:
            detection_target = detection_target.unsqueeze(0)

        detection_target = detection_target.float()
        # detection_target = (detection_target - detection_target.mean()) / detection_target.std()
        detection_target = (detection_target - detection_target.min()) \
                           / (detection_target.max() - detection_target.min())

        if center_mrcnn_input:
            detection_target = (detection_target - detection_target.mean()) / detection_target.std()

        scores, boxes, masks, multimask_out = detector.detect_and_build_mask(detection_target, mask_threshold=0.5)

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

        ax[0][0].imshow(dic, cmap="gray")
        ax[0][1].imshow(detection_target.squeeze().numpy(), cmap="gray")

        ax[1][0].imshow(dic, cmap="gray")
        ax[1][0].imshow(color_overlay, alpha=0.3)

        ax[1][1].imshow(detection_target.squeeze().numpy(), cmap="gray")
        ax[1][1].imshow(color_overlay, alpha=0.3)

        obj_meta = ims._clone_channel('dic')
        obj_meta['type'] = "multimask"
        ims.assign_channel("multimask", multimask_out.numpy().squeeze(), obj_meta)

        # for i in range(masks.shape[2]):
        #     obj_meta = ims._clone_channel('dic')
        #     object_id = str(i)
        #     obj_meta['o-id'] = object_id
        #     ims.assign_channel("obj-mask#" + object_id, masks[:, :, i], obj_meta)
        #
        # try:
        #

        import seaborn as sns
        sns.despine(f)
        plt.tight_layout()
        if target_dir:
            _, out_dir = ims.persist(CsvNamingStrategy(), target_dir)
            f.savefig(out_dir.get_file_path(target_dir.name + "_debug.png"))
        else:
            plt.show()
        plt.close(f)
        if count > max_dets:
            break


def test2():
    ims = CsvNamingStrategy().find_image_sets(Folder("./test_output"), subdirs=True)

    ds = GeneratedImagesetDataset(ims, transforms=get_transform(train=True))

    for img, mask in ds:
        print(img.shape)
        print(mask['masks'].shape)


def generate_dataset(predictor, detector, dataset_folder, output_dir):
    sets = CsvNamingStrategy().find_image_sets(dataset_folder, subdirs=True)

    nuclei_prediction_to_masks(detector, sets, predictor=predictor, target_channel_name="hoechst",
                               output_folder=output_dir,
                               do_output=True, center_mrcnn_input=True)


if __name__ == '__main__':
    # f = "/mnt/unix_data/datastorage/ztz_datasets/01_clean_ds_v2/calcein/train_test_split/test/fullres"

    calcein_base = "/mnt/unix_data/datastorage/ztz_datasets/01_clean_ds_v2/calcein/train_test_split/{}/fullres"
    hoechst_base = "/mnt/unix_data/datastorage/ztz_datasets/01_clean_ds_v2/190305_IBIDI_JIMT1_CT_HOE/train_test_split/{}/fullres"

    calcein_test = Folder(calcein_base.format("test"))
    calcein_train = Folder(calcein_base.format("train"))
    calcein_val = Folder(calcein_base.format("val"))

    hoechst_test = Folder(hoechst_base.format("test"))
    hoechst_train = Folder(hoechst_base.format("train"))
    hoechst_val = Folder(hoechst_base.format("val"))

    predictor_cfg = Sigmoid.Instance.l1()
    predictor = get_predictor(predictor_cfg)

    from vision.detection.det_fluorescence.train_fl_detection import model_id, model_checkpoint_folder

    detector = get_detector(model_id, model_checkpoint_folder)

    base = Folder("./generated_datasets/sigmoid_instance_l1_thesis", create=True)
    #base = Folder("./test_output/sigmoid_instance_bce", create=True)

    out_calcein_base = base.make_sub_folder("calcein")
    out_hoechst_base = base.make_sub_folder("hoechst")

    calcein = True
    hoechst = True

    if calcein:

        for name, f in zip(['test'], [calcein_test, calcein_val, calcein_train]):
            out_folder = out_calcein_base.make_sub_folder(name)
            print("Process folder", f, "out", out_folder)
            generate_dataset(predictor, detector, f, out_folder)

    if hoechst:
        for name, f in zip(['test'], [hoechst_test, hoechst_val, hoechst_train]):
            out_folder = out_hoechst_base.make_sub_folder(name)
            print("Process folder", f, "out", out_folder)
            generate_dataset(predictor, detector, f, out_folder)
