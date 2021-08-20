from multiprocessing import cpu_count

import torch

from utils.file_management.naming_strategies import CsvNamingStrategy

from utils.helper_functions.model_persistence import CheckpointFolder
from utils.helper_functions.torch_utils import training_device
from utils.models.folder import Folder
from utils.nn.cocoapi.custom import coco_utils
from utils.nn.cocoapi.custom.coco_results_handler import CocoResultsHandler
from utils.nn.cocoapi.custom.engine import evaluate, train_one_epoch
from vision.detection.det_fluorescence.nuclei_pred_to_dic_masks import get_detector
from vision.detection.detection_datasets.fl_multimask_dataset import GeneratedImagesetDataset
from vision.detection.model_setup.model_setup import get_transform, load_checkpoint


def ds_to_dataloader(ds, is_train, num_workers=cpu_count()):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
        collate_fn=coco_utils.collate_fn)


def ims_to_ds(root_dir, input_type, center_input, augment=True):
    ims = CsvNamingStrategy().find_image_sets(root_dir, subdirs=True)
    return GeneratedImagesetDataset(ims, get_transform(True), input_image=input_type, center_input=center_input,
                                    augment=augment)


def get_datasets(root_folder: Folder, input_type="dic", augment_train=True):
    calcein_base = root_folder.make_sub_folder("calcein", create=False)
    hoechst_base = root_folder.make_sub_folder("hoechst", create=False)

    calc_test = calcein_base.make_sub_folder("test", create=False)
    calc_train = calcein_base.make_sub_folder("train", create=False)
    calc_val = calcein_base.make_sub_folder("val", create=False)

    hoechst_test = hoechst_base.make_sub_folder("test", create=False)
    hoechst_train = hoechst_base.make_sub_folder("train", create=False)
    hoechst_val = hoechst_base.make_sub_folder("val", create=False)

    calc_train = ims_to_ds(calc_train, input_type, center_input=True, augment=augment_train)
    hoechst_train = ims_to_ds(hoechst_train, input_type, center_input=True, augment=augment_train)

    calc_test = ims_to_ds(calc_test, input_type, center_input=True, augment=False)
    hoechst_test = ims_to_ds(hoechst_test, input_type, center_input=True, augment=False)

    calc_val = ims_to_ds(calc_val, input_type, center_input=True, augment=False)
    hoechst_val = ims_to_ds(hoechst_val, input_type, center_input=True, augment=False)

    train = calc_train + hoechst_train

    test = calc_test + hoechst_test
    val = calc_val + hoechst_val

    return train, val, test, {'calcein': {'train': calc_train, 'val': calc_val, 'test': calc_test},
                              'hoechst': {'train': hoechst_train, 'val': hoechst_val, 'test': hoechst_test}}


# split the dataset in train and test set
torch.manual_seed(1)
batch_size = 3

model_id = "bf_detection_sigmoid_instance_l1_adam_lr0002"

model_checkpoint_folder = CheckpointFolder(
    "/home/mrd/Desktop/unix_data/datastorage/networks/mrcnn_torchvision/det_brightfield/checkpoints/bf_detection", strict_exist=False)

# get the model using our helper function
model = get_detector(model_id=model_id, checkpoint_folder=model_checkpoint_folder).model

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.0005,
#                             momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.Adam(params, lr=0.0002, weight_decay=0.0005)
# and a learning rate scheduler which decreases the learning rate by
# 10x every 5 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=30,
                                               gamma=0.1)

model, epoch = load_checkpoint(model_checkpoint_folder, model_id, model)

device = training_device()

model = model.to(device)

out_folder = model_checkpoint_folder.make_sub_folder(f"results_{model_id}")


def do_eval(model, eval_train=False):
    import utils.nn.cocoapi.custom.custom_coco_eval as cce
    cce.DETECTIONS = [300, 300, 300]

    train, val, test, data_dict = get_datasets(ds_root_folder, augment_train=False)
    if eval_train:
        print("Eval train hoechst")
        _, results = evaluate(model, ds_to_dataloader(data_dict['hoechst']['train'], is_train=False), device=device)
        CocoResultsHandler(results, "hoechst", "train", detections=cce.DETECTIONS).save(out_folder)
        print("Eval train calcein")
        _, results = evaluate(model, ds_to_dataloader(data_dict['calcein']['train'], is_train=False), device=device)
        CocoResultsHandler(results, "calcein", "train", detections=cce.DETECTIONS).save(out_folder)

    print("Eval val calcein")
    _, results = evaluate(model, ds_to_dataloader(data_dict['calcein']['val'], is_train=False), device=device)
    CocoResultsHandler(results, "calcein", "val", detections=cce.DETECTIONS).save(out_folder)

    print("Eval val hoechst")
    _, results = evaluate(model, ds_to_dataloader(data_dict['hoechst']['val'], is_train=False), device=device)
    CocoResultsHandler(results, "hoechst", "val", detections=cce.DETECTIONS).save(out_folder)

    print("Eval test hoechst")
    _, results = evaluate(model, ds_to_dataloader(data_dict['hoechst']['test'], is_train=False), device=device)
    CocoResultsHandler(results, "hoechst", "test", detections=cce.DETECTIONS).save(out_folder)
    print("Eval test calcein")
    _, results = evaluate(model, ds_to_dataloader(data_dict['calcein']['test'], is_train=False), device=device)
    CocoResultsHandler(results, "calcein", "test", detections=cce.DETECTIONS).save(out_folder)


if __name__ == '__main__':

    ds_root_folder = Folder("./generated_datasets/sigmoid_instance_l1")
    train_ds, val_ds, test_ds, all_ds = get_datasets(ds_root_folder)
    # define training and validation data loaders
    data_loader = ds_to_dataloader(train_ds, True)

    data_loader_test = ds_to_dataloader(val_ds, False)

    num_epochs = 60

    do_eval(model, eval_train=False)

    for epoch_id in range(epoch + 1, num_epochs):
        print("Training")
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch_id, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if epoch_id > 0 and epoch_id % 100 == 0:
            evaluate(model, data_loader_test, device=device)
            model.train()

        model_checkpoint_folder.save_training_progress(model_id, epoch_id, model)
        model_checkpoint_folder.cleanup_checkpoints(model_id, num_keep=3)
    do_eval(model, eval_train=True)
