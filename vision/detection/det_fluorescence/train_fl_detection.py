from multiprocessing import cpu_count

import torch

from utils.helper_functions.model_persistence import CheckpointFolder
from utils.helper_functions.torch_utils import training_device
from utils.nn.cocoapi.custom import coco_utils
from utils.nn.cocoapi.custom.coco_results_handler import CocoResultsHandler
from utils.nn.cocoapi.custom.engine import evaluate
from vision.detection.det_fluorescence.nuclei_pred_to_dic_masks import get_detector
from vision.detection.detection_datasets.fl_multimask_dataset import KaggleDs, BBBC006, BBBC039
from vision.detection.model_setup.model_setup import get_transform, load_checkpoint


def ds_to_dataloader(ds, is_train, num_workers=cpu_count()):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
        collate_fn=coco_utils.collate_fn)


train_ds = KaggleDs(get_transform(train=True), max_gt_instances=1000) + BBBC006(get_transform(train=True),
                                                                                max_gt_instances=1000)

test_ds = BBBC039(get_transform(train=False), 1000)

# split the dataset in train and test set
torch.manual_seed(1)
batch_size = 5
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=cpu_count(),
    collate_fn=coco_utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, num_workers=cpu_count(),
    collate_fn=coco_utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_id = "fl_detection_adam"

model_checkpoint_folder = CheckpointFolder("/home/mrd/Desktop/unix_data/datastorage/networks/mrcnn_torchvision/det_fluorescence/checkpoints/fl_detection", strict_exist=False)

# get the model using our helper function
model = get_detector(model_id=model_id, checkpoint_folder=model_checkpoint_folder).model

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.0005,
#                             momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0005)
# and a learning rate scheduler which decreases the learning rate by
# 10x every 5 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=20,
                                               gamma=0.1)

model, epoch = load_checkpoint(model_checkpoint_folder, model_id, model)

model = model.to(training_device())


def do_eval(model, eval_train=False):
    import utils.nn.cocoapi.custom.custom_coco_eval as cce
    cce.DETECTIONS = [300, 300, 300]

    out_folder = model_checkpoint_folder.make_sub_folder(f"results_{model_id}")

    if eval_train:
        print("Eval train KaggleDs")
        _, results = evaluate(model, ds_to_dataloader(KaggleDs(get_transform(train=True), max_gt_instances=1000),
                                                      is_train=False), device=device)
        CocoResultsHandler(results, "KaggleDs", "train", detections=cce.DETECTIONS).save(out_folder)
        print("Eval train BBBC006")
        _, results = evaluate(model, ds_to_dataloader(BBBC006(get_transform(train=True),
                                                              max_gt_instances=1000), is_train=False), device=device)
        CocoResultsHandler(results, "BBBC006", "train", detections=cce.DETECTIONS).save(out_folder)

    print("Eval val BBBC039")
    _, results = evaluate(model,
                          ds_to_dataloader(BBBC039(get_transform(train=False), max_gt_instances=1000), is_train=False),
                          device=device)
    CocoResultsHandler(results, "BBBC039", "val", detections=cce.DETECTIONS).save(out_folder)


if __name__ == '__main__':

    # let's train it for 10 epochs

    do_eval(model, eval_train=False)
    raise EOFError
    num_epochs = 50

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
    print("Eval Train")
    evaluate(model, data_loader, device=device)
    print("Eval Test")
    evaluate(model, data_loader_test, device=device)
