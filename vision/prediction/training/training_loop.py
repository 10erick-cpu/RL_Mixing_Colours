import math
import sys

import torch
from torch import no_grad
from torch.utils.tensorboard import SummaryWriter

from utils.helper_functions.torch_utils import training_device, get_data_loader
from utils.helper_functions.validation_utils import input_output_target_to_grid
from utils.nn.cocoapi.custom import coco_utils
from vision.prediction.settings.training_configuration import TrainParams
from vision.prediction.training.data_processors import MinMaxTargetPreprocessor

validation_activation_dict = {'l1': 'sigmoid',
                              'l2': 'sigmoid',
                              'cross_entropy': 'softmax',
                              'ssim': 'sigmoid',
                              'bce': 'sigmoid'}


def train(params: TrainParams,
          ds_train, ds_test,
          summary_writer=None, preprocessor=None, postprocessor=None,
          device=training_device(), save_interval=10, num_workers=0, debug_eval_interval=1):
    cp_folder = params.get_checkpoint_folder()

    model = params.get_model().to(device)

    optimizer = params.get_optimizer(model)

    step_lr = params.get_step_lr(optimizer)

    if summary_writer is None:
        summary_writer = SummaryWriter(comment=params.get_tensor_board_id(), flush_secs=5)

    epoch = 0
    if params.model.load:
        cp = cp_folder.load_last_training_checkpoint(params.model.id, strict=False)
        if cp is not None:
            model.load_state_dict(cp['model_state_dict'])
            epoch = cp['epoch']
            if f'optimizer_state_dict_{optimizer.__class__.__name__}' in cp:
                optimizer.load_state_dict(cp[f'optimizer_state_dict_{optimizer.__class__.__name__}'])

    debug_eval_indices = torch.arange(len(ds_test))
    debug_eval_indices = torch.cat([debug_eval_indices[:5], debug_eval_indices[-5:]])

    for i in range(epoch, params.training.num_epochs):

        # Train one epoch
        train_one_epoch(model, optimizer,
                        get_data_loader(ds_train, batch_size=params.training.batch_size, num_workers=num_workers),
                        device, i + 1,
                        preprocessor=preprocessor, postprocessor=postprocessor,
                        summary_writer=summary_writer,
                        steps_per_epoch=params.training.steps_per_epoch)

        # Validate
        validate(model, params,
                 data_loader=get_data_loader(ds_test, batch_size=params.validation.batch_size, num_workers=num_workers),
                 device=device, epoch=i + 1, preprocessor=preprocessor,
                 summary_writer=summary_writer,
                 steps_per_epoch=params.validation.steps_per_epoch)

        if i % debug_eval_interval == 0:
            debug_evaluate(model=model, dataset=ds_test, device=device, epoch=i + 1, eval_indices=debug_eval_indices,
                           preprocessor=preprocessor, summary_writer=summary_writer)

        # Save
        if i > 0 and (i + 1) % save_interval == 0:
            cp_folder.save_training_progress(params.model.id, i + 1, model, optimizer)
            cp_folder.cleanup_checkpoints(params.model.id, num_keep=3)
        if step_lr is not None:
            step_lr.step()

    summary_writer.close()
    model = model.to("cpu")

    del model
    del optimizer


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, preprocessor=None,
                    postprocessor=None,
                    summary_writer=None, steps_per_epoch=-1):
    model.train()
    metric_logger = coco_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', coco_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = coco_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for step, (images, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if preprocessor:
            images, targets = preprocessor(images, targets)

        out, loss = model.learn(images.to(device), targets.to(device))

        if postprocessor is None:
            loss = loss.mean()
        else:
            loss = postprocessor(loss, out, targets.to(device))

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)

        if 0 < steps_per_epoch < step:
            break
    # print("write output", epoch,  metric_logger.meters['loss'].avg, "train-" + criterion.__class__.__name__)
    # print("lr", optimizer.param_groups[0]["lr"], epoch)
    summary_writer.add_scalar("train-" + model.train_criterion.__class__.__name__, metric_logger.meters['loss'].avg,
                              epoch)
    summary_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)


@no_grad()
def validate(model, params, data_loader, device, epoch, print_freq=10, preprocessor=None,
             summary_writer=None,
             steps_per_epoch=-1):
    model.eval()
    metric_logger = coco_utils.MetricLogger(delimiter="  ")

    for crit_name in params.validation.metrics:
        metric_logger.add_meter(crit_name, coco_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    if params.loss.name not in ['l1', 'l2', 'ssim']:
        metric_logger.add_meter("val_" + params.loss.name, coco_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Validate epoch: [{}]'.format(epoch)

    for step, (images, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if preprocessor:
            images, targets = preprocessor(images, targets)

        images = images.to(device)
        targets = targets.to(device)

        l1 = model.l1(images, targets).mean().item()
        l2 = model.l2(images, targets).mean().item()
        ssim = model.ssim(images, targets).mean().item()

        if params.loss.name not in ['l1', 'l2', 'ssim']:
            out, train_val = model.learn(images, targets)
            metric_logger.meters["val_" + params.loss.name].update(train_val.item())

        metric_logger.meters["l1"].update(l1)
        metric_logger.meters["l2"].update(l2)
        metric_logger.meters["ssim"].update(ssim)

        if 0 < steps_per_epoch < step:
            break

    for meter_name, meter in metric_logger.meters.items():
        summary_writer.add_scalar("val-" + meter_name, meter.avg, epoch)


@no_grad()
def debug_evaluate(model, dataset, device, epoch, eval_indices, preprocessor=None,
                   summary_writer=None, num_images=10):
    model.eval()

    eval_batch = get_data_loader(ds=dataset, idx=eval_indices, batch_size=num_images, shuffle=False)

    for step, (images, targets, _) in enumerate(eval_batch):

        if preprocessor:
            images, targets = preprocessor(images, targets)

        images = images.to(device)

        out = model(images)

        normed_images = MinMaxTargetPreprocessor.normalize_mean(images).cpu()
        normed_targets = MinMaxTargetPreprocessor.normalize_mean(targets).cpu()
        normed_out = MinMaxTargetPreprocessor.normalize_mean(out).cpu()
        grid = input_output_target_to_grid(normed_images, normed_out, normed_targets.float())

        summary_writer.add_image("debug_output", grid, epoch, dataformats='HWC')
