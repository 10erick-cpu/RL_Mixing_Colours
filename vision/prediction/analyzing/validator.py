import pandas as pd
from torch.autograd.grad_mode import no_grad

from utils.helper_functions.torch_utils import training_device, get_data_loader
from utils.nn.cocoapi.custom import coco_utils
from vision.prediction.settings.training_configuration import TrainParams


class Validator(object):
    @staticmethod
    def evaluate(params, dataset, steps_per_epoch=-1, batch_size=20, dataset_identifier=None, num_workers=4):
        ds_id = dataset_identifier if dataset_identifier is not None else dataset.__class__.__name
        results, logger = Validator(params, auto_load=True).eval_on_dataset(dataset, steps_per_epoch=steps_per_epoch,
                                                                            batch_size=batch_size,
                                                                            dataset_identifier=ds_id,
                                                                            num_workers=num_workers)

        results.to_csv(params.get_checkpoint_folder().get_file_path(f"validation_results_{ds_id}.csv"))

    def __init__(self, params: TrainParams, auto_load=False):
        super().__init__()
        self.params = params

        self.model = self.params.get_model()
        if auto_load:
            if not self.load_model():
                raise ValueError("Validation without loaded model")

    def load_model(self):
        return self.model.load(self.params)

    @no_grad()
    def eval_on_dataset(self, data_set, print_freq=10, preprocessor=None, steps_per_epoch=-1, batch_size=20,
                        device=training_device(), dataset_identifier=None, num_workers=4):
        self.model.eval()
        print(self.model)
        self.model = self.model.to(device)
        data_loader = get_data_loader(data_set, batch_size=batch_size, num_workers=num_workers)

        metric_logger = coco_utils.MetricLogger(delimiter="  ")

        metric_logger.add_meter("val_l1", coco_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter("val_l2", coco_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter("val_ssim", coco_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

        header = 'Validate model:'

        if dataset_identifier is None:
            dataset_identifier = data_set.__class__.__name__

        val_stats = []

        for step, (images, targets, debug_info) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            if preprocessor:
                images, targets = preprocessor(images, targets)

            images = images.to(device)
            targets = targets.to(device)
            l1 = self.model.l1(images, targets).mean(dim=(2, 3))
            l2 = self.model.l2(images, targets).mean(dim=(2, 3))
            ssim = self.model.ssim(images, targets)
            metric_logger.meters["val_l1"].update(l1.mean().item())
            metric_logger.meters["val_l2"].update(l2.mean().item())
            metric_logger.meters["val_ssim"].update(ssim.mean().item())

            for batch_idx in range(len(debug_info['ims'])):
                img_name = debug_info['ims'][batch_idx]
                val_stats.append({'img': img_name,
                                  'l1': l1[batch_idx].item(),
                                  'l2': l2[batch_idx].item(),
                                  'ssim': ssim[batch_idx].item(),
                                  'dataset_id': dataset_identifier})

            if 0 < steps_per_epoch < step:
                break

        self.model = self.model.to("cpu")
        return pd.DataFrame(val_stats), metric_logger
