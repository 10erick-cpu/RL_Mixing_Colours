import copy

import torch
from torch.nn import L1Loss, MSELoss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import SGD, Adam

from utils.helper_functions.model_persistence import CheckpointFolder
from utils.models.dot_dict import DotDict
from vision.prediction.custom_loss.ssim_loss import SSIM
from vision.prediction.model.headless_unet import HeadlessUNet
from vision.prediction.model.network_heads import head_class_for_name


class TrainParams(object):

    def __init__(self, model_id, batch_size=1, num_epochs=100, training_max_steps=-1, validation_max_steps=-1,
                 head="none", cp_dir="./checkpoints"):
        self.optim = DotDict(
            {'name': 'adam',
             'params': DotDict(
                 {
                     'lr': 0.01, 'weight_decay': 0.0005, 'momentum': 0.9
                 }
             )
             }
        )
        self.step_lr = None
        self.model = DotDict(
            {'id': model_id,
             'checkpoint_base_dir': cp_dir,
             'load': True,
             'params': DotDict({'in_channels': 1,
                                'n_classes': 1,
                                'bilinear_upsampling': False,
                                'head': head,
                                'norm_type': 'batch'})}
        )
        self.training = DotDict(
            {'batch_size': batch_size, 'num_epochs': num_epochs, 'steps_per_epoch': training_max_steps}
        )
        self.validation = DotDict(
            {'batch_size': batch_size, 'steps_per_epoch': validation_max_steps, 'metrics': ['l1', 'ssim', 'l2']}
        )

        self.loss = DotDict({
            'name': 'l1'
        })

    def __add_param_to_str(self, key, src_dict, alt_key=None):
        if alt_key is None:
            return key + "=" + str(src_dict[key])
        else:
            return alt_key + "=" + str(src_dict[key])

    def get_tensor_board_id(self):
        name = f"{self.model.id}_{self.optim.name}"
        delim = "-"
        for key, value in self.optim.params.items():
            item = key + "=" + str(value)
            name = name + delim + item

        name += delim + self.__add_param_to_str("head", self.model.params, "activation")
        name += delim + self.__add_param_to_str("batch_size", self.training)
        name += delim + self.__add_param_to_str("norm_type", self.model.params)
        name += delim + self.__add_param_to_str("bilinear_upsampling", self.model.params)
        name += delim + self.__add_param_to_str("name", self.loss, "loss")
        # name += delim + self.__add_param_to_str("num_epochs", self.training)

        return name

    def get_checkpoint_folder(self, create=True):
        if self.model.id is None:
            raise ValueError("Model id cannot be none")
        root = CheckpointFolder(self.model.checkpoint_base_dir, strict_exist=False)
        return CheckpointFolder(root.make_sub_folder(self.get_tensor_board_id(), create=create).path(),
                                strict_exist=create)

    def load_latest_checkpoint(self):

        cp_folder = self.get_checkpoint_folder(create=False)
        if not cp_folder.exists():
            return None

        cp = cp_folder.load_last_training_checkpoint(self.model.id, strict=False)
        if cp is None:
            return None
        return cp, cp['epoch']

    def already_trained(self):
        cp = self.load_latest_checkpoint()
        if cp is None:
            return False

        _, epoch = cp
        if 0 < epoch < self.training.num_epochs:
            #raise ValueError("Configuration ", self.get_tensor_board_id(), "not fully trained, consider retraining")
            pass

        return epoch >= self.training.num_epochs

    def get_criterion(self, name=None, reduction="mean"):

        if name == "l1":
            return L1Loss(reduction=reduction)

        if name == "l2":
            return MSELoss(reduction=reduction)

        if name == "smooth_l1":
            return SmoothL1Loss(reduction=reduction)

        if name == "bce":
            return BCELoss(reduction=reduction)

        if name == "bcewithlogits":
            return BCEWithLogitsLoss(reduction=reduction)

        if name == "cross_entropy":
            return CrossEntropyLoss(reduction=reduction)

        if name == "ssim":
            return SSIM()

        raise ValueError("Unknown criterion", name)

    def get_model(self):
        net = HeadlessUNet(self.model.params)
        head = head_class_for_name(self.model.params.head)
        return head(net, self.get_criterion(self.loss.name))

        # activation = self.model.params.activation
        # if activation == "sigmoid":
        #     return SigmoidUNet(self.model.params)
        # elif activation == "softmax":
        #     return SoftmaxUNet(self.model.params)
        # return UNetWrapper(self.model.params)

    def get_step_lr(self, optim):
        if self.step_lr is None:
            return None
        return torch.optim.lr_scheduler.StepLR(optim, **self.step_lr)

    def get_optimizer(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        if self.optim.name.lower() == "sgd":
            return SGD(params, **self.optim.params)

        if self.optim.name.lower() == "adam":
            param_dict = copy.deepcopy(self.optim.params)
            del param_dict['momentum']
            return Adam(params, **param_dict)

        raise ValueError("Unsupported optimizer", self.optim.name.lower())