
import torch
import torch.nn as nn
from torch.autograd.grad_mode import no_grad
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.nn.modules.loss import SmoothL1Loss

from utils.helper_functions.torch_utils import training_device
from vision.prediction.custom_loss.ssim_loss import SSIM
from vision.prediction.model.headless_unet import HeadlessUNet
from vision.prediction.training.data_processors import MinMaxTargetPreprocessor


def head_class_for_name(name):
    if name == "sigmoid":
        return SigmoidHead
    if name == "softmax":
        return SoftmaxHead
    return RawHead


class NetworkHead(nn.Module):
    def __init__(self, net: HeadlessUNet, train_criterion, device=training_device()):
        super().__init__()
        assert train_criterion.__class__ in self.accepted_losses(), (self.__class__, train_criterion.__class__)
        self.net = net
        self.train_criterion = train_criterion
        self.val_ssim = SSIM(size_average=False)
        self.val_l1 = L1Loss(reduction="none")
        self.val_l2 = MSELoss(reduction="none")
        self.device = device

    def accepted_losses(self):
        raise NotImplementedError()

    def __repr__(self):
        headless = repr(self.net)
        return f"{headless}\nHead: {self.__class__.__name__} \n train loss: {self.train_criterion.__class__.__name__}"

    def _center_data(self, data):
        assert data.dtype == torch.float32
        return MinMaxTargetPreprocessor.do_z_score(data)

    def _norm_min_max_data(self, data):
        assert data.dtype == torch.float32
        return MinMaxTargetPreprocessor.normalize_mean(data)

    def _subtract_mean(self, data):
        assert data.dtype == torch.float32

        batch_size, channels = data.shape[0], data.shape[1]
        im_mean = data.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
        new_batch = (data - im_mean)
        return new_batch

    def learn(self, inputs, targets):
        raise NotImplementedError()

    @no_grad()
    def l1(self, inputs, targets):
        raise NotImplementedError()

    @no_grad()
    def l2(self, inputs, targets):
        raise NotImplementedError()

    @no_grad()
    def ssim(self, inputs, targets):
        raise NotImplementedError()

    def forward(self, input):
        return self.net(input)

    def load(self, params):
        epoch = 0
        cp_folder = params.get_checkpoint_folder(create=False)
        if not cp_folder.exists():
            return False

        if params.model.load:
            cp = cp_folder.load_last_training_checkpoint(params.model.id, strict=False)
            if cp is not None:
                print("CP loaded")
                self.load_state_dict(cp['model_state_dict'])
                epoch = cp['epoch']
                return True

        return False


class RawHead(NetworkHead):
    def accepted_losses(self):
        return [L1Loss, MSELoss, SmoothL1Loss]

    def __init__(self, net, criterion):
        super().__init__(net, criterion)
        self.do_centering = True

    def verify_targets(self, raw_targets):
        assert 0 <= raw_targets.min() <= raw_targets.max() <= 1 and raw_targets.dtype == torch.float32, (
            raw_targets.min(), raw_targets.max())

    def forward(self, input):
        return self.net(input)

    def learn(self, inputs, targets):
        out = self.net(inputs)
        targets = self._center_data(targets) if self.do_centering else targets
        return out, self.train_criterion(out, targets)

    def l1(self, inputs, raw_targets):
        self.verify_targets(raw_targets)
        out = self(inputs)
        targets = self._center_data(raw_targets) if self.do_centering else raw_targets
        r = self.val_l1(out, targets)
        return r

    def l2(self, inputs, raw_targets):
        self.verify_targets(raw_targets)
        out = self.net(inputs)
        targets = self._center_data(raw_targets) if self.do_centering else raw_targets
        return self.val_l2(out, targets)

    def ssim(self, inputs, targets):
        self.verify_targets(targets)
        out = self._norm_min_max_data(self.net(inputs))
        return self.val_ssim(out, targets)


class SigmoidHead(NetworkHead):
    def accepted_losses(self):
        return [L1Loss, MSELoss, BCELoss, SmoothL1Loss, BCEWithLogitsLoss]

    def __init__(self, net, criterion):
        super().__init__(net, train_criterion=criterion)
        self.sigmoid = torch.nn.Sigmoid()
        self.is_bce_with_logits = isinstance(criterion, BCEWithLogitsLoss)

    def verify_targets(self, raw_targets):
        assert 0 <= raw_targets.min() <= raw_targets.max() <= 1 and raw_targets.dtype == torch.float32, (
            raw_targets.min(), raw_targets.max())

    def forward(self, input):
        return self.sigmoid(self.net(input))

    def _process_targets(self, raw_targets):
        targets = self._subtract_mean(raw_targets)
        targets = self._norm_min_max_data(targets)
        return targets

    def learn(self, inputs, targets):
        self.verify_targets(targets)
        targets = self._process_targets(targets)
        if self.is_bce_with_logits:
            out = self.net(inputs)
        else:
            out = self(inputs)
        return out, self.train_criterion(out, targets)

    def l1(self, inputs, targets):
        self.verify_targets(targets)
        targets = self._process_targets(targets)
        out = self(inputs)
        return self.val_l1(out, targets)

    def l2(self, inputs, targets):
        self.verify_targets(targets)
        targets = self._process_targets(targets)
        out = self(inputs)
        return self.val_l2(out, targets)

    def ssim(self, inputs, targets):
        self.verify_targets(targets)
        targets = self._process_targets(targets)
        out = self(inputs)
        return self.val_ssim(out, targets)


class SoftmaxHead(NetworkHead):
    def accepted_losses(self):
        return [CrossEntropyLoss]

    def __init__(self, net, criterion):
        super().__init__(net, criterion)
        self.softmax = torch.nn.Softmax2d()

    def verify_targets(self, raw_targets):
        assert 0 <= raw_targets.min() <= raw_targets.max() <= 1 and raw_targets.dtype == torch.float32, (
            raw_targets.min(), raw_targets.max())

    def forward(self, input):
        return self._output_to_prediction(self.net(input))

    def _output_to_prediction(self, output):
        sm = self.softmax(output)
        result = sm.argmax(dim=1).float().unsqueeze(1)
        return result

    def _process_targets(self, raw_targets):
        targets = self._norm_min_max_data(raw_targets)
        targets = (targets * 255).round().long().squeeze()
        if len(targets.shape) == 2:
            targets = targets.unsqueeze(0)

        return targets

    def learn(self, inputs, targets):
        self.verify_targets(targets)
        out = self.net(inputs)
        targets = self._process_targets(targets)

        # print(targets.min(), targets.max(), targets)
        return out, self.train_criterion(out, targets)

    def l1(self, inputs, targets):
        self.verify_targets(targets)
        out = self(inputs).float()
        out = self._norm_min_max_data(out)

        return self.val_l1(out, targets)

    def l2(self, inputs, targets):
        self.verify_targets(targets)
        out = self(inputs).float()
        out = self._norm_min_max_data(out)

        return self.val_l2(out, targets)

    def ssim(self, inputs, targets):
        self.verify_targets(targets)
        out = self(inputs).float()
        out = self._norm_min_max_data(out)

        return self.val_ssim(out, targets)
