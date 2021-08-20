class CriterionPreprocessor(object):
    def __init__(self):
        super().__init__()

    def __call__(self, *input_data):
        return self.process(*self.prepare_data(input_data))

    def process(self, signal, target, class_labels):
        raise NotImplementedError()

    def prepare_data(self, ds_data):
        if len(ds_data) == 2:
            return ds_data[0], ds_data[1], None
        return ds_data[0], ds_data[1], ds_data[2]


class MinMaxTargetPreprocessor(CriterionPreprocessor):
    def __init__(self, img_op, target_op):
        super().__init__()
        self.img_op = img_op
        self.target_op = target_op

        print(f"MinMaxProcessor - Input image: mean: {self.img_op_mean()}, zscore: {self.img_op_zscore()}")
        print(f"MinMaxProcessor - Target: mean: {self.target_op_mean()}, zscore: {self.target_op_zscore()}")

    def img_op_mean(self):
        return self.img_op == "mean"

    def target_op_mean(self):
        return self.target_op == "mean"

    def img_op_zscore(self):
        return self.img_op == "z-score"

    def target_op_zscore(self):
        return self.target_op == "z-score"

    @staticmethod
    def do_z_score(batch):
        batch_size, channels = batch.shape[0], batch.shape[1]
        im_mean = batch.view(batch_size, channels, -1).mean(2).view(batch_size, channels, 1, 1)
        im_std = batch.view(batch_size, channels, -1).std(2).view(batch_size, channels, 1, 1)
        new_batch = (batch - im_mean) / (im_std + 1e-8)
        return new_batch

    @staticmethod
    def normalize_mean(batch):
        batch_size, channels = batch.shape[0], batch.shape[1]
        im_min = batch.view(batch_size, channels, -1).min(2)[0].view(batch_size, channels, 1, 1)
        im_max = batch.view(batch_size, channels, -1).max(2)[0].view(batch_size, channels, 1, 1)

        new_batch = (batch - im_min) / (im_max - im_min)
        return new_batch

    def process(self, signal, targets, class_labels):
        # assert labels.max() == 255, labels.max()
        batch_size, channels = targets.shape[0], targets.shape[1]

        if self.img_op_mean():
            signal = self.normalize_mean(signal)
        elif self.img_op_zscore():
            signal = self.do_z_score(signal)

        if self.target_op_mean():
            targets = targets.float()
            targets = self.normalize_mean(targets)
        elif self.target_op_zscore():
            targets = targets.float()
            targets = self.do_z_score(targets)

        return signal, targets


class CrossEntropyPreprocessor(MinMaxTargetPreprocessor):
    def process(self, signal, targets, class_labels):
        signal, target = super(CrossEntropyPreprocessor, self).process(signal, targets, class_labels)

        return signal, (target * 255).round().long()


class CriterionPostProcessor(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class L1PostProcessor(CriterionPostProcessor):

    def __call__(self, crit_output, net_out, target):
        valid = target <= 0.3
        upper_valid = target > 0.3

        num_valid = 1
        num_upper = 1

        if crit_output[valid].nelement():
            valid_mean = crit_output[valid].mean()
            # penalty = 0.5 * torch.exp(vals)
            remain_mean = crit_output[upper_valid].mean()
            # num_valid = crit_output[valid].nelement()
            # num_upper = crit_output[upper_valid].nelement()

            return 0.9 * valid_mean + 0.1 * remain_mean

        # crit_output = crit_output * (1 + target)

        return crit_output.mean()
