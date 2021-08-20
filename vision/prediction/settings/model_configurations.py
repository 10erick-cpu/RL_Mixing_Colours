
import os

from vision.prediction.settings.training_configuration import TrainParams

EXP_NAME = "final_runs"
BATCH_SIZE = 40


def get_config(norm, loss, head, n_classes=1, exp_name=EXP_NAME):
    base_config = TrainParams(exp_name, batch_size=BATCH_SIZE, head=head,
                              training_max_steps=1100,
                              validation_max_steps=200,
                              cp_dir=f"{os.path.dirname(__file__)}/checkpoints_{exp_name}")
    print(base_config.model.checkpoint_base_dir)
    base_config.training.num_epochs = 30
    base_config.optim.params.lr = 0.001
    base_config.model.params.n_classes = n_classes
    base_config.model.params.norm_type = norm
    base_config.model.params.head = head
    base_config.loss.name = loss
    base_config.step_lr = {'step_size': 10, 'gamma': 0.2}
    return base_config


def default_configs(exp_name=EXP_NAME):
    configs = [
        Raw.Batch.l1(),
        Raw.Batch.l2(),
        Raw.Batch.smooth_l1(),
        Raw.Instance.l1(),
        Sigmoid.Instance.bce(),
        Sigmoid.Instance.bce_with_logits(),

        Raw.Instance.l2(),
        Raw.Instance.smooth_l1(),
        Sigmoid.Instance.l1(),
        Softmax.Instance.cross_entropy(),

        Sigmoid.Batch.bce(),
        Sigmoid.Batch.l1(),
        

        Softmax.Batch.cross_entropy()
    ]

    return configs


class Raw:
    _HEAD = "none"

    class Batch:
        _NORM = "batch"

        @classmethod
        def smooth_l1(cls):
            return get_config(cls._NORM, 'smooth_l1', Raw._HEAD)

        @classmethod
        def l1(cls):
            return get_config(cls._NORM, 'l1', Raw._HEAD)

        @classmethod
        def l2(cls):
            return get_config(cls._NORM, 'l2', Raw._HEAD)

    class Instance:
        _NORM = "instance"

        @classmethod
        def smooth_l1(cls):
            return get_config(cls._NORM, 'smooth_l1', Raw._HEAD)

        @classmethod
        def l1(cls):
            return get_config(cls._NORM, 'l1', Raw._HEAD)

        @classmethod
        def l2(cls):
            return get_config(cls._NORM, 'l2', Raw._HEAD)


class Sigmoid:
    _HEAD = "sigmoid"

    class Batch:
        _NORM = "batch"

        @classmethod
        def bce(cls):
            return get_config(cls._NORM, 'bce', Sigmoid._HEAD)

        @classmethod
        def bce_with_logits(cls):
            return get_config(cls._NORM, 'bcewithlogits', Sigmoid._HEAD)

        @classmethod
        def smooth_l1(cls):
            return get_config(cls._NORM, 'smooth_l1', Sigmoid._HEAD)

        @classmethod
        def l1(cls):
            return get_config(cls._NORM, 'l1', Sigmoid._HEAD)

        @classmethod
        def l2(cls):
            return get_config(cls._NORM, 'l2', Sigmoid._HEAD)

    class Instance:
        _NORM = "instance"

        @classmethod
        def bce(cls):
            return get_config(cls._NORM, 'bce', Sigmoid._HEAD)

        @classmethod
        def bce_with_logits(cls):
            return get_config(cls._NORM, 'bcewithlogits', Sigmoid._HEAD)

        @classmethod
        def smooth_l1(cls):
            return get_config(cls._NORM, 'smooth_l1', Sigmoid._HEAD)

        @classmethod
        def l1(cls):
            return get_config(cls._NORM, 'l1', Sigmoid._HEAD)

        @classmethod
        def l2(cls):
            return get_config(cls._NORM, 'l2', Sigmoid._HEAD)


class Softmax:
    _HEAD = "softmax"
    _N_CLASSES = 256
    _BSIZE=20

    class Batch:
        _NORM = "batch"

        @classmethod
        def cross_entropy(cls):
            cfg =  get_config(cls._NORM, 'cross_entropy', Softmax._HEAD, n_classes=Softmax._N_CLASSES)
            cfg.training.batch_size = Softmax._BSIZE
            cfg.validation.batch_size = Softmax._BSIZE
            return cfg

    class Instance:
        _NORM = "instance"

        @classmethod
        def cross_entropy(cls):
            cfg = get_config(cls._NORM, 'cross_entropy', Softmax._HEAD, n_classes=Softmax._N_CLASSES)
            cfg.training.batch_size = Softmax._BSIZE
            cfg.validation.batch_size = Softmax._BSIZE
            return cfg
