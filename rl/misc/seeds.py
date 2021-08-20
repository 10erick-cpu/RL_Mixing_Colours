import numpy as np
import tensorflow as tf
import torch


class Seeded(object):
    def __init__(self, np, tf=False, pytorch=False, seed=None):
        self.np_seed = np
        self.tf_seed = tf
        self.pytorch = pytorch
        self.user_seed = seed
        #self.global_seed = self.random_seed()

    def random_seed(self):
        return np.random.randint(2 ** 32 - 1)

    def set_seeds(self, seed):
        if self.np_seed:
            np.random.seed(seed)
        if self.tf_seed:
            tf.random.set_random_seed(seed)
        if self.pytorch:
            torch.random.manual_seed(seed)

    def __enter__(self):
        seed = self.user_seed if self.user_seed is not None else self.random_seed()
        self.set_seeds(seed)
        print("set seed", seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        #self.set_seeds(self.global_seed)
        #print("reset seed", self.global_seed)
        pass
