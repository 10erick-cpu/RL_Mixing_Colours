import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from torchvision.transforms import ToTensor

from utils.helper_functions.misc_utils import RunningMean


class SimulatorEngine(object):

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("base")

    # @timed_execution
    def update(self, mat, lim=255, max_iter=5):

        mat = self(mat)

        iter_count = 0
        if 0 < lim < mat.max():
            while mat.max() > lim:
                if iter_count > max_iter:
                    # raise ValueError("Max iterations reached, grid overflow")
                    return mat
                mat = self(mat)
                iter_count += 1
        # print("engine iter", iter_count)
        return mat


class BasicMeanEngine(SimulatorEngine):

    def __call__(self, mat, *args, **kwargs):
        out = np.zeros_like(mat)

        out[:, :] = mat.mean(axis=(0, 1))
        return out


class ConvolveInterpolateEngine(SimulatorEngine):
    def __init__(self, kernel_size=3, device=None, cpu_conversion=True):
        super(ConvolveInterpolateEngine, self).__init__()
        self.sim_device = device if device is not None else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.kernel = None
        self.setup(kernel_size)
        self.conv_cpu = cpu_conversion

    def setup(self, kernel_size):
        self.kernel = torch.ones((3, 1, kernel_size, kernel_size)) * (1 / (kernel_size * kernel_size))
        self.kernel = self.kernel.to(self.sim_device)

    def __call__(self, mat, *args, **kwargs):
        if not (self.kernel.shape[2] < mat.shape[0] and self.kernel.shape[3] < mat.shape[1]):
            raise ValueError(
                "invalid kernel shape {} for mat shape {}".format(str(self.kernel.shape), str(mat.shape)))
        with torch.no_grad():
            if self.conv_cpu or not isinstance(mat, torch.Tensor):
                cuda_tens = ToTensor()(mat).float()
                cuda_tens.requires_grad = False
                cuda_tens = cuda_tens.to(self.sim_device)
            else:
                cuda_tens = mat
                cuda_tens = cuda_tens.permute(2, 0, 1)
            cuda_tens = cuda_tens[None, :, :, :]

            cuda_tens = F.conv2d(cuda_tens, self.kernel, groups=3, stride=1)

            cuda_tens = F.interpolate(cuda_tens, size=(mat.shape[0], mat.shape[1]), mode='nearest'
                                      )

            cuda_tens = cuda_tens.permute(0, 2, 3, 1)
            # print(cuda_tens.sum().item())
            # print(mat.sum().item())
            # print()

            if self.conv_cpu:
                result = cuda_tens[0].cpu().numpy()
            else:
                result = cuda_tens[0]

            return result


class BlurEngine(SimulatorEngine):
    def __init__(self, kernel_size):
        self.kernel = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

    def __call__(self, mat, *args, **kwargs):
        cv2.blur(mat, ksize=self.kernel, dst=mat)
        return mat


class GaussianBlurEngine(SimulatorEngine):
    def __init__(self, update_weight=1, sigma=(0.5, 1, 0)):
        self.sigma = sigma
        self.runtime = RunningMean()
        self.update_weight = update_weight

    def __call__(self, mat, *args, **kwargs):
        start = time.time()
        dst = gaussian_filter(mat, sigma=self.sigma, order=0, mode="reflect")

        self.runtime.update(time.time() - start)
        if self.update_weight == 1:
            return dst

        return (1 - self.update_weight) * mat + self.update_weight * dst


class ConvolveEngine(SimulatorEngine):
    def __init__(self, kernel_size=3):
        self.kernel = torch.ones((kernel_size, kernel_size)) * (1 / (kernel_size * kernel_size))
        self.kernel = self.kernel.numpy()

    def __call__(self, mat, *args, **kwargs):
        kernel = np.array([
            [1 / 8, 1 / 8, 1 / 8],
            [1 / 8, 0.1, 1 / 8],
            [1 / 8, 1 / 8, 1 / 8]])

        # kernel = np.ones_like(kernel)
        # kernel *= 1 / 9
        for i in range(mat.shape[2]):
            mat[:, :, i] = convolve2d(mat[:, :, i], self.kernel, mode="same", boundary='sym')
        return mat
