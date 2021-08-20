import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils.helper_functions.img_utils import load_image


class PatchExtractor(object):
    def __init__(self, patch_h, patch_w, stride_h, stride_w, adaptive_stride=False):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.adaptive_stride = adaptive_stride

    def pad_delta(self, data):
        h = data.shape[0] / (self.patch_h + self.stride_h)
        w = data.shape[1] / (self.patch_w + self.stride_w)
        h = np.ceil(h) * (self.patch_h + self.stride_h)
        w = np.ceil(w) * (self.patch_w + self.stride_w)
        pad_h = h - data.shape[0]
        pad_w = w - data.shape[1]
        return int(pad_h), int(pad_w)

    def view_patch(self, idx, data):
        return data[idx[0]:idx[1], idx[2]:idx[3]]

    def add_to(self, idx, window, out):
        out[idx[0]:idx[1], idx[2]:idx[3]] += window

    def __call__(self, data, window_func=None):
        visit_count = torch.zeros_like(data)
        rows, cols = data.shape[0], data.shape[1]
        if cols % (self.patch_w + self.stride_w):
            raise ValueError("Invalid stride", cols % (self.patch_w + self.stride_w))
        if rows % (self.patch_h + self.stride_h):
            raise ValueError("Invalid stride", rows % (self.patch_h + self.stride_h))
        indices = []
        for row in range(rows):
            for col in range(cols):
                col_idx_start = col * self.stride_w
                col_idx_end = col_idx_start + self.patch_w
                row_idx_start = row * self.stride_h
                row_idx_end = row_idx_start + self.patch_h

                abort = False
                if row_idx_end > rows or col_idx_end > cols:
                    if self.adaptive_stride:
                        # row_idx_end = min(row_idx_end, rows)
                        # col_idx_end = min(col_idx_end, cols)
                        abort = True
                    else:
                        break

                idx = (row_idx_start, row_idx_end, col_idx_start, col_idx_end)
                indices.append(idx)
                visit_count[idx[0]:idx[1], idx[2]:idx[3]] += 1

                if window_func:
                    window = window_func(idx, self.view_patch(idx, data))

                    patch = self.view_patch(idx, visit_count)
                    patch[:, :] = 10
                    # ax.cla()
                    # ax.imshow(data, cmap="gray")
                    # ax.imshow(visit_count, alpha=0.2)
                    # plt.pause(0.1)
                else:
                    # yield idx, window
                    pass

                if abort:
                    break

        # plt.imshow(visit_count)
        # plt.show()
        return indices, visit_count

    def rebuild(self, windows, out_shape):
        print(windows.shape)

        rows, cols = windows.shape[0], windows.shape[1]
        result = np.zeros(out_shape)
        for col in range(cols):
            for row in range(rows):
                print("row", row, "col", col)
                data = windows[row, col]
                print(data)
                row_h, col_w = data.shape
                assert row_h == self.patch_h
                col_idx_start = col * self.stride
                col_idx_end = col_idx_start + col_w
                row_idx_start = row * self.stride
                row_idx_end = row_idx_start + row_h

                print(col_idx_end)
                print(data)

                result[row_idx_start:row_idx_end, col_idx_start:col_idx_end] = data

        return result


class PatchAnalyzer(nn.Module):
    def __init__(self, batch_size, patch_h=196, patch_w=256, stride_h=98, stride_w=128):
        super().__init__()
        self.bsize = batch_size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.extractor = PatchExtractor(patch_h, patch_w, stride_h, stride_w, adaptive_stride=False)

    def forward(self, input: np.ndarray, net: nn.Module, **kwargs):
        if len(input.shape) == 2:
            h, w = input.shape
            b, c = 1, 1
        elif len(input.shape) == 3:
            c, h, w = input.shape
            b = 1
        else:
            b, c, h, w = input.shape

        assert b == 1
        input = input.squeeze()

        p_h, p_w = self.extractor.pad_delta(input.squeeze())
        print("pad", p_h, p_w)

        input = torch.nn.ReflectionPad2d((0, p_w, 0, p_h))(input[None, None, :, :]).squeeze()

        indices, visit_count = self.extractor(input.squeeze())

        assert visit_count.shape == input.squeeze().shape

        batches = [indices[i:i + self.bsize] for i in range(0, len(indices), self.bsize)]
        out = torch.zeros(input.shape)
        windows = []
        for batch in batches:
            for idx in batch:
                window = self.extractor.view_patch(idx, input)
                windows.append(window)
            batch_tensor = torch.from_numpy(np.stack(windows))
            if len(batch_tensor.shape) == 3:
                batch_tensor = batch_tensor.unsqueeze(1)

            result = net(batch_tensor).squeeze()
            for i in range(len(batch)):
                self.extractor.add_to(batch[i], result[i], out)
            windows.clear()

        out /= visit_count
        return out[:h, :w]


def __test():
    path = '/mnt/unix_data/datastorage/labelbox annotation/hoechst/val/date=190305_exp=set1_part=P00001_plate=A2_time=T00000_type=DIC_well=W00002_z=20x_zlevel=Z00000.png'

    patch_h = 196
    patch_w = 256
    stride_h = int(patch_h // 2)
    stride_w = int(patch_w // 2)

    import torch

    data = load_image(path, True, True)
    data = torch.from_numpy(data)[None, None, :, :].float()

    def net_fn(inp):
        return inp

    pa = PatchAnalyzer(10, patch_h, patch_w, stride_h, stride_w)

    result = pa(data, net=net_fn)

    f, ax = plt.subplots(1, 2, sharey=True, sharex=True)

    ax[0].imshow(data.numpy().squeeze(), cmap="gray")
    ax[1].imshow(result.numpy().squeeze(), cmap="gray")

    plt.show()

    np.testing.assert_array_equal(data.numpy().squeeze(), result)


if __name__ == '__main__':
    __test()
