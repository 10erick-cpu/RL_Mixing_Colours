import numpy as np
import torch
from torchvision.utils import make_grid


def input_output_target_to_grid(input, output, target, nrows=3):
    assert len(input.shape) == 4, "only implemented for batch tensors"
    if len(output.shape) == 3:
        output = output.unsqueeze(1)

    if len(target.shape) == 3:
        target = target.unsqueeze(1)
    # import matplotlib.pyplot as plt
    # plt.imshow(input[0].cpu().numpy().squeeze())
    # plt.show()
    # plt.imshow(output[0].cpu().numpy().squeeze())
    # plt.show()

    r = None
    for zero, one, three in zip(input.float(), output.float(), target.float()):
        if r is None:
            r = torch.stack((zero, one, three), dim=-1)
        else:
            cat = torch.stack((zero, one, three), dim=-1)
            r = torch.cat((r, cat), dim=3)

    r = r.permute(3, 0, 1, 2)
    grid = make_grid(r, nrow=nrows)
    return grid.cpu().numpy().transpose((1, 2, 0))


# def jaccard_score(y_pred, y_label):
#     return jaccard_similarity_score(y_label, y_pred)


def accuracy(pred, gt):
    pred = pred.squeeze()
    gt = gt.squeeze()
    if pred.dtype != torch.int64 or gt.dtype != torch.int64:
        raise ValueError("Accuracy requires integer values", pred.dtype, gt.dtype)

    def accuracy(out, label):
        hit = (out.cpu() == label).float()
        sum_dims = tuple(np.arange(len(hit.shape)))[1:]
        return (hit.sum(dim=sum_dims) / label[0].nelement())

    return accuracy(pred, gt)
    correct_arr = (pred == gt)
    wrong_arr = (pred != gt)

    correct = correct_arr.sum()
    wrong = wrong_arr.sum()
    assert len(gt.shape) == 2
    num_elems = gt.shape[0] * gt.shape[1]

    assert correct + wrong == num_elems, (num_elems, correct + wrong)

    return correct / num_elems


if __name__ == '__main__':
    test = np.zeros((100, 100))

    gt = test.copy()
    test[0, 0] = 1

    acc = accuracy(test, gt)
    assert acc == 0.9999, acc

    test[0, 0] = 0
    acc = accuracy(test, gt)
    assert acc == 1.0, acc

    test = np.zeros((100, 100))
    test[:50, :] = 1

    acc = accuracy(test, gt)
    assert acc == 0.5, acc
