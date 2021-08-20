import cv2
import numpy as np


def extract_rect(mat, rect_tpl):
    # ---> x
    # |
    # v
    # y
    # rect: (x, y,  w, h)
    # mat: (row, col)
    return mat[rect_tpl[0]: rect_tpl[2], rect_tpl[1]:rect_tpl[3]]


def rect(mat, min_id, max_id):
    return mat[min_id[0]:max_id[0], min_id[1]:max_id[1]]


def rect_for_center(mat, center, rect_size):
    lo = (center[0] - rect_size // 2, center[1] - rect_size // 2)
    hi = (center[0] + rect_size // 2, center[1] + rect_size // 2)

    lo = clamp_index(mat.shape, lo)
    hi = clamp_index(mat.shape, hi)

    return mat[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1]


def rect_assign(mat, center, rect_size, val):
    lo = (center[0] - rect_size // 2, center[1] - rect_size // 2)
    hi = (center[0] + rect_size // 2, center[1] + rect_size // 2)

    lo = clamp_index(mat.shape, lo)
    hi = clamp_index(mat.shape, hi)

    mat[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1] = val


def rect_add(mat, center, rect_size, val):
    lo = (center[0] - rect_size // 2, center[1] - rect_size // 2)
    hi = (center[0] + rect_size // 2, center[1] + rect_size // 2)

    lo = clamp_index(mat.shape, lo)
    hi = clamp_index(mat.shape, hi)

    mat[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1] += val


def clamp_index(max_val, target, min_val=(0, 0)):
    target = np.asarray(target)
    for i in range(len(target)):
        if not min_val[i] < target[i] < max_val[i]:
            target[i] = max(min_val[i], min(target[i], max_val[i]))

    return target


def check_index(min_val, max_val, index):
    for i in range(len(index)):
        if not min_val[i] <= index[i] < max_val[i]:
            return False
    return True


def safe_val_read(index_tuple, mat):
    if not check_index((0, 0), mat.shape, index_tuple):
        return None
    return mat[index_tuple]


def safe_add(val):
    return val if val is not None else 0


def distance_to_index(mat, target_idx):
    mask = np.ones_like(mat, dtype=np.uint8)
    mask[target_idx] = 0
    distance = np.zeros_like(mask, dtype=np.float32)
    cv2.distanceTransform(mask, cv2.DIST_L2, 3, dst=distance)
    return distance



def normalize_2d(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


def distance_normed_to_index(mat, index):
    return normalize_2d(distance_to_index(mat, index))


def calc_inverse_normed_distance_map(height, width, index=None):
    d = np.zeros((height, width))
    if index is None:
        return 1 - distance_normed_to_index(d, center_index(d))
    return 1 - distance_normed_to_index(d, index)


def center_index(mat):
    h, w = mat.shape
    return h // 2, w // 2
