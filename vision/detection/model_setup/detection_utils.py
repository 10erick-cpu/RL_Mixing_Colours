import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.helper_functions.img_utils import gray2rgb


def remove_detection(target_image, mask, fill="mean"):
    if fill == "mean":
        target_image[:, mask.squeeze() > 0] = target_image.mean()
    else:
        target_image[:, mask.squeeze() > 0] = fill
    return target_image


def draw_detection(draw_image, mask, box, color):
    mask = mask.cpu().numpy().squeeze()

    draw_image[mask > 0] = color
    draw_bbox(draw_image, box, color=color.tolist())

    return draw_image


def threshold_mask(mask, th_min):
    mask[mask < th_min] = 0
    mask[mask > 0] = 1
    return mask


def do_multi_detection(input_image, detector, iterations=5,
                       confidence_th=0.6, mask_th=0.5,
                       colors=[32, 64, 128, 184, 255],
                       show=False):
    display_image = input_image.cpu().numpy()
    display_image = (display_image * 255).astype(np.uint8).squeeze()

    color_overlay = gray2rgb(np.zeros_like(display_image[0], dtype=np.uint8))

    for i in range(iterations):

        scores, boxes, masks = detector.detect(input_image)
        for idx, mask in enumerate(masks):
            if scores[idx] <= confidence_th:
                continue

            mask = threshold_mask(mask, mask_th)
            draw_detection(color_overlay, mask, boxes[idx], np.random.choice(colors, size=3))
            input_image = remove_detection(input_image, mask, fill=0)
        if show:
            plt.imshow(display_image, cmap="gray")
            plt.imshow(color_overlay, alpha=0.3)
            plt.title("#1")
            plt.show()

    return color_overlay


def do_single_detection(input_tensor, detector, confidence_th=0.6, mask_th=0.5, colors=[32, 64, 128, 184, 255]):
    color_overlay = gray2rgb(np.zeros_like(input_tensor[0], dtype=np.uint8))

    scores, boxes, masks = detector.detect(input_tensor)

    print("num detections", len(boxes))
    for idx, mask in enumerate(masks):
        if scores[idx] <= confidence_th:
            continue

        mask = threshold_mask(mask, mask_th)
        draw_detection(color_overlay, mask, boxes[idx], color=np.random.choice(colors, size=3, replace=False))

    return color_overlay


def draw_bboxes(image, boxes):
    for coords in boxes:
        draw_bbox(image, coords)
    return image


def draw_bbox(image, box, color=(255, 0, 0)):
    coords = box.round().int().cpu().numpy().squeeze().tolist()
    x1, y1, x2, y2 = coords

    cv2.rectangle(image, (x1, y1), (x2, y2), color=tuple(color), thickness=2)
    return image


def detections_to_multimask(detections):
    multi_mask = None
    for mask in detections:
        if multi_mask is None:
            multi_mask = mask.cpu().numpy()
        else:
            # mask = torch.nn.functional.sigmoid(mask)
            mask = mask.cpu().numpy()
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            # mask = (mask - mask.min()) / (mask.max() - mask.min())
            # mask[mask >= 0.5] = 1

            # plt.imshow(mask.squeeze())
            # plt.show()
            multi_mask[mask == 1] = 1
    return multi_mask
