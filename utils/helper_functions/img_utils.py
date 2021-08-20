import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

CH_RED = 0
CH_GREEN = 1
CH_BLUE = 2


def show_subplot(data, cmaps=None, share=True):
    num_plots = len(data)
    f, ax = plt.subplots(1, num_plots, sharex=share, sharey=share)
    for i in range(num_plots):
        elem = data[i]
        if torch.is_tensor(elem):
            elem = elem.cpu().numpy()

        ax[i].imshow(elem.squeeze(), cmap="gray" if cmaps is None else cmaps[i])

    plt.show()
    plt.close(f)


def load_image(image_path, force_grayscale=False, force_8bit=False, shape=None):
    img = np.asarray(cv2.imread(image_path, -cv2.IMREAD_ANYDEPTH))
    if force_grayscale and not is_grayscale(img):
        img = rgb2gray(img)

    if len(img.shape) == 3 and img.shape[2] > 1:
        img = bgr2rgb(img)

    if img.dtype != np.uint8 and force_8bit:
        img = image_to_8bit(img)

    if shape is not None:
        img = cv2.resize(img, shape)

    return img


def save_image(file, image, force_grayscale=False):
    from skimage import io

    if not is_grayscale(image):
        if not force_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = rgb2gray(image)
    if force_grayscale:
        Image.fromarray(image).save(file)
        #io.imsave(file, image, check_contrast=False)
    else:
        io.imsave(file, image)


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def remove_blobs(img, size_th):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = size_th

    # your answer image
    img2 = np.zeros(output.shape)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2


def image_to_8bit(img):
    if img.dtype == np.uint8:
        return img
    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def histogram_equalization(img):
    if not is_grayscale(img):
        print("HistEq requires grayscale image, converting rgb input")
        return cv2.equalizeHist(rgb2gray(img))
    return cv2.equalizeHist(img)


def image_to_float(img):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def load_and_parse_grayscale_8bit_img(file, equalize=False):
    img = cv2.imread(file, -1)
    img = img / np.max(img)
    img *= 255
    img = img.astype(np.uint8)
    if equalize:
        img = cv2.equalizeHist(img)
    return img

    # img = img / np.iinfo(np.int16).max

    # res = np.hstack((img, equ))  # stacking images side-by-side
    # cv2.imwrite('res.png', res)

    # img = img / np.iinfo(np.int16).max

    print(np.iinfo(np.int8).max)
    # img *=np.iinfo(np.int8).max
    img = img.astype(np.uint8)
    print(img[:5, :5])

    return img


def is_grayscale(img):
    return len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)


def gray2rgb(img):
    if is_grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        print("gray2rgb warning: image already rgb, no conversion performed")
        return img


def draw_fluorescence_overlays(dic, red=None, green=None, blue=None, overlay_weight=1):
    result = dic
    if is_grayscale(result):
        result = gray2rgb(result)
    if red is not None:
        result = mix_grayscale_to_rgb(red, 'r', result, gray_weight=overlay_weight)
    if green is not None:
        result = mix_grayscale_to_rgb(green, 'g', result, gray_weight=overlay_weight)
    if blue is not None:
        result = mix_grayscale_to_rgb(blue, 'b', result, gray_weight=overlay_weight)
    return result


def mix_grayscale_to_rgb(grayscale, channel, rgb, gray_weight=0.5, gamma=0):
    assert is_grayscale(grayscale)
    assert len(rgb.shape) == 3

    gray_rgb = np.zeros_like(rgb)
    if channel == 'r':
        channel = 2
    elif channel == 'g':
        channel = 1
    elif channel == 'b':
        channel = 0
    gray_rgb[:, :, channel] = grayscale
    return cv2.addWeighted(gray_rgb, gray_weight, rgb, 1, gamma)


def bgr2rgb(cv_img):
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


def rgb2gray(cv_img):
    if len(cv_img.shape) == 2:
        return cv_img

    return cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)


def fill_polygon(dst_mat, polygon, fill_value=1):
    cv2.fillPoly(dst_mat, pts=[polygon], color=fill_value)


def draw_mask(annotation_object, dst_mat, fill=True, outline=True):
    assert dst_mat.dtype == np.uint8

    polygon = []
    for data in annotation_object.points:
        x = data[0]
        y = data[1]
        polygon.append((x, y))
    if len(polygon) < 2:
        return dst_mat
    img = Image.fromarray(dst_mat)
    ImageDraw.Draw(img).polygon(polygon, outline=outline, fill=fill)
    mask = np.array(img).astype(np.uint8)
    return mask


def img_plot(img_1, img_2=None, share_scale=True, grayscale=False):
    if img_2 is None:
        plt.imshow(img_1) if not grayscale else plt.imshow(img_1, cmap="gray")
        plt.show()
        return
    print("Plotting 2 imgs")
    figsize = (16, 16)
    f, axarr = plt.subplots(1, 2, sharex=share_scale, sharey=share_scale)

    if len(img_1.shape) < 3 or img_1.shape[2] == 1:
        axarr[0].imshow(img_1, cmap="gray")
    else:
        axarr[0].imshow(img_1)

    if len(img_2.shape) < 3 or img_2.shape[2] == 1:
        axarr[1].imshow(img_2, cmap="gray")
    else:
        axarr[1].imshow(img_2)
    plt.show()


def apply_CLAHE_with_grayscale_image(rgb_image, blocksize=127, max_slope=3.0):
    rgb = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(apply_CLAHE_with_rgb_image(rgb, blocksize, max_slope), cv2.COLOR_RGB2GRAY)


def apply_CLAHE_with_rgb_image(rgb_image, blocksize=127, max_slope=3.0):
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=max_slope, tileGridSize=(blocksize, blocksize))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb


def apply_CLAHE(img_path, blocksize=127, bins=256, max_slope=3.0):
    bgr = cv2.imread(img_path)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=max_slope, tileGridSize=(blocksize, blocksize))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb


# def rolling_ball_subtract_bg_with_path(img_path, ball_radius=30, light_bg=True, paraboloid=False, presmooth=True):
#     img = cv2.imread(img_path, 0)
#     return rolling_ball_subtract_bg(img, ball_radius, light_bg, paraboloid, presmooth)


# def rolling_ball_subtract_bg(img, ball_radius=30, light_bg=True, paraboloid=False, presmooth=True):
#     from cv2_rolling_ball import subtract_background_rolling_ball
#     img, background = subtract_background_rolling_ball(img, ball_radius, light_background=light_bg,
#                                                        use_paraboloid=paraboloid, do_presmooth=presmooth)
#     return img, background


def mask_array_to_multi_mask(mask_arr):
    assert len(mask_arr.shape) == 3
    num_masks = mask_arr.shape[2]
    assert num_masks > 1

    result = np.zeros(shape=(mask_arr.shape[0], mask_arr.shape[1]), dtype=np.uint8)

    for mask_id in range(num_masks):
        mask = mask_arr[:, :, mask_id]
        result = np.bitwise_or(result, mask)

    return result


def create_threshold_otsu(img_arr, smooth=True, smooth_window=(5, 5), smooth_sigma=(1, 1)):
    result = []

    for im in img_arr:
        r = im
        if smooth:
            r = cv2.GaussianBlur(im, smooth_window, sigmaX=smooth_sigma[0], sigmaY=smooth_sigma[1])

        return_prim, prim = cv2.threshold(r, 0, np.iinfo(np.uint16).max if im.dtype == np.uint16 else 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result.append(prim)
        print("otsu return vals: prim {0:.3f}".format(return_prim))
    return result


def create_threshold(imgs, blocksize=257, const=1.5, hole_fill=True):
    result = []

    for im in imgs:
        im_th = cv2.adaptiveThreshold(im, np.iinfo(np.uint16).max if im.dtype == np.uint16 else 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, blocksize, const)

        kernel = np.ones((5, 5), np.uint8)
        im_th = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
        im_th = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel)
        if hole_fill:
            im_th = fill_holes(im_th)
        result.append(im_th)

    return result


def fill_holes(input_mask):
    des = input_mask
    _, contour, hier = cv2.findContours(des, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    des = cv2.drawContours(des, contour, -1, 1, cv2.FILLED)
    return cv2.bitwise_or(des, input_mask)


def detection_mask_iou(prediction, label, complete_iou=True):
    raise NotImplementedError()
    assert prediction.shape == label.shape

    label_region = np.where(label == 1)

    if complete_iou:

        score = jaccard_score(label, prediction)
    else:
        score = jaccard_score(label[label_region], prediction[label_region])

    # im_pred = prediction
    # im_pred * 255
    #
    # im_label = label
    # im_label * 255
    #
    # f, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
    # axarr[0].imshow(im_pred)
    # axarr[1].imshow(im_label)
    # plt.show()

    return score


def mask_rcnn_detections_to_multimask(det_result):
    masks = det_result['masks']
    multi_mask = mask_array_to_multi_mask(masks)
    return multi_mask

#
# def find_detection_label(prediction, mask_arr, exclude_idx, abort_threshold=1.0):
#     assert len(mask_arr.shape) == 3
#     num_masks = mask_arr.shape[2]
#     assert num_masks > 1
#
#     top_iou = 0
#     top_mask = None
#     top_mask_idx = None
#     with progressbar.ProgressBar(max_value=num_masks) as bar:
#
#         for mask_id in range(num_masks):
#
#             if mask_id in exclude_idx:
#                 continue
#             label_mask = mask_arr[:, :, mask_id]
#             iou = detection_mask_iou(prediction, label_mask)
#             if iou > top_iou:
#                 top_iou = iou
#                 top_mask = label_mask
#                 top_mask_idx = mask_id
#             bar.update(mask_id + 1)
#
#             if top_iou >= abort_threshold:
#                 print("aborting with mask_id ", mask_id)
#                 return top_mask_idx, top_mask, top_iou
#         if top_mask is None:
#             print("No mask found for prediction")
#
#         return top_mask_idx, top_mask, top_iou

#
# def find_undetected_labels(detections, masks, iou_threshold=0.4):
#     undetected = []
#     found_labels_idx = []
#     for detection_id in range(detections.shape[2]):
#         detection = detections[:, :, detection_id]
#         r_mask_idx, r_mask, r_iou = find_detection_label(detection, masks[0], found_labels_idx,
#                                                          0.8)
#         if r_iou == 0 or r_iou <= iou_threshold:
#             undetected.append(idx)
#         else:
#             found_labels_idx.append(r_mask_idx)
#
#         print_progress(detection_id + 1, detections.shape[2], "find_undetected_labels")
#     return undetected, found_labels_idx
