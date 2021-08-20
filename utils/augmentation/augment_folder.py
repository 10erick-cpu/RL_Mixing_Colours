import copy

import Augmentor
import matplotlib.pyplot as plt
from utils.file_management.naming_strategies import CsvNamingStrategy
from utils.models.folder import Folder


def build_pipeline(images, labels, crop_w, crop_h):
    p = Augmentor.DataPipeline(images=images, labels=labels)
    p.crop_by_size(1, crop_w, crop_h, centre=False)
    p.flip_left_right(0.25)
    p.flip_top_bottom(0.25)
    p.random_brightness(0.5, 0.7, 1.5)
    p.random_color(0.5, 0.7, 1.0)
    p.random_contrast(0.5, 0.7, 1.5)
    p.random_distortion(0.33, 12, 12, 3)
    p.rotate(0.33, 10, 10)
    p.shear(0.25, 3, 3)
    p.skew_left_right(0.15, 0.25)
    p.skew_top_bottom(0.15, 0.25)
    return p


def check_image_valid(image, target):
    print(image.mean(), image.std(), image.max(), target.mean(), target.std(), target.max())
    if image.std() <= 6 or image.max() <= 20 or target.std() <= 3 or target.max() <= 20:
        print("invalid img detected", "max", target.max())

        return False
    return True


def augment(input_dir: Folder, output_dir: Folder, target_channel, samples_per_image=100, max_pauses=-1,
            ims_filter=None):
    ims = CsvNamingStrategy().find_image_sets(input_dir)

    for idx, im in enumerate(ims):
        if ims_filter is not None and not ims_filter(im):
            continue

        im_subdir = output_dir.make_sub_folder(CsvNamingStrategy().get_file_names(im)['dic'].split(".")[0])
        new_im = copy.deepcopy(im)

        dic, label = im.dic, im.image(target_channel)
        # dic = Image.fromarray(dic)
        # label = Image.fromarray(label)

        p = build_pipeline([[dic, label]], [0, 1], crop_w=256, crop_h=195)

        for i in range(samples_per_image):
            print(f"\r img {idx + 1}/{len(ims)} sample {i + 1}/{samples_per_image}", end="")
            batch, labels = p.sample(1)
            img, fl = batch[0]

            num_pauses = 0
            while not check_image_valid(img, fl):

                if num_pauses <= max_pauses:
                    f, ax = plt.subplots(1, 4, sharex=True, sharey=True)

                    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255, interpolation=None)
                    ax[1].imshow(fl, cmap="gray", vmin=0, vmax=255, interpolation=None)

                    test = (fl - fl.min()) / (fl.max() - fl.min())
                    test *= 255
                    ax[2].imshow(test, cmap="gray", vmin=0, vmax=255,
                                 interpolation=None)
                    ax[3].imshow(label, cmap="gray", vmin=0, vmax=255, interpolation=None)
                    # plt.pause(1)
                    plt.show()
                    plt.close()
                batch, labels = p.sample(1)
                img, fl = batch[0]
                num_pauses += 1

            new_im.images['dic'] = img
            new_im.images[target_channel] = fl
            new_im.channels['dic']['cropid'] = str(i)
            new_im.channels[target_channel]['cropid'] = str(i)

            new_im.persist(CsvNamingStrategy(), im_subdir)
        im.empty_cache()


base_dir = "/media/mrd/thesis_data/cleaned/data_manual_process/train_test_split"
base_dir = Folder(base_dir)

folders = ['train', 'val', 'test']

for f_name in folders:
    src_folder = base_dir.make_sub_folder(f_name, create=False)
    target_dir = src_folder.make_sub_folder("crops")
    augment(src_folder, target_dir, target_channel='calcein', ims_filter=None, samples_per_image=500)


def filter_fn(im):
    q = im.quality()
    return q is None or q <= 3
