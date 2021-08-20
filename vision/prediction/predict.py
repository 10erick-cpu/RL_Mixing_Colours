import torch
from torch.autograd.grad_mode import no_grad

from datasets.clean_ds_v1.hoechst_ds import HoechstCropsVal
from datasets.imagesets.clean_ds_patches_20x import CalceinDenseCropsTrain, CalceinDenseCropsVal
from helper_functions.img_utils import load_image
from helper_functions.tile_manager import TileManager
from helper_functions.torch_utils import training_device
from models.filesystem_utils import Folder
from new_training.patches.calc_hoechst.train_default_configs import configs
from new_training.training_objectives import MinMaxTargetPreprocessor
from training_utils import get_data_loader

params = configs[-1]
# params.model.params.bilinear_upsampling=False
params.training.batch_size=12
model = params.get_model()
model.load(params)
model.to(training_device())
model.eval()


@no_grad()
def predict_ds(ds):
    for img, target, _ in get_data_loader(ds, batch_size=1, shuffle=False, num_workers=0):
        out = model(img.to(training_device()))

        img = img.cpu().numpy().squeeze()
        out = out.cpu().numpy().squeeze()

        import matplotlib.pyplot as plt

        f, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 8))

        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(out, cmap="gray")
        ax[2].imshow(img, cmap="gray")
        ax[2].imshow(out, cmap="jet", alpha=0.5)
        plt.show()
        plt.close(f)


@no_grad()
def predict_fp(fp, tile=False):
    import matplotlib.pyplot as plt
    for img_path in fp:
        print(img_path)
        src_img = load_image(img_path, force_8bit=True, force_grayscale=True)

        if tile:
            imgs = TileManager.to_tiles(src_img, 16)
            for img in imgs:
                img = torch.from_numpy(img[None, None, :, :]).float()
                # img = (img - img.min()) / (img.max() - img.min())
                # img = (img - img.mean()) / (img.std())
                print("img", img.shape)
                out = model(img.to(training_device())).squeeze()  # .argmax(dim=0)
                print("out", out.shape)
                img = img.cpu().numpy().squeeze()
                out = out.cpu().numpy().squeeze()

                f, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 8))

                ax[0].imshow(img, cmap="gray")
                ax[1].imshow(out, cmap="gray")
                ax[2].imshow(img, cmap="gray")
                ax[2].imshow(out, cmap="jet", alpha=0.5)
                plt.show()
        else:
            f, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 8))

            img = torch.from_numpy(src_img[None, None, :, :]).float()
            out = model(img.to(training_device())).squeeze()

            img = img.cpu().numpy().squeeze()
            out = out.cpu().numpy().squeeze()
            # out[out<0]=0

            ax[0].imshow(img, cmap="gray")
            ax[1].imshow(out, cmap="gray")
            ax[2].imshow(img, cmap="gray")
            ax[2].imshow(out, cmap="jet", alpha=0.5)
            plt.show()


training_set = CalceinDenseCropsTrain()

# training_set.init()

val_set = HoechstCropsVal()

val_set.init()

predict_ds(val_set)

predict_fp(Folder(
    "/mnt/unix_data/datastorage/ztz_datasets/01_clean_ds_v2/190305_IBIDI_JIMT1_CT_HOE/train_test_split/val/fullres").make_file_provider(
    contains="DIC", include_subdirs=True))
