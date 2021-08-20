# from helper_functions.model_persistence import CheckpointFolder
# from models.filesystem_utils import Folder
# from skimage.measure import compare_ssim
# from torch.nn import CrossEntropyLoss, Softmax2d, Softmax, LogSoftmax, Sigmoid
#
# from trainer import StatsModule
# from unet_debug_utils import _load_debug_info
# import matplotlib.pyplot as plt
# from objective_collection import *
#
#
# def accuracy_pixels(out, label):
#     if isinstance(out, torch.FloatTensor):
#         out = (out.cpu() * 255).round().long()
#
#     if isinstance(label, torch.FloatTensor):
#         label = (label.cpu() * 255).round().long()
#
#     correct_map = out == label
#     acc = (out == label).sum().item() / label.nelement()
#
#     std_error = (out.float() - label.float()).std()
#     mean_error = abs((out.float() - label.float()).mean())
#     return correct_map, acc, (mean_error, std_error)
#
#
# def show_debug_info(inputs, outputs, labels, objective, max_batch_view=3, display="all"):
#     _parse_and_show_debug_info(inputs, outputs, labels, np.ones(len(labels)),
#                                objective if isinstance(objective, str) else objective.__class__.__name__,
#                                max_batch_view=max_batch_view, display=display)
#
#
# def _parse_and_show_debug_info(inputs, outputs, labels, valid_classes,
#                                objective_name, max_batch_view=3, display="all"):
#     input_data = inputs
#
#     criterion_map = objective_for_name(objective_name, reduction="none")
#
#     loss_map = criterion_map.get_debug_loss(outputs, (labels, valid_classes), TrainingConfig(), eval_cpu=True)
#
#     if len(outputs.shape) == 3:
#         outputs = outputs[None, :, :, :]
#     batch_num = min(outputs.shape[0], max_batch_view)
#
#     plot_count = 5
#     if display == "output_target":
#         plot_count = 3
#     elif display == "output_target_loss":
#         plot_count = 4
#     plot_count += 1
#
#     fig, ax = plt.subplots(batch_num, plot_count, figsize=(12, 6), sharex=True, sharey=True)
#
#     for batch_i in range(batch_num):
#         input_img = process_input_img(input_data[batch_i])
#         out = process_output(outputs[batch_i], criterion_map)
#         target = process_label(labels[batch_i])
#         #b_loss = process_loss_map(loss_map[batch_i].squeeze())
#         b_loss = input_img
#
#         plot_ax = ax[batch_i] if batch_num > 1 else ax
#         if batch_i == 0:
#             if batch_num > 1:
#                 set_titles(plot_ax, display=display)
#             else:
#                 set_titles(plot_ax, display=display)
#
#         show_output(plot_ax, input_img, out, target, b_loss, display)
#     plt.tight_layout()
#     plt.show()
#
#
# def set_titles(axis, display):
#     axis[0].set_title("input")
#     axis[1].set_title("output")
#     axis[2].set_title("target")
#     if display == "all" or display == "output_target_loss":
#         axis[4].set_title("loss")
#     if display == "all":
#         axis[5].set_title("y-p")
#
#
# def show_output(axis, input_img, b_out, b_label, b_loss, display, convert=False):
#     axis[0].imshow(input_img, cmap="gray")
#
#     if isinstance(b_out, torch.FloatTensor) and convert:
#         b_out = (b_out.cpu() * 255).round().long()
#
#     axis[1].imshow(b_out, cmap="gray")
#     b_label = b_label.float()
#     b_label = (b_label-b_label.min())/(b_label.max()-b_label.min())
#
#     axis[2].imshow(b_label, cmap="gray")
#     #acc_map, acc, mean_std = accuracy_pixels(b_out, b_label)
#     ssim_out = b_out.cpu().numpy().squeeze()#.astype(np.uint8)
#     ssim_target = b_label.cpu().numpy().squeeze()#.astype(np.uint8)
#
#     ssim = compare_ssim(ssim_out, ssim_target)
#     print(ssim)
#     axis[3].imshow(ssim_out-ssim_target, cmap="jet")
#     #plt.title(
#     #    "Err map: acc {0:0.2f} |\n mean {1:0.2f} std {2:0.2f} |\nssim: {3:0.3f}".format(acc, mean_std[0], mean_std[1],
#     #                                                                                 ssim))
#
#     if display == "all" or display == "output_target_loss":
#         axis[4].set_ylabel("l: {0:.3f}".format(b_loss.mean().item()))
#         axis[4].imshow(b_loss, cmap="jet")
#     if display == "all":
#         axis[5].imshow(b_loss, cmap="jet")
#
#
# def process_input_img(b_input_img):
#     return b_input_img.squeeze().numpy()
#
#
# def process_output(batch_i_output, objective):
#     if isinstance(objective, L1Reconstruction):
#         batch_i_output = torch.nn.Sigmoid()(batch_i_output[None, :, :, :])
#         batch_i_output = normalize_tensor(batch_i_output)
#         if batch_i_output.shape[1] == 2:
#             batch_i_output = batch_i_output[:, 0, :, :]
#
#         pass
#     if isinstance(objective, BinaryCrossEntropySingleOutputLayer) or isinstance(objective, FocalLossSingleOutputLayer):
#         batch_i_output = torch.nn.Sigmoid()(batch_i_output[None, :, :, :])
#     elif isinstance(objective, CrossEntropy256Out):
#         sm_bi = torch.nn.Softmax2d()(batch_i_output[None, :, :, :])
#         batch_i_output = sm_bi.argmax(dim=1)
#         # batch_i_output = LogSoftmax(2)(batch_i_output).argmax(dim=0)
#
#     batch_i_output = normalize_tensor(batch_i_output,batch_normalize=False)
#     return batch_i_output.squeeze()
#
#
# def process_label(batch_i_label):
#     return batch_i_label.squeeze()
#
#
# def process_loss_map(batch_i_loss_map):
#     return batch_i_loss_map
#
#
# def objective_for_name(obj_name, reduction):
#     constructor = globals()[obj_name]
#     criterion = constructor("adabound", reduction=reduction)
#     train_cfg = TrainingConfig()
#     train_cfg.TRAIN_DEVICE = train_cfg.WAIT_DEVICE
#     criterion.criterion = criterion._build_criterion(train_cfg)
#     # criterion.criterion = criterion._build_criterion(train_cfg)
#     criterion.stats = StatsModule(train_cfg)
#     return criterion
#
#
# def inspect_debug_folder(config, model_id, last_only=False):
#     model_folder = Folder(config.DEBUG_PATH).make_sub_folder(model_id)
#     debug_points = model_folder.get_folders(abs_path=False)
#     debug_points = [(int(fn.split("_")[1]), fn) for fn in debug_points]
#
#     debug_points = sorted(debug_points, key=lambda x: x[0], reverse=True)
#
#     for idx, folder in debug_points:
#         folder = model_folder.make_sub_folder(folder)
#         print(folder)
#         inputs, outputs, labels, reconstruction_target, valid_classes, info_json = _load_debug_info(folder.path_abs)
#         objective = info_json['loss_type']
#         show_debug_info(inputs, outputs, labels, objective, max_batch_view=3,
#                         display="output_target_loss")
#         if last_only:
#             break
#
#
# if __name__ == '__main__':
#     #model_id = "thesis_exp_unet_l1_ssim_comb_20x_ztz_txred"
#
#
#     from training.patches.train_patches_l1 import train_cfg, model_id
#     #cfg = TrainingConfig()
#     #cfg.DEBUG_PATH = "/home/mrd/Desktop/unix_data/datastorage/networks/unet-pytorch/training/patches/debug_info"
#
#     #cfg.CP_FOLDER = CheckpointFolder("/mnt/unix_data/datastorage/thesis_experiment_results/fl_reconstruction")
#     inspect_debug_folder(train_cfg, model_id, last_only=False)
