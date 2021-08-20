# from collections import deque
#
# import torch
# from torch import nn
# from unet_utils.convert_unet import get_architecture
#
# from unet_model import Unet, InSilicoLabelingUnet
# from unet_settings_global import MODEL_ID_1CHAN, UNET_CP_FOLDER_DEFAULT, MODEL_ID_3CHAN, MODEL_ID_IN_SILICO_LABELING
# import numpy as np
#
#
# def update_children(children, layer_queue, include_transpose=False, expand_channels=False):
#     import_idx = -1
#     for child in children:
#         if isinstance(child, nn.Conv2d) or (include_transpose and isinstance(child, nn.ConvTranspose2d)):
#             import_idx += 1
#             import_layer = layer_queue.pop()
#             print("update child {}\n  with layer {} - {}".format(child.weight.data.shape,
#                                                                  torch.from_numpy(import_layer.data.weights).shape,
#                                                                  import_layer))
#             print()
#             if child.weight.data.shape[1] != import_layer.data.weights.shape[1] and expand_channels:
#                 print("Applying channel expansion")
#                 src_w = import_layer.data.weights[0][0]
#                 new_w = [src_w for idx in range(child.weight.data.shape[1])]
#                 new_layer = np.stack(new_w, axis=0)
#                 new_tensor = np.stack([new_layer for i in range(child.weight.data.shape[0])], axis=0)
#                 import_layer.data.weights = new_tensor
#
#             child.weight.data = torch.from_numpy(import_layer.data.weights)
#             child.bias.data = torch.from_numpy(import_layer.data.bias)
#
#
# def set_encoder(encoder, layer_queue, expand_channels=False):
#     update_children(encoder.encode.children(), layer_queue, include_transpose=False, expand_channels=expand_channels)
#
#
# def set_decoder(decoder, layer_queue):
#     update_children(decoder.decode.children(), layer_queue, include_transpose=True)
#
#
# def import_weights():
#     arch = get_architecture()
#     arch = deque(reversed(arch))
#
#     model = Unet(num_input_channels=3, model_id=MODEL_ID_3CHAN)
#
#     # checkpoint = model.load_latest_checkpoint(UNET_CP_FOLDER_DEFAULT, model.model_id)
#
#     set_encoder(model.enc1, layer_queue=arch, expand_channels=True)
#     set_encoder(model.enc2, layer_queue=arch)
#     set_encoder(model.enc3, layer_queue=arch)
#     set_encoder(model.enc4, layer_queue=arch)
#
#     set_decoder(model.center, layer_queue=arch)
#     set_decoder(model.dec4, layer_queue=arch)
#     set_decoder(model.dec3, layer_queue=arch)
#     set_decoder(model.dec2, layer_queue=arch)
#     update_children(model.dec1.children(), layer_queue=arch, include_transpose=True)
#     update_children([model.final], layer_queue=arch)
#
#     UNET_CP_FOLDER_DEFAULT.save_model(model.model_id + "_updated_weights", 0, model)
#     assert len(arch) == 0
#
#
# # import_weights()
#
#
# def single_output_unet_to_3_chan_output():
#     src_model_id = MODEL_ID_1CHAN
#
#     src_model = Unet(src_model_id)
#     out_model = InSilicoLabelingUnet()
#
#     checkpoint = src_model.load_latest_checkpoint(UNET_CP_FOLDER_DEFAULT, src_model.model_id)
#     model_state_dict = checkpoint['model_state_dict']
#     print(model_state_dict.keys())
#     final = model_state_dict['final.weight']
#     # torch.Size([2, 128, 1, 1])
#     # torch.Size([4, 128, 1, 1])
#     print("src", final.shape)
#     #bg = final.clone()[None, 0, :, :]
#     classes = final[1].clone().repeat(out_model.num_classes, 1, 1, 1)
#     #print("bg", bg.shape)
#     print("classes", classes.shape)
#     #final = torch.cat((bg, classes))
#     print("result", final.shape)
#     model_state_dict['final.weight'] = classes
#
#     final_bias = model_state_dict['final.bias']
#
#     bg = final_bias[0].view(1)
#     classes = final_bias[1].repeat(out_model.num_classes)
#     #result_bias = torch.cat((bg, classes))
#     #print(result_bias)
#     model_state_dict['final.bias'] = classes
#     out_model.load_state_dict(model_state_dict, strict=False)
#     UNET_CP_FOLDER_DEFAULT.save_model(out_model.model_id, 0, out_model)
#
#
# if __name__ == '__main__':
#     single_output_unet_to_3_chan_output()
