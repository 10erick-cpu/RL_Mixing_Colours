import torch

from utils.helper_functions.model_persistence import CheckpointFolder

TRAINING_DEBUG = False

MODEL_ID_1CHAN = "unet_model_ccells_multiclass_finetune"

MODEL_ID_3CHAN = "unet_model_3chan_in_test"

MODEL_ID_IN_SILICO_LABELING="unet_model_in_silico_labeling"
MODEL_ID_IN_SILICO_LABELING_1CHAN="insilico_single_layer"

UNET_CP_FOLDER_DEFAULT = CheckpointFolder("/home/mrd/Desktop/unix_data/datastorage/networks/unet-pytorch/model_cp", False)

UNET_DETECTION_CONFIDENCE = 0.85
# TRAINING
UNET_LEARNING_RATE = 0.0005
UNET_LEARNING_MOMENTUM = 0.99
UNET_TRAIN_BATCH_SIZE = 11
UNET_WEIGHT_DECAY = 0.01

#UNET_DATASET_MEAN = [6606.56022606]
#UNET_DATASET_STD = [1236.53873848]
UNET_DATASET_MEAN_RGB = [139.15208455]
UNET_DATASET_STD = [23.10146104]
#UNET_DATASET_STD = None

# UNET_DATASET_MEAN_3CHAN = [0.54772437, 0.08148979, 0.11874993]
# UNET_DATASET_STD_3CHAN = [7.2866397, 3.257237,  2.5051954]

UNET_DATASET_MEAN_3CHAN = [6606.56022606,  744.51323982, 1355.09701226]
UNET_DATASET_STD_3CHAN = [1236.53873848,  395.9550118,  450.02286211]

# INPUT_SCALING
UNET_INPUT_RESCALE_FACTOR = 0.5

UNET_RESIZE_TARGET_HEIGHT = int(1024 * UNET_INPUT_RESCALE_FACTOR)
UNET_RESIZE_TARGET_WIDTH = int(1344 * UNET_INPUT_RESCALE_FACTOR)

UNET_INPUT_RESCALE_FACTOR_NO_BATCH = 0.85
UNET_RESIZE_TARGET_HEIGHT_NO_BATCH = int(1024 * UNET_INPUT_RESCALE_FACTOR_NO_BATCH)
UNET_RESIZE_TARGET_WIDTH_NO_BATCH = int(1344 * UNET_INPUT_RESCALE_FACTOR_NO_BATCH)

UNET_TRAIN_BCE_MODE = False
UNET_TRAIN_BATCH_NORM_MOMENTUM=0.1
UNET_TRAIN_BATCH_NORM_TRACK_RUNNING_MEAN=True

UNET_TRAIN_IGNORE_TARGET_INDEX=-100

CUDA_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
