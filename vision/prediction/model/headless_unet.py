from torch import nn

# from fnet_model import FNet
from vision.prediction.model.fnet_model import FNet
from vision.prediction.training.data_processors import MinMaxTargetPreprocessor


class HeadlessUNet(nn.Module):

    def __init__(self, model_params):
        super().__init__()
        self.model = FNet(n_out_channels=model_params['n_classes'], norm_type=model_params['norm_type'])
        self.n_classes = model_params['n_classes']

    def forward(self, input, **kwargs):
        b, c, h, w = input.shape
        assert b

        if kwargs.get("center_input", True):
            #print("zscore input")
            input = MinMaxTargetPreprocessor.do_z_score(input)

        return self.model(input)

    def load(self, params):
        epoch = 0
        cp_folder = params.get_checkpoint_folder(create=False)
        if not cp_folder.exists():
            return False

        if params.model.load:
            cp = cp_folder.load_last_training_checkpoint(params.model.id, strict=False)
            if cp is not None:
                print("CP loaded")
                self.load_state_dict(cp['model_state_dict'])
                epoch = cp['epoch']
                return True

        return False
