import datetime
import os

import torch
from utils.models.folder import Folder


class CheckpointFolder(Folder):
    def __init__(self, dir_path, strict_exist=True):
        super().__init__(dir_path)
        self.model_templ_str = "id_{}_cp_epoch_{:05d}_{:%Y%m%d}.pth"
        self.training_progr_templ_str = "id_{}_cp_training_epoch_{:05d}_{:%Y%m%d}.tar"
        if not self.exists():
            if strict_exist:
                raise KeyError("Checkpoint folder does not exist", self.path_abs)
            else:
                print("Creating non-existent checkpoint folder at {}".format(self.path_abs))
                self.make_dir()

    def save_state_dict(self, model_id, epoch, state_dict):
        save_path = self.get_file_path(self.training_progr_templ_str.format(model_id, epoch, datetime.datetime.now()))

        torch.save(state_dict, save_path)

    def save_training_progress(self, model_id, epoch, model, optimizer=None):

        state = dict()
        state['epoch'] = epoch
        state['model_state_dict'] = model.state_dict()
        if optimizer:
            state[f'optimizer_state_dict_{optimizer.__class__.__name__}'] = optimizer.state_dict()
        self.save_state_dict(model_id, epoch, state_dict=state)

    # def save_training_progress(self, model_id, state_dict, epoch):
    #     save_path = self.get_file_path(self.training_progr_templ_str.format(model_id, epoch, datetime.datetime.now()))
    #     torch.save(state_dict, save_path)

    def load_last_training_checkpoint(self, model_id, strict=True):
        checkpoints = self.find_checkpoints('tar', model_id, strict=False)
        if len(checkpoints) == 0:
            if strict:
                raise KeyError("Unable to load latest training progress: no file found", self.path(), model_id)
            else:
                return None

        cp_file_path = self.get_file_path(checkpoints[0])
        checkpoint = torch.load(cp_file_path, map_location={'cuda:0': 'cpu'})
        print("Checkpoint loaded:", cp_file_path)
        return checkpoint

    def save_model(self, model_id, epoch, model):
        state = model.state_dict()
        path = self.get_file_path(self.model_templ_str.format(model_id, epoch, datetime.datetime.now()))
        torch.save(state, path)

    def load_latest_model_cp(self, model_id, abs_path=True):
        checkpoints = self.find_checkpoints('pth', model_id, strict=False)
        if len(checkpoints) == 0:
            print("Unable to load latest model: no models found")
            return False
        # self.load_model(target_model, self.get_file_path(checkpoints[0]))
        return self.get_file_path(checkpoints[0])

    def list_checkpoints(self, model_id):
        return self.find_checkpoints('pth', model_id, strict=False)

    def find_checkpoints(self, extension, model_id, strict=True):
        checkpoints = sorted([f for f in self.get_files([extension], False) if f.lower().startswith("id_" + model_id)],
                             reverse=True)
        if strict and len(checkpoints) == 0:
            raise KeyError("Unable to load latest model: no models found")
        return checkpoints

    def load_model(self, model, path):
        print("Loading model {}".format(path))
        model.load_state_dict(torch.load(path))
        return model

    def cleanup_checkpoints(self, model_id, num_keep=3):
        cps_tp = self.find_checkpoints("tar", model_id, strict=False)
        cps_model = self.find_checkpoints("pth", model_id, strict=False)
        removed = []
        if len(cps_tp) > num_keep:
            remove = cps_tp[3:]
            for cp in remove:
                print("clean up", cp)
                os.remove(self.get_file_path(cp))
                removed.append(cp)

        if len(cps_model) > num_keep:
            remove = cps_model[3:]
            for cp in remove:
                print("clean up", cp)
                os.remove(self.get_file_path(cp))
                removed.append(cp)
        return removed
