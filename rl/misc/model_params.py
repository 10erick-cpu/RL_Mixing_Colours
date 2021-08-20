import json
import os

from stable_baselines import DQN, HER, PPO2#, PPO1
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.her import HERGoalEnvWrapper

from utils.helper_functions.misc_utils import timestamp_now_str
from utils.models.dot_dict import DotDict


def class_for_param(p_type):
    if p_type == "dqn":
        return DQNParams
    elif p_type == "ppov1":
        return PPOv1Params
    elif p_type == "ppov2":
        return PPOv2Params
    raise KeyError("unknown type", p_type)


class Params(DotDict):

    def __init__(self, model_type, description=None, **kwargs):
        self.model_type = model_type
        self.lr = None
        self.layers = None
        self.gamma = 0.99
        self.description = description
        self._set_defaults()
        super().__init__(**kwargs)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self, f)

    def copy(self):
        data = dict(**self)
        del data['model_type']
        return class_for_param(self.model_type)(**data)

    @staticmethod
    def load(path):
        print("load path", path)
        assert os.path.exists(path)
        with open(path, 'rb') as f:
            obj = json.load(f)

        if 'model' in obj:
            p_type = obj['model']
            del obj['model']
        else:
            p_type = obj['model_type']
            del obj['model_type']
        clz = class_for_param(p_type)

        params = clz(**obj)
        # params.replay = DotDict(params.replay)
        return params

    def _set_defaults(self):
        raise NotImplementedError()

    def short_name(self):
        raise NotImplementedError()

    def tensorboard_id(self):
        return timestamp_now_str() + "_" + self.short_name()

    def build_env(self, env_selection_cls):
        raise NotImplementedError()

    def get_model(self, env, tensorboard_log_dir=None):
        raise NotImplementedError()


class DQNParams(Params):
    def __init__(self, **kwargs):
        super(DQNParams, self).__init__('dqn', **kwargs)

    def _set_defaults(self):
        self.layers = [64, 64]
        self.batch_size = 32
        self.lr = 5e-4

        self.target_network_updates = 500
        self.replay = DotDict({'type': 'default'})
        self.learning_start = 0
        self.buffer_size = 50000
        self.double_q = True
        self.dueling = False
        self.layer_norm = False

    def plot_name(self):
        name = f"{self.model_type}|{self.replay['type']}"
        return name

    def __repr__(self):
        return self.short_name()

    def short_name(self):
        if self.description is not None:
            return self.description
        name = f"agent={self.model_type}_replay={self.replay['type']}_layers={str(self.layers)}_duel={self.dueling}"
        name += f"_ln={self.layer_norm}_bsize={self.batch_size}_lr={self.lr}"
        return name

    def uses_flat_env(self):
        return self.replay['type'] != 'her'

    def build_env(self, env_selection_cls):
        if self.uses_flat_env():
            return env_selection_cls.get_flatten_env()
        return env_selection_cls.get_plain_env()

    def use_her(self, mode='future', k=4):
        self.replay = DotDict({'type': 'her', 'goal_selection_strategy': mode, 'n_sampled_goal': k})

    def use_exp_replay(self, alpha=0.6, beta0=0.4, eps=1e-6, beta_iters=None):
        self.replay = DotDict({'type': 'prioritized',
                               'prioritized_replay': True,
                               'prioritized_replay_alpha': alpha,
                               'prioritized_replay_beta0': beta0,
                               'prioritized_replay_eps': eps,
                               'prioritized_replay_beta_iters': beta_iters
                               })

    def get_model(self, env, tensorboard_log_dir=None):
        policy_type = 'LnMlpPolicy' if self.layer_norm else 'MlpPolicy'
        replay_type = self.replay['type']

        cleaned_replay_params = DotDict(self.replay)
        del cleaned_replay_params['type']

        if replay_type == 'default':
            return DQN(policy_type, env, learning_rate=self.lr,
                       gamma=self.gamma,
                       buffer_size=self.buffer_size,
                       double_q=self.double_q,
                       batch_size=self.batch_size,
                       learning_starts=self.learning_start,
                       target_network_update_freq=self.target_network_updates,
                       policy_kwargs={'dueling': self.dueling, 'layers': self.layers},
                       tensorboard_log=tensorboard_log_dir
                       )
        elif replay_type == 'prioritized':
            return DQN(policy_type, env, learning_rate=self.lr,
                       buffer_size=self.buffer_size,
                       gamma=self.gamma,
                       batch_size=self.batch_size,
                       learning_starts=self.learning_start,
                       target_network_update_freq=self.target_network_updates,
                       double_q=self.double_q,
                       **cleaned_replay_params,
                       policy_kwargs={'dueling': self.dueling, 'layers': self.layers},
                       tensorboard_log=tensorboard_log_dir
                       )

        elif replay_type == 'her':

            return HER(policy_type, HERGoalEnvWrapper(env), learning_rate=self.lr,
                       model_class=DQN,
                       gamma=self.gamma,
                       double_q=self.double_q,
                       buffer_size=self.buffer_size,
                       batch_size=self.batch_size,
                       learning_starts=self.learning_start,
                       target_network_update_freq=self.target_network_updates,
                       **cleaned_replay_params,
                       policy_kwargs={'dueling': self.dueling, 'layers': self.layers},
                       tensorboard_log=tensorboard_log_dir
                       )
        else:
            raise KeyError("Unknown replay type", self.replay.type)


class PPOv1Params(Params):

    def _set_defaults(self):
        self.timesteps_per_actorbatch = 300
        self.clip_param = 0.2
        self.entcoeff = 0.01
        self.optim_epochs = 4
        self.optim_stepsize = 1e-3
        self.optim_batchsize = 64
        self.lam = 0.95
        self.adam_epsilon = 1e-5
        self.schedule = 'linear'

    def __init__(self, **kwargs):
        super().__init__('ppov1', **kwargs)

    def short_name(self):
        if self.description is not None:
            return self.description
        name = f"agent={self.model_type}"
        return name

    def build_env(self, env_selection_cls):
        return env_selection_cls.get_flatten_env()

    def get_model(self, env, tensorboard_log_dir=None):

        policy_type = 'LnMlpPolicy' if self.layer_norm else 'MlpPolicy'
        return PPO1(policy_type, env,
                    timesteps_per_actorbatch=self.timesteps_per_actorbatch,
                    gamma=self.gamma,
                    clip_param=self.clip_param,
                    entcoeff=self.entcoeff,
                    optim_epochs=self.optim_epochs,
                    optim_batchsize=self.optim_batchsize,
                    optim_stepsize=self.optim_stepsize,
                    lam=self.lam,
                    adam_epsilon=self.adam_epsilon,
                    schedule=self.schedule,
                    policy_kwargs={'layers': self.layers})


class PPOv2Params(Params):

    def _set_defaults(self):
        self.n_steps = 128
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.lam = 0.95
        self.nminibatches = 4
        self.noptepochs = 4
        self.cliprange = 0.2
        self.cliprange_vf = None
        self.lr = 2.5e-4

    def __init__(self, **kwargs):
        super().__init__('ppov2', **kwargs)

    def short_name(self):
        if self.description is not None:
            return self.description
        name = f"agent={self.model_type}"
        return name

    def build_env(self, env_selection_cls):
        return env_selection_cls.get_flatten_env()

    def get_model(self, env, tensorboard_log_dir=None):
        policy_type = 'LnMlpPolicy' if self.layer_norm else 'MlpPolicy'
        return PPO2(policy_type, DummyVecEnv([lambda :env]),
                    gamma=self.gamma,
                    n_steps=self.n_steps,
                    ent_coef=self.ent_coef,
                    learning_rate=self.lr,
                    vf_coef=self.vf_coef,
                    max_grad_norm=self.max_grad_norm,
                    lam=self.lam,
                    nminibatches=self.nminibatches,
                    noptepochs=self.noptepochs,
                    cliprange=self.cliprange,
                    cliprange_vf=self.cliprange_vf,
                    policy_kwargs={'layers': self.layers})
