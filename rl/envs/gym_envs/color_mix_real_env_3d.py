import json
import pickle
import time

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np

from rl.envs.configurations import action_spaces
from rl.envs.configurations.goals import AbsDistGoal
from rl.envs.gym_envs.color_mix_env_3d import ColorMix3D

from utils.fluids.envs.env_criteria.dones import Never
from utils.fluids.envs.env_criteria.goals import RandomGoalGenerator
from utils.fluids.envs.env_criteria.step_limit import StepLimitCounter
from utils.fluids.envs.real_env.real_env import RealEnvironment
from utils.fluids.simulation_devices import Fluid
from utils.fluids.time_providers import RealTime
from utils.helper_functions.misc_utils import timestamp_to_str
from utils.models.folder import Folder

def process_config(cfg):
    if 'pump_mapping' in cfg:
        obj_dict = dict()

        for idx, mapping in enumerate(cfg['pump_mapping'].keys()):
            mixture = cfg['pump_mapping'][mapping]['mixture']
            fluid_type = cfg['pump_mapping'][mapping]['fluid_type']
            obj_dict[mapping] = dict()

            obj_dict[mapping]['port'] = cfg['pump_mapping'][mapping]['port']
            obj_dict[mapping]['channel'] = cfg['pump_mapping'][mapping]['channel']
            obj_dict[mapping]['fluid_type'] = Fluid(mixture, fluid_type)
        cfg['pump_mapping'] = obj_dict
    return cfg


def load_config():
    import os

    folder = Folder(os.path.dirname(os.path.abspath(__file__)))
    cfg = folder.get_file_path("env_config_colormix_real.json")
    with open(cfg, 'r') as f:
        return json.load(f)


class ColorMixReal3D(ColorMix3D):
    def _get_pump_mapping(self):
        raise ValueError("Should be loaded from json config file")

    def _get_env(self, sim_config, exp_config):

        cfg = load_config()

        cfg = process_config(cfg)
        goal_gen = RandomGoalGenerator(min_goal_val=110, max_goal_val=200)

        self.pump_mapping = cfg['pump_mapping']

        camera_index = cfg['camera_index']

        exp_config.INF_DURATION_PER_CYCLE = cfg.get('inf_duration_per_cycle', exp_config.INF_DURATION_PER_CYCLE)
        exp_config.CYCLE_DURATION = cfg.get('cycle_duration', exp_config.CYCLE_DURATION)

        sim_config.SIMULATION_SECONDS_STEPS = exp_config.CYCLE_DURATION
        sim_config.TIME_PROVIDER = RealTime()

        action_handler = action_spaces.SimpleDiscrete()
        return RealEnvironment(
            sim_config=sim_config,
            done_criterion=Never(),
            pump_mapping=self.pump_mapping,
            flush_port=cfg['flush_port'],
            reward_provider=self,
            goal_criterion=AbsDistGoal(goal_generator=goal_gen),
            action_handler=action_handler,
            max_step_counter=StepLimitCounter(cfg.get('step_limit', 100)),
            inf_duration_per_cycle=exp_config.INF_DURATION_PER_CYCLE,
            camera_obs_id=camera_index,
            real_mode=True
        )

    @staticmethod
    def perform_range_test(env, save_path=None, display=False):
        np.set_printoptions(suppress=True)

        action_data = []

        actions = [7, 15, 7, 15, 7, 15]
        for action_ch, action in enumerate(actions):
            print(action_ch, action)

        for action_ch, action in enumerate(actions):
            action_data.append([])
            print("range test action", action)
            state = env.reset(reset_duration=0)
            done = False

            env.env.max_step_counter.max_steps = 60
            while not done:
                state, reward, done, info = env.step(action)
                step_data = info['step_data']

                assert step_data.action_value == 300, step_data.action_value

                ds = step_data.next_state.device_state
                action_data[action_ch].append(ds)

                delta = step_data.curr_state.device_state - step_data.next_state.device_state
                print(f"\r current state {step_data.next_state.device_state}", end="")
                if False and not done and step_data.step_count > 10 and abs(delta).sum() < 1:
                    done = True

        if save_path is None:
            import matplotlib.pyplot as plt
            for action_ch in range(len(action_data)):
                data = np.asarray(action_data[action_ch])
                plt.title(f"{action_ch}")
                plt.plot(data[:, 0], color="r")
                plt.plot(data[:, 1], color="g")
                plt.plot(data[:, 2], color="b")
                plt.ylim(bottom=0, top=265)
                plt.show()
        else:
            with open(save_path, 'wb') as f:
                pickle.dump(action_data, f)

    def close(self):
        super(ColorMixReal3D, self).close()
        self.env.pump_manager.stop()

    def _render_for_plt(self, **kwargs):
        if self.plt_cache is None:
            self.plt_cache = plt.subplots(1, 3, figsize=(12, 5))

        result = super(ColorMixReal3D, self)._render_for_plt(table_size=6, **kwargs)
        f, ax = self.plt_cache

        real_obs, target_area = self.env.camera_observer.capture_observation(draw_rect=True)

        real_obs = cv2.resize(real_obs, None, None, 0.2, 0.2)

        ax[2].imshow(real_obs)
        ax[2].axis('off')
        return self.plt_cache


if __name__ == '__main__':

    timestamp = timestamp_to_str(time.time())
    fname = f"./real_env_test_{timestamp}_flush_blue.pkl"
    print(fname)

    env = gym.make('envs:ColorMixReal3D-v0')
    a_space = env.action_space
    obs_space = env.observation_space

    #
    # ColorMixReal3D.perform_range_test(env, fname)
    #
    # raise Exception("done")

    mode = "rgb_array"
    for action in range(a_space.n):
        deltas = []
        state = env.reset()
        for i in range(500):

            obs, reward, done, info = env.step(action)

            sd = env.last_step_data
            print("reward", reward)

            deltas.append(sd.curr_state.device_state - sd.next_state.device_state)
            env.render(mode, wait_delay=0.1)
            if sd.action_value in [32, 64, 128, 256]:
                env.render(mode, wait_delay=0.5)
                # time.sleep(60)

            state = obs
            if done:
                break

        deltas = np.asarray(deltas)
        print("Action ", action)
        print(deltas.mean(), deltas.std())
