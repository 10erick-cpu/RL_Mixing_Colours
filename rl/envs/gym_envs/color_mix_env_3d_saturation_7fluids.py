
import gym
import numpy as np
from rl.envs.gym_envs.color_mix_env_3d_saturation import ColorMix3DSaturation
from utils.fluids.envs.env_criteria.goals import DeterministicGoalGenerator
from utils.fluids.simulation_devices import Fluid


class ColorMix3DSaturation7Fluids(ColorMix3DSaturation):

    def _get_pump_mapping(self):
        pump_mapping = {
            'blue': {'port': "/dev/ttyUSB1", 'channel': 1,
                     'fluid_type': Fluid([0, 0, 0.5, 0.5],
                                         "color",
                                         saturation_window=120, saturation_inf_sum=240)},
            'red': {'port': "/dev/ttyUSB0", 'channel': 0,
                    'fluid_type': Fluid([0.5, 0, 0, 0.5],
                                        "color",
                                        saturation_window=120, saturation_inf_sum=240)},
            # 'red_green': {'port': "/dev/ttyUSB2", 'channel': 2,
            #               'fluid_type': Fluid([0.25, 0.0, 0.0, 0.5],
            #                                   "color",
            #                                   saturation_window=120, saturation_inf_sum=240)},
            'g': {'port': "/dev/ttyUSB2", 'channel': 2, 'fluid_type': Fluid([0, 0.5, 0, 0.5], "color")},
            'rg': {'port': "/dev/ttyUSB3", 'channel': 3,
                   'fluid_type': Fluid([0.45, 0.05, 0, 0.5], "color", saturation_window=120, saturation_inf_sum=240)},
            'rb': {'port': "/dev/ttyUSB4", 'channel': 4,
                   'fluid_type': Fluid([0.45, 0, 0.05, 0.5], "color", saturation_window=120, saturation_inf_sum=240)},
            'gb': {'port': "/dev/ttyUSB5", 'channel': 5,
                   'fluid_type': Fluid([0, 0.45, 0.05, 0.5], "color", saturation_window=120, saturation_inf_sum=240)},
            'w': {'port': "/dev/ttyUSB6", 'channel': 6,
                  'fluid_type': Fluid([0, 0.0, 0.0, 1], "color", saturation_window=120, saturation_inf_sum=240)}
        }
        return pump_mapping

def run():
    env = gym.make('envs:ColorMix3DSaturated-v1')

    dgg = DeterministicGoalGenerator([[200]], allow_index_overflow=True)
    dgg.advance_goal_idx()
    env.env.goal_criterion.goal_generator = dgg
    state = env.reset()
    a_space = env.action_space
    obs_space = env.observation_space

    for action in range(a_space.n):
        deltas = []
        state = env.reset()
        for i in range(500):

            action = 7 if i < 20 else 14
            obs, reward, done, info = env.step(action)

            sd = env.last_step_data

            deltas.append(sd.curr_state.device_state - sd.next_state.device_state)
            env.render('plt', wait_delay=0.5)
            state = obs
            if done:
                break

        deltas = np.asarray(deltas)
        print("Action ", action)
        print(deltas.mean(), deltas.std())

if __name__ == '__main__':
    run()
