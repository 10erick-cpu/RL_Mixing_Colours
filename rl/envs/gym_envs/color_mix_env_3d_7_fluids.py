import gym
import numpy as np

from rl.envs.gym_envs.color_mix_env_3d import ColorMix3D
from utils.fluids.envs.env_criteria.goals import DeterministicGoalGenerator
from utils.fluids.simulation_devices import Fluid


class ColorMix3D7Fluid(ColorMix3D):

    def _get_pump_mapping(self):
        pump_mapping = {

            'r': {'port': "/dev/ttyUSB0", 'channel': 0, 'fluid_type': Fluid([0.5, 0, 0, 0.5], "color")},
            'g': {'port': "/dev/ttyUSB1", 'channel': 1, 'fluid_type': Fluid([0, 0.5, 0, 0.5], "color")},
            'b': {'port': "/dev/ttyUSB2", 'channel': 2, 'fluid_type': Fluid([0, 0, 0.5, 0.5], "color")},

            'rg': {'port': "/dev/ttyUSB3", 'channel': 3, 'fluid_type': Fluid([0.45, 0.5, 0, 0.5], "color")},
            'rb': {'port': "/dev/ttyUSB4", 'channel': 4, 'fluid_type': Fluid([0.45, 0, 0.05, 0.5], "color")},
            'gb': {'port': "/dev/ttyUSB5", 'channel': 5, 'fluid_type': Fluid([0, 0.45, 0.05, 0.5], "color")},
            'w': {'port': "/dev/ttyUSB6", 'channel': 6, 'fluid_type': Fluid([0, 0.0, 0.0, 1], "color")}
        }
        return pump_mapping


def run():
    env = gym.make('envs:ColorMix3D-v1')

    # out_path = "./ColorMix3D_color_range.pkl"
    #
    # ColorMix3D.perform_range_test(env, save_path=out_path)
    #
    # return

    dgg = DeterministicGoalGenerator([[155], [120], [175]],
                                     allow_index_overflow=True)
    env.env.goal_criterion.goal_generator = dgg
    dgg.advance_goal_idx()
    state = env.reset()
    a_space = env.action_space
    obs_space = env.observation_space
    print(a_space)

    for action in range(a_space.n):
        deltas = []
        state = env.reset()
        dgg.advance_goal_idx()
        for i in range(500):
            action = 7 if i <= 30 else 15
            if action == 15 and i >= 45:
                action = 4 * 7
            obs, reward, done, info = env.step(53)

            sd = env.last_step_data

            deltas.append(sd.curr_state.device_state - sd.next_state.device_state)
            env.render('plt', wait_delay=0.2)
            if True or sd.action_value in [32, 64, 128, 256]:
                pass
                # time.sleep(60)

            state = obs
            if done:
                break

        deltas = np.asarray(deltas)
        print("Action ", action)
        print(deltas.mean(), deltas.std())


if __name__ == '__main__':
    run()
