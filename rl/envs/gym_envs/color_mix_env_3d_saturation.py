import gym
import matplotlib.pyplot as plt
import numpy as np

from rl.envs.gym_envs.color_mix_env_3d import ColorMix3D
from rl.envs.gym_envs.color_mix_env_3d import ExperimentConfig
from utils.fluids.envs.env_criteria.goals import DeterministicGoalGenerator
from utils.fluids.simulation_devices import Fluid


class ColorMix3DSaturation(ColorMix3D):

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
            'red_green': {'port': "/dev/ttyUSB2", 'channel': 2,
                          'fluid_type': Fluid([0.45, 0.05, 0.0, 0.5],
                                              "color",
                                              saturation_window=120, saturation_inf_sum=240)}
        }
        return pump_mapping

    def _get_obs(self, state):
        ag = np.asarray([state.device_state[0]])
        dg = state.goal_state
        obs = np.concatenate((state.device_state, state.pump_history))
        return dg, ag, obs

    def _render_for_plt(self, **kwargs):

        if self.plt_cache is None:
            self.plt_cache = plt.subplots(1, 3, figsize=(12, 5))
            pass

        f, ax = self.plt_cache

        ax[0].cla()
        ax[1].cla()
        ax[2].cla()

        row_labels = ['Red', 'Green', 'Blue']
        column_labels = ['Goal val.', 'Current val.']

        sd = self.last_step_data
        if sd is None:
            print("No step data available for rendering")
            return
        a_val = sd.action_value
        a_ch = sd.action_ch

        dg, ag, obs = self._get_obs(sd.next_state)

        bar_heights = [0 for i in range(self.env.action_handler.pump_count)]

        bar_heights[a_ch] = a_val

        pump_history = sd.next_state.pump_history

        labels = []

        fluid_sat = []
        for idx in range(len(bar_heights)):
            for key, value in self.pump_mapping.items():
                if value['channel'] == idx:
                    labels.append(key)
                    fluid_sat.append(value['fluid_type'].effectivity(pump_history[idx]))

        ax[2].bar(np.arange(len(pump_history)), fluid_sat)
        ax[2].set_title("fluid infusion effectivity")

        # labels =[f"{list(self.pump_mapping.keys())[idx]}" for idx in range(len(bar_heights))]

        ax[1].bar(np.arange(self.env.action_handler.pump_count), bar_heights, color="k")
        action_names = self.env.action_handler.actions[
                       :len(self.env.action_handler.actions) // self.env.action_handler.pump_count]

        ax[1].set_xticks(np.arange(len(bar_heights)))
        ax[1].set_xticklabels(labels)
        ax[1].set_yticks(action_names)
        ax[1].set_ylim([0, 300])

        ax[1].set_yticklabels(action_names)
        ax[1].set_title("current infusion")

        # ax[1].invert_yaxis()

        s_r, s_g, s_b = obs[:3]

        h = obs[3:]

        table_data = [
            [*dg, f"{s_r:.2f}"],
            ["-", f"{s_g:.2f}"],
            ["-", f"{s_b:.2f}"]
        ]

        table = ax[0].table(rowLabels=row_labels, colLabels=column_labels,
                            cellText=table_data,
                            loc="center", colWidths=[0.2 for x in column_labels],
                            cellLoc="center", edges="vertical")

        ax[0].set_title(f"{sd.step_count * ExperimentConfig.CYCLE_DURATION} seconds elapsed (step {sd.step_count}/{sd.max_steps})")

        # table.auto_set_font_size(False)
        table.scale(2, 4)
        # table.set_fontsize(kwargs.get("table_size", 5))

        ax[0].axis('off')
        return plt.gcf(), plt.gca()


def run():
    env = gym.make('envs:ColorMix3DSaturated-v0')

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

            action = 25
            obs, reward, done, info = env.step(action)

            sd = env.last_step_data

            deltas.append(sd.curr_state.device_state - sd.next_state.device_state)
            env.render('plt', wait_delay=0.1)
            state = obs
            if done:
                break

        deltas = np.asarray(deltas)
        print("Action ", action)
        print(deltas.mean(), deltas.std())


if __name__ == '__main__':
    run()
