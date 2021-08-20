import gym
import matplotlib.pyplot as plt
import numpy as np

from rl.envs.configurations import action_spaces
from rl.envs.configurations.goals import AbsDistGoal

from rl.envs.gym_envs.color_mix_env_3d import ExperimentConfig, ColorMix3D
from utils.fluids.envs.env_criteria.dones import Never
from utils.fluids.envs.env_criteria.goals import RandomGoalGenerator, DeterministicGoalGenerator, GoalCriterion
from utils.fluids.envs.env_criteria.step_limit import StepLimitCounter
from utils.fluids.envs.env_state.fluid_env_initializers import RandomSameInitializer
from utils.fluids.envs.env_state.state_extractors import MultiChanSimStateExtractor
from utils.fluids.envs.env_utilities import IntervalPumpManager
from utils.fluids.envs.fluid_env_v2 import FluidEnvBaseV2
from utils.fluids.simulation_devices import Fluid


class ColorMix3DContinuous(ColorMix3D):
    upper_lim = 2
    lower_lim = -2

    def _get_pump_mapping(self):
        pump_mapping = {
            'blue': {'port': "/dev/ttyUSB1", 'channel': 1, 'fluid_type': Fluid([0, 0, 0.5, 0.5], "color")},
            'red': {'port': "/dev/ttyUSB0", 'channel': 0, 'fluid_type': Fluid([0.5, 0, 0, 0.5], "color")}
        }
        return pump_mapping

    def _get_action_space(self):
        num_pumps = len(self._get_pump_mapping())
        low = np.zeros(num_pumps) - self.lower_lim
        hi = np.zeros(num_pumps) + self.upper_lim

        return gym.spaces.Box(low, hi)

    def _get_env(self, sim_config, exp_config):
        self.pump_mapping = self._get_pump_mapping()

        goal_gen = RandomGoalGenerator(min_goal_val=100, max_goal_val=200)
        # goal_gen = DeterministicGoalGenerator([[190]], allow_index_overflow=True)
        # goal_gen.advance_goal_idx()

        return FluidEnvBaseV2(sim_config=sim_config,
                              pump_manager=IntervalPumpManager(
                                  self.pump_mapping,
                                  schedule_on_time=exp_config.INF_DURATION_PER_CYCLE),
                              state_extractor=MultiChanSimStateExtractor(3),
                              done_criterion=Never(),
                              reward_provider=self,
                              goal_criterion=AbsDistGoal(goal_generator=goal_gen),
                              action_handler=action_spaces.Continuous(),
                              max_step_counter=StepLimitCounter(100),
                              # initializer=RandomSameInitializer(grid_min=244, grid_max=245),
                              initializer=RandomSameInitializer(grid_min=100, grid_max=200),
                              return_raw_state=True)

    def step(self, actions):
        # print(actions)
        actions = (actions + self.upper_lim) / (self.upper_lim - self.lower_lim)
        actions *= 300
        actions = np.round(actions).astype(np.int)
        # print(actions)
        r = super(ColorMix3DContinuous, self).step(actions)
        # print(r)
        # print()
        return r

    def _render_for_plt(self, **kwargs):

        if self.plt_cache is None:
            self.plt_cache = plt.subplots(1, 2, figsize=(12, 5))
            pass

        f, ax = self.plt_cache

        ax[0].cla()
        ax[1].cla()

        row_labels = ['Red', 'Green', 'Blue']
        column_labels = ['Goal val.', 'Current val.']

        sd = self.last_step_data
        if sd is None:
            print("No step data available for rendering")
            return
        a_val = sd.action_value
        a_ch = sd.action_ch

        dg, ag, obs = self._get_obs(sd.next_state)

        bar_heights = a_val

        labels = []
        for idx in range(len(bar_heights)):
            for key, value in self.pump_mapping.items():
                if value['channel'] == idx:
                    labels.append(key)

        # labels =[f"{list(self.pump_mapping.keys())[idx]}" for idx in range(len(bar_heights))]

        ax[1].barh(np.arange(self.env.action_handler.pump_count), bar_heights, color="k")
        y_vals = self.env.action_handler.actions[
                 :len(self.env.action_handler.actions) // self.env.action_handler.pump_count]

        ax[1].set_yticks(np.arange(len(bar_heights)))
        ax[1].set_yticklabels(labels)
        ax[1].set_xticks(y_vals)
        ax[1].set_xticklabels(y_vals, rotation=45, fontsize=8)

        ax[1].invert_yaxis()

        table_data = [
            [*dg, f"{obs[-3]:.2f}"],
            ["-", f"{obs[-2]:.2f}"],
            ["-", f"{obs[-1]:.2f}"]
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
    env = gym.make('envs:ColorMix3DContinuous-v0')

    dgg = DeterministicGoalGenerator([[140], [180]], allow_index_overflow=True)
    dgg.advance_goal_idx()
    env.env.goal_criterion.goal_generator = dgg
    state = env.reset()
    a_space = env.action_space
    obs_space = env.observation_space
    actions = np.ones(a_space.shape)
    deltas = []
    state = env.reset()
    for i in range(500):

        actions = np.asarray([-1, -1])

        s = state['observation']
        g = state['desired_goal']

        if s[0] < g:
            actions[0] = 1
        else:
            actions[1] = 1

        obs, reward, done, info = env.step(actions)

        sd = env.last_step_data

        deltas.append(sd.curr_state.device_state - sd.next_state.device_state)
        env.render('plt', wait_delay=0.1)
        state = obs
        if done:
            break


if __name__ == '__main__':
    run()
