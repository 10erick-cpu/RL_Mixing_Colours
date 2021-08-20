import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from rl.envs.configurations import action_spaces
from rl.envs.configurations.goals import AbsDistGoal
from utils.fluids.envs.env_criteria.dones import Never
from utils.fluids.envs.env_criteria.goals import RandomGoalGenerator, DeterministicGoalGenerator, GoalCriterion
from utils.fluids.envs.env_criteria.rewards import RewardProvider
from utils.fluids.envs.env_criteria.step_limit import StepLimitCounter
from utils.fluids.envs.env_state.fluid_env_initializers import RandomSameInitializer, SameInitializer
from utils.fluids.envs.env_state.state_extractors import MultiChanSimStateExtractor
from utils.fluids.envs.env_stepper import StepData
from utils.fluids.envs.env_utilities import IntervalPumpManager
from utils.fluids.envs.fluid_env_v2 import FluidEnvBaseV2
from utils.fluids.envs.fluid_simulator import SimulatorConfig
from utils.fluids.simulation_devices import ColorMixTestDevice, Fluid, ColorMixer
from utils.fluids.simulation_engines import BasicMeanEngine
from utils.fluids.time_providers import TimeProvider, SimulatedTime


class ExperimentConfig(object):
    CYCLE_DURATION = 6
    INF_DURATION_PER_CYCLE = 3


def build_config(tp: TimeProvider = SimulatedTime()):
    simulator_config = SimulatorConfig()

    simulator_config.TIME_PROVIDER = tp

    simulator_config.SIMULATION_SECONDS_STEPS = ExperimentConfig.CYCLE_DURATION

    simulator_config.NUM_CHANNELS = 3
    simulator_config.ENGINE.type = BasicMeanEngine()
    simulator_config.DEVICE = ColorMixTestDevice(0)

    return simulator_config


class ColorMix3D(gym.GoalEnv, RewardProvider):

    def _get_pump_mapping(self):
        pump_mapping = {
           'b': {'port': "COM15", 'channel': 1, 'fluid_type': Fluid([0, 0, 0.5, 0.5], "color")},
           'r': {'port': "COM11", 'channel': 0, 'fluid_type': Fluid([0.5, 0, 0, 0.5], "color")}
           # 'b': {'port': "/dev/ttyUSB1", 'channel': 1, 'fluid_type': Fluid([0, 0, 0.5, 0.5], "color")},
           # 'r': {'port': "/dev/ttyUSB0", 'channel': 0, 'fluid_type': Fluid([0.5, 0, 0, 0.5], "color")}
        }
        return pump_mapping

    def pump_manager(self, pump_manager=None):
        return self.env.set_pump_manager(pump_manager)

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
                              action_handler=action_spaces.SimpleDiscrete(),
                              max_step_counter=StepLimitCounter(100),
                              # initializer=RandomSameInitializer(grid_min=244, grid_max=245),
                              initializer=RandomSameInitializer(grid_min=100, grid_max=200),
                              return_raw_state=True)

    def _get_action_space(self):
        return gym.spaces.Discrete(len(self.env.action_space()))

    def set_mixer(self, mixer: ColorMixer):
        self.env.sim.device_state.set_color_mixer(mixer)

    def __init__(self):
        super().__init__()
        self.last_step_data: StepData = None
        self.env = self._get_env(build_config(SimulatedTime()), ExperimentConfig)

        obs = self.reset(reset_duration=0)

        self.action_space = self._get_action_space()

        self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=gym.spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        self.plt_cache = None

    def goal_generator(self, gen=None):
        if gen is None:
            return self.env.goal_criterion.goal_generator
        else:
            self.env.goal_criterion.goal_generator = gen

    def initializer(self, init=None):
        if init is None:
            return self.env.initializer
        else:
            self.env.initializer = init

    def get_reward(self, state, action, next_state):
        return 0

    def terminal_reward_max_step_count(self, data: StepData):
        return 0

    def compute_reward(self, achieved_goal, desired_goal, info, add_amount_penalty=False):
        # dist_penalty = -abs(
        #     F.smooth_l1_loss(torch.from_numpy(desired_goal).float(), torch.from_numpy(achieved_goal).float(),
        #                      reduction="sum").numpy().item())

        # dist_penalty = -abs(
        #     F.mse_loss(torch.from_numpy(desired_goal).float(), torch.from_numpy(achieved_goal).float(),
        #                      reduction="sum").numpy().item())

        dist_penalty = -abs(
            F.l1_loss(torch.from_numpy(desired_goal).float(), torch.from_numpy(achieved_goal).float(),
                      reduction="sum").numpy().item())
        # dist_penalty = -abs((desired_goal - achieved_goal)).mean()
        if not add_amount_penalty:
            return dist_penalty
        amount_penalty = -info['step_data'].action_value / 300
        return dist_penalty + amount_penalty

    def _get_obs(self, state):
        ag = np.asarray([state.device_state[0]])
        dg = state.goal_state
        # obs = np.concatenate([step_data.next_state.pump_states, step_data.next_state.device_state])
        obs = state.device_state
        return dg, ag, obs

    def step(self, action):
        # start = time.time()

        step_data = self.env.step(action)

        self.last_step_data = step_data

        dg, ag, obs = self._get_obs(step_data.next_state)
        # obs, reward, done, info
        info = {'step_data': step_data}
        reward = self.compute_reward(ag, dg, info)

        # print("\rstep took {}s".format(time.time() - start), end="")
        return self.__make_goal_dict(ag, dg, obs), reward, step_data.reset_required, info

    def __make_goal_dict(self, ag, dg, obs):
        return {'achieved_goal': ag, 'desired_goal': dg, 'observation': obs}

    def reset(self, **kwargs):
        self.last_step_data = None
        # self.plt_cache = None

        raw_obs, data = self.env.reset(**kwargs)

        dg, ag, obs = self._get_obs(data)

        return self.__make_goal_dict(ag, dg, obs)

    @staticmethod
    def perform_range_test(env, save_path=None, display=False, num_runs=1):
        num_actions = env.action_space.n
        np.set_printoptions(suppress=True)
        env.env.initializer = SameInitializer(value=0)

        action_data = []

        actions = []

        grid_states = []
        for i in range(num_runs):
            actions.append(8)
            actions.append(17)

        base_intensity = 180

        for action_ch, action in enumerate(actions):
            action_data.append([])
            grid_states.append([])
            print("range test action", action)
            state = env.reset()
            init_val = np.asarray([100, 100, base_intensity]) if action_ch % 2 == 0 else np.asarray([base_intensity, 100, 100])

            env.env.sim.device_state.reset_fixed_channel_wise(init_val)
            done = False

            env.env.max_step_counter.max_steps = 60
            while not done:
                state, reward, done, info = env.step(action)

                step_data = info['step_data']

                assert step_data.action_value == 300, step_data.action_value

                ds = step_data.next_state.device_state
                grid_state = env.env.sim.device_state.grid[:, :, :].mean(
                    axis=(0, 1)) / env.env.sim.device_state.device.vol_per_pix()
                action_data[action_ch].append(ds)
                grid_states[action_ch].append(grid_state)

                delta = step_data.curr_state.device_state - step_data.next_state.device_state

                if False and not done and abs(delta).sum() < 1:
                    done = True

        if display:
            import matplotlib.pyplot as plt
            for action_ch in range(len(action_data)):
                data = np.asarray(action_data[action_ch])
                plt.title(f"{action_ch}")
                plt.plot(data[:, 0], color="r")
                plt.plot(data[:, 1], color="g")
                plt.plot(data[:, 2], color="b")
                plt.ylim(bottom=0, top=265)
                plt.show()
        elif save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(action_data, f)

        return action_data, grid_states

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

        bar_heights = [0 for i in range(self.env.action_handler.pump_count)]

        bar_heights[a_ch] = a_val

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

        ax[0].set_title(
            f"{sd.step_count * ExperimentConfig.CYCLE_DURATION} seconds elapsed (step {sd.step_count}/{sd.max_steps})")

        # table.auto_set_font_size(False)
        table.scale(2, 4)
        # table.set_fontsize(kwargs.get("table_size", 5))

        ax[0].axis('off')
        return plt.gcf(), plt.gca()

    def render(self, mode='human', **kwargs):

        if mode == "plt":
            plt.ion()
            self._render_for_plt(kwargs=kwargs)

            plt.pause(kwargs.get("wait_delay", 0.1))

        elif mode == "rgb_array":
            result = self._render_for_plt(kwargs=kwargs)
            if result is None:
                result = plt.gcf(), plt.gca()
            fig, ax = result

            fig.suptitle(kwargs.get("title", ""))
            fig.canvas.draw()

            width, height = fig.get_size_inches() * fig.get_dpi()
            width = int(width)
            height = int(height)
            mplimage = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
            # plt.close()
            return mplimage

        return self.env.render("channel_mean")


def range_test():
    def display_range_test(action_data):
        for action_ch in range(len(action_data)):
            data = np.asarray(action_data[action_ch])
            plt.title(f"{action_ch}")
            plt.plot(data[:, 0], color="r")
            plt.plot(data[:, 1], color="g")
            plt.plot(data[:, 2], color="b")
            # plt.ylim(bottom=0, top=265)
            plt.show()

    e = ColorMix3D()

    # m = np.asarray([[-0.0984, 0.7386, 0.5415],
    #                 [0.7000, -.1000, .5000],
    #                 [0.6502, 0.3318, -0.1884]])
    # e.set_mixer(ColorMixer(m=m))

    action_data, grid_state = ColorMix3D.perform_range_test(e)

    display_range_test(action_data)


def run():
    env = gym.make('envs:ColorMix3D-v0')

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
    range_test()
