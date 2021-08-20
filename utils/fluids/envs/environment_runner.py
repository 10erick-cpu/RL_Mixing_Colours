import time

import pandas as pd
import torch
from utils.fluids.envs.env_criteria.goals import NoMoreGoalsError

stats_columns = ['episode_id', 'step', 'state', 'action', 'reward', 'next_state', 'done']


class EnvStepInfo(object):
    def __init__(self, **kwargs):
        self.episode = kwargs.get("episode_id", None)
        self.step = kwargs.get("step", None)
        self.max_length = kwargs.get("max_length", None)
        self.env = kwargs.get("env", None)
        self.raw_obs = kwargs.get("raw_obs", None)

        self.state = kwargs.get("state", None)
        self.next_state = kwargs.get("next_state", None)
        self.action = kwargs.get("action", None)
        self.reward = kwargs.get("reward", None)
        self.done = kwargs.get("done", None)

    def set(self, **kwargs):
        self.episode = kwargs.get("episode_id", self.episode)
        self.step = kwargs.get("step", self.step)
        self.max_length = kwargs.get("max_length", self.max_length)
        self.env = kwargs.get("env", self.env)

        self.state = kwargs.get("state", self.state)
        self.next_state = kwargs.get("next_state", self.next_state)
        self.action = kwargs.get("action", self.action)
        self.reward = kwargs.get("reward", self.reward)
        self.done = kwargs.get("done", self.done)


class EnvironmentPlayer(object):

    def __init__(self):
        self.plotter = None

    def set_plotter(self, plotter):
        self.plotter = plotter
        self.plotter.set_player(self)

    def select_action(self, env_step_info: EnvStepInfo) -> int:
        raise NotImplementedError()

    def on_begin_episode(self, episode_id):
        self.plotter.on_episode_begin(episode_id)

    def on_end_episode(self, episode_id):
        self.plotter.on_episode_end(episode_id)

    def visualize_agent(self, plot_axis, env_step_info: EnvStepInfo):
        pass

    def name(self):
        return "env_player_base"

    def on_step_complete(self, env_step_info: EnvStepInfo):

        if self.plotter:
            self.plotter.on_agent_step(env_step_info)


class EnvironmentRunner(object):
    def __init__(self, env, actions):
        self.env = env
        self.actions = actions

    def run(self, num_runs, player: EnvironmentPlayer, render=False, max_length=2000):
        stats = []
        for i in range(num_runs):
            print("\rRunner completion: {0:.2f}%".format(((i + 1) / num_runs) * 100), end="")
            try:
                self.__run(i, self.env, player, max_length, render, stats)

            except NoMoreGoalsError as e:
                print("Runner out of goals")
                break
            except Exception as e:
                print("Exception caught, continue with next run", e)
                print(e)
                raise e

        print()
        return pd.DataFrame(stats, columns=stats_columns)

    def __run(self, episode_id, env, player: EnvironmentPlayer, max_length, render, stats):
        step = 0
        print()
        state = env.reset()
        player.on_begin_episode(episode_id)
        raw_obs=None
        step_duration = None
        done = False

        while True:
            print("\rStep {}/{}".format(step, max_length),
                  "{0:.3f}".format(step_duration) if step_duration is not None else -1, "ms", end="")
            start = time.time()
            esi = EnvStepInfo(state=state, raw_obs=raw_obs, max_length=max_length, step=step, episode_id=episode_id,
                              env=env)

            action = player.select_action(esi)
            #raw_obs, (next_state, reward, done, _) = env.step(action)
            next_state, reward, done, _ = env.step(action)

            esi.set(action=action, next_state=next_state, reward=reward, done=done, raw_obs=raw_obs)
            step_duration = (time.time() - start) * 1000
            player.on_step_complete(esi)

            data = {'episode_id': episode_id,
                    'step': step, 'state': state.cpu().numpy() if isinstance(state, torch.Tensor) else state,
                    'action': self.actions[action],
                    'reward': reward.cpu().item() if isinstance(next_state, torch.Tensor) else reward,
                    'next_state': next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state,
                    'done': done}

            stats.append(data)

            if render:
                env.render('human')
            state = next_state
            step += 1

            if done:
                break

            if step > max_length:
                break

        player.on_end_episode(episode_id)
