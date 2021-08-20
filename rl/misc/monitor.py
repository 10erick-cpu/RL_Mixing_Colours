import csv

import numpy as np
from stable_baselines.bench.monitor import get_monitor_files, Monitor
from stable_baselines.her import HERGoalEnvWrapper
from stable_baselines.results_plotter import load_results, ts2xy

from rl.misc.evaluation_utils import eval_agent
from utils.models.folder import Folder
from rl.misc.plotting import load_result_file


class EpisodeTracker(object):
    def __init__(self, target_file):
        self.target_file = target_file
        self.file_handler = open(target_file, "wt")
        self.cache = []
        self.step_id = 1
        self.episode_id = 0
        self.logger = csv.DictWriter(self.file_handler,
                                     fieldnames=('episode_id', 'step', 's_r', 's_g', 's_b', 'action', 'reward', 'goal', 'done'))
        self.logger.writeheader()
        self.file_handler.flush()

    def reset(self):
        self.step_id = 1
        self.episode_id += 1
        self.flush()

    def step(self, action, data):
        n_s, reward, done, info = data
        if isinstance(n_s, dict):
            obs = n_s['observation']
            if obs.shape[0] == 3:
                s_r, s_g, s_b = obs
            else:
                (s_r, s_g, s_b) = obs[:3]
            goal = n_s['desired_goal'].squeeze()
        else:
            (s_r, s_g, s_b), goal = n_s[:3], n_s[-1]
        cache_data = self.episode_id, self.step_id, s_r, s_g, s_b, action, reward, goal, done

        self.add_to_cache(cache_data)
        self.step_id += 1

    def add_to_cache(self, elem):
        self.cache.append(elem)

    def flush(self):
        for episode_id, step_id, s_r, s_g, s_b, action, reward, goal, done in self.cache:
            step_info = {'episode_id': episode_id, 'step': step_id,
                         's_r': s_r, 's_g': s_g, 's_b': s_b,
                         'action': action, 'reward': reward,
                         'goal': goal, 'done': done}
            self.logger.writerow(step_info)
        self.file_handler.flush()
        self.cache.clear()


class CustomMonitor(Monitor):

    def __init__(self, env, log_dir: Folder, log_f_name, data, print_interval=50, eval_interval=50, reset_keywords=(),
                 info_keywords=()):
        super().__init__(env, log_dir.get_file_path(log_f_name) if log_f_name else None, True, reset_keywords, info_keywords)
        self.log_dir = log_dir
        self.current_log_fname = log_dir.get_file_path(log_f_name)
        self.log_f_name = log_f_name
        self.print_interval = print_interval
        self.eval_interval = eval_interval
        self.data = data
        self.best_mean_reward, self.n_steps = -np.inf, 0
        self.episode_tracker = EpisodeTracker(log_dir.get_file_path("episode_tracker.csv"))
        self.step_size = 1 if self.data.cfg.model_type == 'dqn' else self.data.cfg.timesteps_per_actorbatch / 100

    def load_logs(self, to_xy=True):
        files = get_monitor_files(self.log_dir.path())
        print(files)
        result = []

        if to_xy:
            x, y = ts2xy(load_results(self.log_dir.path()), 'timesteps')
            result.append((x, y))
        else:
            result.append(load_results(self.log_dir.path()))
        return result

    def print_stats(self):
        # Evaluate policy training performance
        x, y = ts2xy(load_result_file(self.current_log_fname), 'timesteps')

        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print("episode", len(x), "Best total reward: {:.2f} - Last total reward: {:.2f}".format(self.best_mean_reward, mean_reward))
            return mean_reward
        return None

    def eval_agent(self, agent, step, output_dir=None, fname=None, num_runs=1):

        env = self.data.cfg.build_env(self.data.env_init_cls)

        if 'replay' in self.data.cfg and self.data.cfg.replay.type == "her":
            env = HERGoalEnvWrapper(env)

        data = eval_agent(env, agent.predict, num_runs=num_runs)

        if output_dir is not None:
            fname = output_dir.get_file_path(f"val_result_step_{step}.csv" if fname is None else fname)

            data.to_csv(fname)

        return data

    def reset(self, **kwargs):
        result = super(CustomMonitor, self).reset(**kwargs)

        self.episode_tracker.reset()
        return result

    def step(self, action):
        data = super(CustomMonitor, self).step(action)
        self.episode_tracker.step(action, data)
        self.n_steps += self.step_size
        return data

    def __call__(self, _locals, _globals, **kwargs):
        """
                    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
                    :param _locals: (dict)
                    :param _globals: (dict)
                    """

        # Print stats every 1000 calls
        if (self.n_steps + 1) % self.print_interval == 0:
            mean_reward = self.print_stats()

            # New best model, you could save the agent here
            if mean_reward and mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                _locals['self'].save(self.log_dir.get_file_path(f'{self.log_f_name}_best_model.pkl'))

        if self.eval_interval > 0 and (self.n_steps + 1) % self.eval_interval == 0:
            mean_reward = self.eval_agent(_locals['self'], self.n_steps, self.log_dir, num_runs=2)

        return True
