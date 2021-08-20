import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc

from rl.misc.model_params import Params
from rl.misc.play_agent import play
from rl.visualization.dataframe_utils import RunProcessor
from utils.helper_functions.misc_utils import flatten_dataframe_index, sort_natural_keys
from utils.models.folder import Folder

SNS_CI = .68


class AgentRun(object):
    _FILE_TRAIN_HISTORY = "episode_tracker.csv"
    _FILE_VAL_FINAL = "val_result_final.csv"
    _FILE_VAL_STEP = "val_result_step_{}.csv"

    def __init__(self, agent, folder: Folder, id: int = None):
        self.root = folder
        self.id = int(self.root.name.split("_")[-1])
        self.agent = agent
        if id is not None:
            assert self.id == id

    def play(self, env_init_helper, action_selector_fn=None, checkpoint_name="final_model.pkl", num_episodes=5,
             max_steps=50, goals=None,
             init=None, episode_tracker=None):

        env = self.agent.cfg.build_env(env_init_helper)

        model = self.load_model(env, checkpoint_name)

        def action_selector(obs):
            # agent prediction returns a tuple of (action_idx, q-values), we only need the idx
            return model.predict(obs)[0]

        play(model, action_selector_fn if action_selector_fn is not None else action_selector, env, num_episodes,
             max_steps, goals, init, episode_tracker=episode_tracker)

    def load_model(self, env, checkpoint_name="final_model.pkl"):
        cfg = self.agent.cfg

        model = cfg.get_model(env)
        if checkpoint_name is None:
            return model
        model = model.load(self.root.get_file_path(checkpoint_name), env=model.env)

        assert model.env
        return model

    def _get_file(self, name, load=False, process=True, data_dict=None):
        file = self.root.get_file_path(name)
        if not os.path.exists(file):
            print(file, "does not exist")
            return None
        if not load:
            return file
        return self._load_log(file, process=process, data_dict=data_dict)

    def _load_log(self, path, process=True, data_dict=None):
        df = pd.read_csv(path)
        df['agent_run_id'] = self.id
        df['agent'] = self.agent.name()
        if data_dict is not None:
            for key, value in data_dict.items():
                if key in df.columns:
                    raise KeyError("Key", key, "already exists in df", df)
                df[key] = value

        if process:
            return RunProcessor.process_df(df)
        return df

    def train_log(self, load=True, process=True):
        return self._get_file(self._FILE_TRAIN_HISTORY, load=load, process=process)

    def val_final_log(self, load=True, process=True):
        return self._get_file(self._FILE_VAL_FINAL, load=load, process=process)

    def val_step(self, step, load=True, process=True, data_dict=None):
        return self._get_file(self._FILE_VAL_STEP.format(step), load=load, process=process, data_dict=data_dict)

    def __episode_to_val_step(self, episode, ep_length=100, adjust=True):
        if adjust:
            return episode * ep_length - 1
        return episode * ep_length

    def __val_step_to_episode(self, val_step, ep_length=100, adjust=True):
        if adjust:
            return (val_step + 1) // ep_length
        return val_step // ep_length

    def val_step_iter(self, adjust_step_id=True, eps=None):
        if eps is None:
            for file in self.root.make_file_provider(extensions="csv", contains="val_result_step_"):
                step = int(file.split("_")[-1].split(".")[0])
                step_ep = self.__val_step_to_episode(step, adjust=adjust_step_id)
                val_step_data = self.val_step(step, data_dict={'val_step': step_ep})
                yield step, val_step_data
        else:
            for ep_id in eps:
                if adjust_step_id:
                    step_id = self.__episode_to_val_step(ep_id)
                else:
                    step_id = ep_id

                yield ep_id, self.val_step(step_id, data_dict={'val_step': ep_id})


class AgentHistory(object):
    def __init__(self, agent_base_dir):
        self.root_dir = agent_base_dir
        self.cfg = Params.load(self.root_dir.get_file_path("config.json"))
        self.cache = dict()
        self.alternative_name = None

    def get_agent(self, run_id):
        for agent_run_id, agent_run in self._run_iter():
            if run_id == agent_run_id:
                return agent_run

    def rename(self, name):
        self.alternative_name = name

    def name(self):
        return self.alternative_name or self.cfg.short_name()

    def _run_iter(self):

        for run_id, run_name in enumerate(sorted(self.root_dir.get_folders(abs_path=False), key=sort_natural_keys)):
            run_folder = self.root_dir.make_sub_folder(run_name, create=False)

            yield (run_id, AgentRun(self, run_folder, run_id))

    def train_logs(self):

        for run_id, run in self._run_iter():
            processed, dataframe = run.train_log()
            yield run_id, run, processed, dataframe

    def val_step_logs(self, adjust_step_id=True):
        for run_id, run in self._run_iter():

            val_steps = []
            for val_step_id, val_step in run.val_step_iter(adjust_step_id=adjust_step_id):
                val_steps.append((val_step_id, val_step))
            yield run, val_steps

    def val_logs(self):
        for run_id, run in self._run_iter():
            yield run_id, run, run.val_final_log()

    def load_training(self):
        if 'training' in self.cache:
            return self.cache['training']

        data = []
        full_data = []
        for run_id, agent_run in self._run_iter():
            proc, raw = agent_run.train_log()

            data.append(proc)
            full_data.append(raw)

        result = pd.concat(data), pd.concat(full_data)
        self.cache['training'] = result
        return result

    def load_final_validation(self):
        if 'val_final' in self.cache:
            return self.cache['val_final']
        data = []
        full_data = []
        for run_id, agent_run in self._run_iter():
            episodes, full_df = agent_run.val_final_log()

            data.append(episodes)
            full_data.append(full_df)

        result = pd.concat(data), pd.concat(full_data)
        self.cache['val_final'] = result
        return result

    def load_eval_steps(self, adjust_step_id=True, val_steps=None):
        if 'eval_steps' in self.cache and self.cache['val_steps'] == val_steps:
            return self.cache['eval_steps']
        data = []
        full_data = []
        for run_id, agent_run in self._run_iter():
            for step, val_step_data in agent_run.val_step_iter(adjust_step_id, val_steps):
                episodes, full_df = val_step_data
                data.append(episodes)
                full_data.append(full_df)

        result = pd.concat(data), pd.concat(full_data)
        self.cache['eval_steps'] = result
        self.cache['val_steps'] = val_steps
        return result

    def smooth_dataframe(self, df, groupby, window_size, target_col, out_col_name):
        processed = []
        for idx, group in df.groupby(groupby):
            data = group.copy()
            if window_size:
                data[out_col_name] = data[target_col].rolling(window_size, min_periods=0, center=True).mean()
            else:
                data[out_col_name] = data[target_col]

            processed.append(data)
        return pd.concat(processed)

    def auc_training(self, color, reduce_runs=True, ax=None, label=None, show=False):

        ag, data = self.load_training()

        if reduce_runs:
            data = data.groupby(['agent', 'episode_id']).agg({'reward': 'mean'}).reset_index()
            x = data['episode_id'].to_numpy()
            y = data['reward'].to_numpy()

            result = auc(x, y)

            data = data.groupby(['agent']).agg({'reward': 'mean'}).reset_index()
            data['auc_train'] = result

            if show is None:
                return data

            sns.barplot(data=data, x="agent", y="auc_train", ax=ax or plt.gca(), color=color, label=label, ci="sd")


        else:
            data = data.groupby(['agent', 'agent_run_id', 'episode_id']).agg({'reward': 'mean'}).reset_index()

            results = []
            for idx, group in data.groupby(['agent_run_id']):
                df = group.copy()
                auc_val = auc(df['episode_id'], df['reward'])

                results.append({'agent': self.name(), 'agent_run_id': df['agent_run_id'].iloc[0], "auc_train": auc_val})
            results = pd.DataFrame(results)
            print(results)
            if show is None:
                return results

            sns.barplot(data=results, x="agent_run_id", y="auc_train", ax=ax or plt.gca(), color=color, label=label)
        sns.despine(ax=ax)
        if show:
            plt.show()

    def mr_training_data(self, smooth=0, reward_reduction="mean", reduce_runs=True):
        data_ag, full_df = self.load_training()
        grouped = full_df.groupby(['agent', 'agent_run_id', 'episode_id']).agg(
            {'reward': reward_reduction}).reset_index()

        grouped = self.smooth_dataframe(grouped, ['agent', 'agent_run_id'], smooth, 'reward', 'reward_smoothed')

        if reduce_runs:
            gb = ['agent', 'episode_id']
        else:
            gb = ['agent', 'agent_run_id', 'episode_id']
        reduced = grouped.groupby(gb).agg(
            {'reward_smoothed': ['mean', 'std']}).reset_index()

        reduced = flatten_dataframe_index(reduced)

        reduced['fill_min'] = reduced['reward_smoothed_mean'] - reduced['reward_smoothed_std'] * 0.5
        reduced['fill_max'] = reduced['reward_smoothed_mean'] + reduced['reward_smoothed_std'] * 0.5
        return reduced

    def mr_training(self, color, ax=None, reduce_runs=True, reward_reduction="mean", smooth=5, label=None, show=False,
                    ci=SNS_CI,
                    fill=True):

        reduced = self.mr_training_data(reduce_runs=reduce_runs, reward_reduction=reward_reduction, smooth=smooth)

        if ax is None:
            ax = plt.gca()

        sns.lineplot(data=reduced, x='episode_id', y='reward_smoothed_mean', ci=None,
                     hue="agent_run_id" if not reduce_runs else None,
                     color=color,
                     ax=ax, label=label)
        if fill and reduce_runs:
            ax.fill_between(reduced['episode_id'], reduced['fill_min'], reduced['fill_max'], alpha=0.4, color=color)

        ax.set(xlabel='Training Episode', ylabel=f'Mean Reward\n(smooth={smooth})')
        if show:
            plt.show()
        return reduced

    def sp_training(self, color, ax=None, reduce_runs=True, spag_reduction="mean", smooth=5, show=False, label=None):

        ax = plt.gca() if ax is None else ax

        data_ag, df = self.load_training()

        test = data_ag.groupby(['agent', 'episode_id', 'agent_run_id']).agg(
            {'success_percentage_after_goal': spag_reduction}).reset_index()

        test = self.smooth_dataframe(test, ['agent', 'agent_run_id'], smooth, 'success_percentage_after_goal',
                                     'spag_smoothed')

        reduced = test.groupby(['agent', 'episode_id']).agg(
            {'spag_smoothed': ['mean', 'std']}).reset_index()

        reduced = flatten_dataframe_index(reduced)

        reduced['fill_min'] = reduced['spag_smoothed_mean'] - reduced['spag_smoothed_std'] * 0.5
        reduced['fill_max'] = reduced['spag_smoothed_mean'] + reduced['spag_smoothed_std'] * 0.5

        sns.lineplot(data=test, x="episode_id", y="spag_smoothed",
                     hue="agent_run_id" if not reduce_runs else None, ci=None, ax=ax, label=label, color=color)

        ax.fill_between(reduced['episode_id'], reduced['fill_min'], reduced['fill_max'], alpha=0.4, color=color)

        ax.set(xlabel='Training Episode', ylabel=f'SPAG\n(smooth={smooth})')

        if show:
            plt.show()
        return test

    def mrag_eval(self, show=False, barplot=False, reduce_runs=True, ax=None):
        ax = plt.gca() if ax is None else ax

        data_ag, df = self.load_eval_steps(val_steps=None)

        data_ag = data_ag.groupby(['agent', 'val_step', 'agent_run_id']).agg(
            {'mean_reward_after_goal': 'mean'}).reset_index()

        if barplot:
            sns.barplot(data=data_ag, x='val_step', y='mean_reward_after_goal', ci=SNS_CI,
                        hue="agent_run_id" if not reduce_runs else None,
                        palette=sns.color_palette("hls", len(data_ag['agent_run_id'].unique())),
                        ax=ax)
        else:
            sns.lineplot(data=data_ag, x='val_step', y='mean_reward_after_goal', ci=SNS_CI,
                         hue="agent_run_id" if not reduce_runs else None,
                         palette=sns.color_palette("hls", len(data_ag['agent_run_id'].unique())),
                         ax=ax)
        ax.axhline(y=-1, c="k")
        if show:
            plt.show()

        sns.despine()
        return data_ag

    def spag_eval(self, color, ax=None, show=False, reduce_runs=True, label=None, smooth=0, spag_reduction="mean"):
        ax = plt.gca() if ax is None else ax

        data_ag, df = self.load_eval_steps(val_steps=None)

        test = data_ag.groupby(['agent', 'val_step', 'agent_run_id']).agg(
            {'success_percentage_after_goal': spag_reduction}).reset_index()

        test = self.smooth_dataframe(test, ['agent', 'agent_run_id'], smooth, 'success_percentage_after_goal',
                                     'spag_smoothed')

        reduced = test.groupby(['agent', 'val_step']).agg(
            {'spag_smoothed': ['mean', 'std']}).reset_index()

        reduced = flatten_dataframe_index(reduced)

        reduced['fill_min'] = reduced['spag_smoothed_mean'] - reduced['spag_smoothed_std'] * 0.5
        reduced['fill_max'] = reduced['spag_smoothed_mean'] + reduced['spag_smoothed_std'] * 0.5

        sns.lineplot(data=reduced, x="val_step", y="spag_smoothed_mean",
                     hue="agent_run_id" if not reduce_runs else None, ci=None, ax=ax, label=label, color=color)

        ax.fill_between(reduced['val_step'], reduced['fill_min'], reduced['fill_max'], alpha=0.4, color=color)

        ax.set(xlabel='Training Episode', ylabel=f'SPAG\n(smooth={smooth})')
        ax.set_xticks(reduced['val_step'].unique())

        if show:
            plt.show()
        return test

    def get_best_agent(self):
        best_id = self.get_best_agent_id()
        best_id = best_id[0]['agent_run_id']
        return best_id, self.get_agent(best_id)

    def get_best_agent_id(self):
        data_ag, df = self.load_eval_steps(val_steps=[100])
        test = data_ag.groupby(['agent', 'val_step', 'agent_run_id']).agg(
            {'mean_reward_after_goal': 'mean',
             'success_percentage_after_goal': 'mean'}).reset_index()

        r = []
        for idx, group in test.groupby(['agent', 'agent_run_id', 'val_step']):
            data = group.copy()
            data['mean_reward'] = data['mean_reward_after_goal'].mean()
            data['spag'] = data['success_percentage_after_goal'].mean()
            r.append(data)
        r = pd.concat(r)
        print(r[r['agent_run_id'] == 2]['mean_reward_after_goal'].mean())

        test = test.groupby(['agent', 'agent_run_id']) \
            .agg({'mean_reward_after_goal': 'mean',
                  'success_percentage_after_goal': 'mean'}).reset_index()

        print(test)
        return test.iloc[test['mean_reward_after_goal'].idxmax()], test.iloc[
            test['success_percentage_after_goal'].idxmax()]

    def mrag_training_evaluation(self, color, ax=None, show=False, reduce_runs=True, label=None, smooth=0,
                                 reward_reduction="mean", only_output=False):
        ax = plt.gca() if ax is None and not only_output else ax

        data_ag, df = self.load_eval_steps(val_steps=None)

        test = data_ag.groupby(['agent', 'val_step', 'agent_run_id']).agg(
            {'mean_reward_after_goal': reward_reduction}).reset_index()

        test = self.smooth_dataframe(test, ['agent', 'agent_run_id'], smooth, 'mean_reward_after_goal',
                                     'mean_reward_after_goal_smoothed')

        gb = ['agent', 'val_step'] if reduce_runs else ['agent', 'val_step', 'agent_run_id']

        reduced = test.groupby(gb).agg(
            {'mean_reward_after_goal_smoothed': ['mean', 'std']}).reset_index()

        reduced = flatten_dataframe_index(reduced)

        reduced['fill_min'] = reduced['mean_reward_after_goal_smoothed_mean'] - reduced[
            'mean_reward_after_goal_smoothed_std'] * 0.5
        reduced['fill_max'] = reduced['mean_reward_after_goal_smoothed_mean'] + reduced[
            'mean_reward_after_goal_smoothed_std'] * 0.5
        print(reduced)
        # sns.lineplot(data=reduced, x="val_step", y="mean_reward_after_goal_smoothed_mean",
        #              hue="agent_run_id" if not reduce_runs else None, ci=None, ax=ax, label=label, color=color)
        if only_output:
            return reduced

        reduced.plot(kind="bar", x="val_step", y="mean_reward_after_goal_smoothed_mean", ax=ax,
                     yerr="mean_reward_after_goal_smoothed_std")

        # ax.fill_between(reduced['val_step'], reduced['fill_min'], reduced['fill_max'], alpha=0.4, color=color)

        ax.set(xlabel='training step', ylabel=f'Mean reward after goal\n(smooth={smooth})')
        ax.set_xticks(reduced['val_step'].unique())

        if show:
            plt.show()
        return reduced

    def mr_training_evaluation(self, color, ax=None, show=False, reduce_runs=True, label=None, smooth=0, reward_reduction="mean",
                               only_output=False):
        ax = plt.gca() if ax is None and not only_output else ax

        data_ag, df = self.load_eval_steps(val_steps=None)

        test = df.groupby(['agent', 'val_step', 'agent_run_id']).agg(
            {'reward': reward_reduction}).reset_index()

        test = self.smooth_dataframe(test, ['agent', 'agent_run_id'], smooth, 'reward',
                                     'reward_smoothed')

        gb = ['agent', 'val_step'] if reduce_runs else ['agent', 'val_step', 'agent_run_id']

        reduced = test.groupby(gb).agg(
            {'reward_smoothed': ['mean', 'std']}).reset_index()

        reduced = flatten_dataframe_index(reduced)

        reduced['fill_min'] = reduced['reward_smoothed_mean'] - reduced['reward_smoothed_std'] * 0.5
        reduced['fill_max'] = reduced['reward_smoothed_mean'] + reduced['reward_smoothed_std'] * 0.5
        print(reduced)
        # sns.lineplot(data=reduced, x="val_step", y="mean_reward_after_goal_smoothed_mean",
        #              hue="agent_run_id" if not reduce_runs else None, ci=None, ax=ax, label=label, color=color)
        if only_output:
            return reduced

        reduced.plot(kind="bar", x="val_step", y="reward_smoothed_mean", ax=ax, yerr="reward_smoothed_std")

        # ax.fill_between(reduced['val_step'], reduced['fill_min'], reduced['fill_max'], alpha=0.4, color=color)

        ax.set(xlabel='training step', ylabel=f'Mean reward after goal\n(smooth={smooth})')
        # ax.set_xticks(reduced['val_step'].unique())

        if show:
            plt.show()
        return reduced

    def mrag_validation(self, ax=None, reduce_runs=True, label=None, show=False,
                        barplot=False):

        ax = plt.gca() if ax is None else ax

        data_ag, df = self.load_final_validation()

        if barplot:
            sns.barplot(data=data_ag, x='goal', y='mean_reward_after_goal', ci=SNS_CI,
                        hue="agent_run_id" if not reduce_runs else None,
                        palette=sns.color_palette("hls", len(data_ag['agent_run_id'].unique())),
                        ax=ax, label=label)
        else:
            sns.lineplot(data=data_ag, x='goal', y='mean_reward_after_goal', ci=SNS_CI,
                         hue="agent_run_id" if not reduce_runs else None,
                         palette=sns.color_palette("hls", len(data_ag['agent_run_id'].unique())),
                         ax=ax, label=label)

        if show:
            plt.show()
        return data_ag

    def plot_val_episode(self, val_step, episode_id, agent_run_id=None, eval_iter_id="reduce", ax=None, show=True,
                         label=None, palette=None, legend=False):
        # TODO: add final validation logic
        ag, episodes = self.load_eval_steps(val_steps=[val_step] if val_step is not None else None)

        # label = label if l or self.name()
        ax = plt.gca() if ax is None else ax

        if agent_run_id is None:
            runs = episodes['agent_run_id'].unique()
            target_ep = episodes[
                (episodes.episode_id == episode_id)]

            for run in runs:
                self.plot_val_episode(val_step, episode_id, run, "reduce", ax=ax, show=False, label=f"run{run}" if legend else None,
                                      palette=palette[f"run{run}"])

            if show:
                plt.show()

            return

        if eval_iter_id is not None and isinstance(eval_iter_id, int):
            target_ep = episodes[
                (episodes.episode_id == episode_id)
                & (episodes.agent_run_id == agent_run_id)
                & (episodes.eval_iter_id == eval_iter_id)]
        else:
            target_ep = episodes[
                (episodes.episode_id == episode_id)
                & (episodes.agent_run_id == agent_run_id)]

        print(target_ep['reward'].mean())
        sns.lineplot(data=target_ep, x="step", y="s_r", hue="eval_iter_id" if eval_iter_id is None else None,
                     label=label, ci=SNS_CI, ax=ax, color=palette)
        ax.axhline(y=target_ep['goal'].iloc[0] + 1.5, color="black", alpha=0.3)
        ax.axhline(y=target_ep['goal'].iloc[0] - 1.5, color="black", alpha=0.3)
        ax.axhline(y=target_ep['goal'].iloc[0], linestyle="dashed", color="black", alpha=0.6)
        if show:
            plt.show()
        return target_ep

    def plot_training_episode(self, episode_id, agent_run_id=None, eval_iter_id="reduce", ax=None, show=True,
                              label=None, color=None):
        processed, raw = self.load_training()
        episodes = raw
        label = label or self.name()
        ax = plt.gca() if ax is None else ax

        if agent_run_id is None:
            runs = episodes['agent_run_id'].unique()

            for run in runs:
                self.plot_training_episode(episode_id, run, "reduce", ax=ax, show=False, label=f"run{run}",
                                           color=color)

            if show:
                plt.show()

            return

        if eval_iter_id is not None and isinstance(eval_iter_id, int):
            target_ep = episodes[
                (episodes.episode_id == episode_id)
                & (episodes.agent_run_id == agent_run_id)
                & (episodes.eval_iter_id == eval_iter_id)]
        else:
            target_ep = episodes[
                (episodes.episode_id == episode_id)
                & (episodes.agent_run_id == agent_run_id)]

        print("mean episode reward",target_ep['reward'].mean())
        sns.lineplot(data=target_ep, x="step", y="s_r", hue="eval_iter_id" if eval_iter_id is None else None,
                     label=label, ci=SNS_CI, ax=ax, color=color)
        ax.axhline(y=target_ep['goal'].iloc[0] + 1, color="black", alpha=0.5)
        ax.axhline(y=target_ep['goal'].iloc[0] - 1, color="black", alpha=0.5)
        ax.axhline(y=target_ep['goal'].iloc[0], linestyle="dashed", color="black")

        ax.set(xlabel='Timestep', ylabel=f'Value Red Channel (s_r)')
        if show:
            plt.show()
        return target_ep

    def plot_goal_distribution(self, ax=None, show=False):
        data, data_ag = self.load_training()
        ax = plt.gca() if ax is None else ax
        pd.options.display.multi_sparse = False
        data = data.groupby(['agent_run_id', 'episode_id']).agg({'goal': 'first'}).reset_index()
        data = data.drop('episode_id', 1)

        data = data.groupby(['agent_run_id']).goal.value_counts().reset_index(name="count")

        sns.lineplot(data=data, x="goal", y='count', ci='sd', hue="agent_run_id", ax=ax)

        ax.set(xlabel='Goal State Value', ylabel=f'Number of Occurrences during Training')
        if show:
            plt.show()

    def __repr__(self):
        return self.name()
