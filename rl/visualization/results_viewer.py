import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rl.visualization.agent_history import AgentHistory
from utils.helper_functions.colors import ColorGenerator
from utils.models.folder import Folder

SNS_CI = "sd"


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class ResultsViewer(object):
    def __init__(self, results_folder):

        if isinstance(results_folder, str):
            results_folder = Folder(results_folder)
        self.root_folder = results_folder

    # def list_environments(self):
    #     return self.root_folder.get_folders(abs_path=False)

    def list_experiments(self):
        return self.root_folder.get_folders(abs_path=False)

    def view_experiment(self, exp_name):
        exp_folder = self.root_folder.make_sub_folder(exp_name, create=False)
        if not exp_folder.exists():
            raise KeyError("experiment", exp_folder, "not found")

        return ExperimentViewer(exp_folder)

    def view_experiment_folder(self, folder_path):
        exp_folder = Folder(folder_path)

        return ExperimentViewer(exp_folder)


class ExperimentViewer(object):

    @staticmethod
    def for_output_only(agents, output_dir):
        ev = ExperimentViewer(output_dir, agents)

        ev.show_overview()

    def __init__(self, exp_base_folder, agents=None):
        self.root = exp_base_folder

        if agents is None:
            self.agents = {}
            for f in sorted(self.root.get_folders(abs_path=False), key=natural_keys):
                print(f)
                agent = AgentHistory(self.root.make_sub_folder(f))
                if agent.name() in self.agents:
                    print("Warning: two agents have the same id, attempt to override by date")
                    print(agent.name(), "@", f)

                self.agents[agent.name()] = agent
        else:
            self.agents = agents

    def _val_runs_data(self, val_steps=None, after_goal=True, agents=None):
        data = []

        if agents is None:
            agents = self.agents

        for agent_name, agent in agents.items():
            ag, df = agent.load_eval_steps(val_steps=val_steps)

            data.append(ag if after_goal else df)

        data_ag = pd.concat(data)
        return data_ag

    def _training_data(self, processed=True, agents=None):
        data = []
        if agents is None:
            agents = self.agents

        for agent_name, agent in agents.items():
            proc, raw = agent.load_training()

            data.append(proc if processed else raw)

        result = pd.concat(data)
        return result

    def list_agents(self):
        return list(self.agents.keys())

    def get_agent(self, key) -> AgentHistory:
        return self.agents[key]

    def get_color_order(self, vals):
        vals = sorted(vals, key=natural_keys)
        return vals, sns.color_palette("hls", len(vals))

    def test(self):
        f, ax = plt.subplots(2, 2, figsize=(14, 8))

        # training
        self.sp_training_2(ax[0][0], smooth=5)
        ax[0][0].legend_.remove()
        self.mr_training_2(ax[1][0], smooth=5)
        ax[0][0].set_title("Training")
        ax[1][0].legend_.remove()
        # validation

        self.sp_training_2(ax[0][1], smooth=20)

        self.mr_training_2(ax[1][1], smooth=20)

        # self.mrag_eval(ax=ax[0][1], barplot=True)
        ax[0][1].legend_.remove()
        # self.spag_eval(ax=ax[1][1], barplot=True)

        ax[0][1].set_title("Validation")
        plt.legend(bbox_to_anchor=(1.0, 1), loc='center right', borderaxespad=0.)
        plt.tight_layout()

        plt.show()

    def __plot_and_save(self, plot_fn, target_path):
        f, ax = plt.subplots()
        plot_fn(ax)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
        plt.tight_layout()
        sns.despine(f)
        f.savefig(target_path)
        plt.close(f)

    def save_overview(self, out_dir=None, agents=None, smooth=10):
        if out_dir is None:
            out_dir = self.root

        if agents is None:
            agents = self.agents

        palette = ColorGenerator(agents.keys())

        slf = self

        def mr(ax):
            slf.mr_training(palette, ax=ax, smooth=smooth, agents=agents)

        def spag_train(ax):
            slf.spag_training(palette, ax=ax, smooth=smooth, agents=agents)

        def auc_train(ax):
            slf.auc_training(palette, ax=ax, agents=agents)

        def spag_eval(ax):
            slf.spag_eval(palette, ax=ax, barplot=False, agents=agents, smooth=0)

        self.__plot_and_save(mr, out_dir.get_file_path("mr_training.png"))
        self.__plot_and_save(spag_train, out_dir.get_file_path("spag_training.png"))
        self.__plot_and_save(auc_train, out_dir.get_file_path("auc_train.png"))
        self.__plot_and_save(spag_eval, out_dir.get_file_path("spag_eval.png"))
        self.show_overview(save=out_dir.get_file_path("overview.png"), smooth=smooth)

    def show_overview(self, save=False, filter_agents=None, smooth=2):
        f, ax = plt.subplots(2, 2, figsize=(14, 8))
        agents = dict()

        if filter_agents is not None:
            for agent in self.agents.keys():
                if agent in filter_agents:
                    agents[agent] = self.agents[agent]
        else:
            agents = self.agents

        palette = ColorGenerator(agents.keys())

        # training

        self.mr_training(palette, ax=ax[0][0], smooth=smooth, agents=agents)
        ax[0][0].legend_.remove()
        self.spag_training(palette, ax=ax[1][0], smooth=smooth, agents=agents)

        ax[1][0].legend_.remove()
        # validation
        self.auc_training(palette=palette, ax=ax[0][1], agents=agents)
        # ax[0][1].legend_.remove()
        self.spag_eval(palette=palette, ax=ax[1][1], barplot=False, agents=agents, smooth=0)

        plt.legend(bbox_to_anchor=(0.75, 1.3), loc='center', borderaxespad=0.)
        plt.tight_layout()

        if save:
            if isinstance(save, str):
                f.savefig(save)
            else:
                f.savefig(self.root.get_file_path("exp_overview.png"))
        else:
            plt.show()

    def mr_training_2(self, palette, ax=None, show=False, barplot=False, reward_reduction='mean', smooth=0):
        data_ag = self._training_data(processed=False)

        data_ag = data_ag.groupby(['episode_id', 'agent', 'agent_run_id']).agg(
            {'reward': reward_reduction}).reset_index()

        # data_ag['reward'] = data_ag['reward'].rolling(smooth, min_periods=smooth).mean()
        if smooth:
            data_ag['reward'] = data_ag.groupby(['agent'])['reward'].apply(
                lambda x: x.rolling(center=False, window=smooth, min_periods=smooth).mean())

        # data_ag = data_ag.groupby(['agent', 'agent_run_id']).agg({'reward': 'mean'}).reset_index()

        if barplot:
            sns.barplot(data=data_ag, x='episode_id', y='reward', ci=SNS_CI,
                        hue="agent",
                        hue_order=palette.keys(),
                        palette=palette.as_dict(),
                        ax=ax)
        else:
            sns.lineplot(data=data_ag, x='episode_id', y='reward', ci=SNS_CI,
                         hue="agent",
                         hue_order=palette.keys(),
                         palette=palette.as_dict(),
                         ax=ax)
        ax.set_ylabel("mean reward")
        sns.despine()
        if show:
            plt.show()

        return data_ag

    def mrag_eval_new(self, palette, show=False, ax=None, agents=None, smooth=0):
        ax = plt.gca() if ax is None else ax
        if agents is None:
            agents = self.agents
        outs = []
        colors = []
        for idx, (agent_name, agent) in enumerate(agents.items()):
            out = agent.mrag_training_evaluation(color=palette[agent_name], ax=None, only_output=True, reduce_runs=True, smooth=smooth,
                                                 label=agent.name())
            colors.append(palette[agent_name])
            outs.append(out)

        outs = pd.concat(outs)
        print(outs)
        pivot = outs.pivot(index="val_step", values=["mean_reward_after_goal_smoothed_mean", "mean_reward_after_goal_smoothed_std"],
                           columns="agent")
        print(pivot)
        pivot.plot(kind="bar", y="mean_reward_after_goal_smoothed_mean", yerr="mean_reward_after_goal_smoothed_std", ax=ax, color=colors)
        # outs.groupby(['agent']).plot(kind='bar', x="val_step", y="mean_reward_after_goal_smoothed_mean", ax=ax, yerr="mean_reward_after_goal_smoothed_std")
        ax.set_xlabel("training episode")
        ax.set_ylabel("mean reward after goal")
        if show:
            plt.show()

    def mrag_eval(self, palette, show=False, barplot=False, ax=None, agents=None):
        ax = plt.gca() if ax is None else ax

        data_ag = self._val_runs_data(val_steps=None, after_goal=True, agents=agents)

        data_ag = data_ag.groupby(['agent', 'val_step', 'agent_run_id']).agg({'mean_reward_after_goal': 'mean'}).reset_index()

        if barplot:
            sns.barplot(data=data_ag, x='val_step', y='mean_reward_after_goal', ci=SNS_CI,
                        hue="agent",
                        hue_order=palette.keys(),
                        palette=palette.as_dict(),
                        ax=ax)
        else:
            sns.lineplot(data=data_ag, x='val_step', y='mean_reward_after_goal', ci=SNS_CI,
                         hue="agent",
                         hue_order=palette.keys(),
                         palette=palette.as_dict(),
                         ax=ax)

        ax.axhline(y=-1, c="k", linestyle="dashed")
        ax.set_xlabel("training episode")
        ax.set_ylabel("mean reward after goal")
        sns.despine()
        if show:
            plt.show()

        return data_ag

    def auc_training(self, palette, ax=None, show=False, agents=None, reduce_runs=False):
        ax = plt.gca() if ax is None else ax

        if agents is None:
            agents = self.agents

        data = []
        for agent_name, agent in agents.items():
            d = agent.auc_training(ax=ax, label=agent, reduce_runs=reduce_runs, color=palette[agent_name], show=None)
            data.append(d)

        sns.barplot(data=pd.concat(data), x="agent", y="auc_train", ax=ax, palette=palette.get_palette())
        if not show:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        ax.set_title("Area Under the Curve (AUC)\n(Mean Training Reward)")
        ax.set_ylabel("AUC")

        sns.despine()
        if show:
            plt.show()

    def spag_eval(self, palette, ax=None, show=False, barplot=True, agents=None, smooth=0, reduce_runs=True):
        ax = plt.gca() if ax is None else ax

        if agents is None:
            agents = self.agents

        if not barplot:

            for agent_name, agent in agents.items():
                agent.spag_eval(ax=ax, smooth=smooth, label=agent, reduce_runs=reduce_runs, color=palette[agent_name])

        else:

            data_ag = self._val_runs_data(val_steps=None, after_goal=True, agents=agents)

            data_ag = data_ag.groupby(['agent', 'val_step', 'agent_run_id']).agg(
                {'success_percentage_after_goal': 'mean'}).reset_index()

            if barplot:
                sns.barplot(data=data_ag, x='val_step', y='success_percentage_after_goal', ci=SNS_CI,
                            hue="agent",
                            hue_order=palette.keys(),
                            palette=palette.as_dict(),
                            ax=ax)
            else:
                sns.lineplot(data=data_ag, x='val_step', y='success_percentage_after_goal', ci=SNS_CI,
                             hue="agent",
                             hue_order=palette.keys(),
                             palette=palette.as_dict(),
                             ax=ax)
            print(data_ag.groupby(['agent', 'val_step'])['success_percentage_after_goal'].describe()[['mean', 'std']])

        # ax.axhline(y=-1, c="k")

        ax.set(xlabel='Training Episode', ylabel=f'SPAG\n(smooth={smooth})')

        ax.set_title("SPAG Validation")

        sns.despine()
        if show:
            plt.show()

    def sp_training_2(self, palette, ax=None, show=False, barplot=False, smooth=0):
        ax = plt.gca() if ax is None else ax

        data = self._training_data()

        data = data.groupby(['agent', 'agent_run_id', 'episode_id'])

        data = data.agg({'success_percentage_after_goal': 'mean'}).reset_index()

        # print(data[data['agent'] == 'bsize=8'].head())

        if smooth:
            data['success_percentage_after_goal'] = data.groupby(['agent'])['success_percentage_after_goal'].apply(
                lambda x: x.rolling(center=False, window=smooth, min_periods=smooth).mean())
            pass

        data = data.reset_index()

        if barplot:
            sns.barplot(data=data, x='episode_id', y='success_percentage_after_goal', ci=SNS_CI,
                        hue="agent",
                        hue_order=palette.keys(),
                        palette=palette.as_dict(),
                        ax=ax)
        else:
            sns.lineplot(data=data, x='episode_id', y='success_percentage_after_goal', ci=SNS_CI,
                         hue="agent",
                         hue_order=palette.keys(),
                         palette=palette.as_dict(),
                         ax=ax)

        sns.despine()
        if show:
            plt.show()

        return data

    def mr_training(self, palette, ax=None, agents=None, smooth=0, show=False, reduce_runs=True):
        ax = plt.gca() if ax is None else ax

        if agents is None:
            agents = self.agents

        for idx, (agent_name, agent) in enumerate(agents.items()):
            agent.mr_training(color=palette[agent_name], ax=ax, reduce_runs=reduce_runs, smooth=smooth, label=agent.name())

        ax.set_title("Mean Training Reward")
        # ax.set(xlabel="Training Episode", ylabel="Mean Reward")
        if show:
            plt.show()

    def mrag_validation(self, ax=None, agents=None, show=False, train_val_step=None, barplot=False):
        ax = plt.gca() if ax is None else ax

        if agents is None:
            agents = self.agents

        for agent_name, agent in agents.items():
            agent.mrag_validation(ax=ax, reduce_runs=True, label=agent.name(), barplot=barplot)

        title = f"Validation results at episode {train_val_step}" if train_val_step is not None else "Validation results after training"
        ax.set_title(title)
        if show:
            plt.show()

    def spag_training(self, palette, ax=None, agents=None, smooth=0, show=False, reduce_runs=True):
        ax = plt.gca() if ax is None else ax

        if agents is None:
            agents = self.agents

        for agent_name, agent in agents.items():
            agent.sp_training(ax=ax, smooth=smooth, label=agent, reduce_runs=reduce_runs, color=palette[agent_name])
        ax.set_title("SPAG Training")
        if show:
            plt.show()

    def spag_validation(self, ax=None, agents=None, show=False, train_val_step=None, barplot=True):
        ax = plt.gca() if ax is None else ax

        if agents is None:
            agents = self.agents

        if not barplot:

            raise NotImplementedError()

        else:

            data = []
            for agent_name, agent in agents.items():
                ag, full_df = agent.load_final_validation()
                data.append(ag)

            data = pd.concat(data)

            test = data.groupby(['agent', 'goal', 'agent_run_id', 'eval_iter_id']).agg(
                {'success_percentage_after_goal': 'mean'}).reset_index()

            order = sorted(data.agent.unique(), key=natural_keys)
            if barplot:

                # data.plot(kind="bar", x="goal", y="success_percentage_after_goal")

                sns.barplot(data=test, ax=ax, x="goal", y="success_percentage_after_goal", hue="agent", hue_order=order,
                            ci=SNS_CI)

            else:

                sns.lineplot(data=data, ax=ax, x="goal", y="success_percentage_after_goal", hue="agent", hue_order=order,
                             ci=SNS_CI)
        sns.despine()
        # plt.legend()

        title = f"Validation results at episode {train_val_step}" if train_val_step is not None else "Validation results after training"
        ax.set_title(title)
        if show:
            plt.show()


if __name__ == '__main__':
    # rv = ResultsViewer("/Users/Dennis/Desktop/thesis/coding/mfd-rl/sim_analysis/experiments/colormix_3d_simple/param_experiments/results")
    pass
