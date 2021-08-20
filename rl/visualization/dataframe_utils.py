import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.models.dot_dict import DotDict

pd.set_option('display.expand_frame_repr', False)


# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

class RunProcessor(object):

    @staticmethod
    def episode_iter(df):
        if 'eval_iter_id' in df.columns:
            gb = df.groupby(['episode_id', 'eval_iter_id'])
        else:
            gb = df.groupby(['episode_id'])

        for group in gb:
            group = group[1].reset_index()
            yield group

    @staticmethod
    def process_df(df):
        episode_infos = []

        for episode in RunProcessor.episode_iter(df):
            ep_s = DotDict()

            if 'eval_iter_id' in episode:
                ep_s.eval_iter_id = episode['eval_iter_id'].iloc[0]

            if 'agent' in episode:
                ep_s['agent'] = episode['agent'].iloc[0]

            if 'agent_run_id' in episode:
                ep_s['agent_run_id'] = episode['agent_run_id'].iloc[0]

            if 'val_step' in episode:
                ep_s['val_step'] = episode['val_step'].iloc[0]

            target_s_r = 'n_s_r' if 'n_s_r' in episode else 's_r'

            episode['abs_dist'] = (episode['goal'] - episode[target_s_r]).abs()
            episode['ga'] = episode['abs_dist'] <= 1.5
            # episode['ga_class'] = pd.cut(episode['abs_dist'].abs(), bins=[0, 1, 2, 3, 4, np.inf],
            #                              labels=["0<x<1", "1<x<2", "2<x<3", "3<x<4", "4+"], include_lowest=True, right=True)
            # episode['step'] = episode['step']-1
            # episode['counts'] = episode['ga_class'].value_counts(sort=False, normalize=True)

            ep_s.mean_dist_to_goal = episode['abs_dist'].mean()

            ep_s.total_steps = episode['ga'].count()
            # ep_s.success_bin_a = (episode['ga_class'] == "0<x<1").sum() / ep_s.total_steps
            # ep_s.success_bin_b = (episode['ga_class'] == "1<x<2").sum() / ep_s.total_steps
            # ep_s.success_bin_c = (episode['ga_class'] == "3<x<4").sum() / ep_s.total_steps
            # ep_s.success_bin_d = (episode['ga_class'] == "4+").sum() / ep_s.total_steps
            ep_s.success_steps = episode['ga'].sum()
            ep_s.success_percentage = ep_s.success_steps / ep_s.total_steps

            if ep_s.success_steps > 0:
                ep_s.first_goal_reached = episode['ga'].idxmax()
                ep_s.success_percentage_after_goal = ep_s.success_steps / (ep_s.total_steps - ep_s.first_goal_reached)
                steps_after_goal = episode.iloc[ep_s.first_goal_reached:].reset_index()
                ep_s.mean_reward_after_goal, ep_s.std_after_goal = steps_after_goal['reward'].mean(), steps_after_goal['reward'].std()
            else:
                ep_s.first_goal_reached = -1
                ep_s.success_percentage_after_goal = 0

                val_non_successful = -100

                ep_s.mean_reward_after_goal, ep_s.std_after_goal = val_non_successful, val_non_successful

            ep_s.init_state = episode['s_r'].iloc[0]
            ep_s.end_state = episode['s_r'].iloc[-1]
            ep_s.episode_id = episode['episode_id'].iloc[0]
            ep_s.goal = episode['goal'][0]

            episode_infos.append(ep_s)
        return pd.DataFrame(episode_infos).sort_values('episode_id'), df

    def process_path(self, path):
        df = pd.read_csv(path)
        return self.process_df(df)


def process_raw_transitions(df):
    def unwrap(g):
        return g.item()

    # df['goal'] = df['goal'].reset_index().apply(func=unwrap, axis=1)

    df['abs_dist'] = df['goal'].squeeze() - df['s_r']

    df['ga'] = df['abs_dist'].abs() <= 1

    gb = df.groupby(['agent', 'agent_run_id', 'episode_id', 'eval_iter_id'])
    groups = list(gb)

    data = []
    for group in groups:
        group = group[1].reset_index()
        max_idx = group['ga'].idxmax()
        after_goal = group.iloc[max_idx:].reset_index()

        after_goal['perc_success'] = after_goal['ga'].sum() / group['ga'].count()
        data.append(after_goal)

    after_goal = pd.concat(data)
    return df, after_goal


def plot_mean_distance_after_goal(df, barplot=True, ax=None, hue="agent"):
    mag_ep = df.groupby(['goal', 'agent', 'eval_iter_id', 'agent_run_id']).agg(
        {'abs_dist': 'mean', 'ga': 'sum'})

    # mag_ep.columns = mag_ep.columns.map(lambda x: '_'.join([*map(str, x)]))
    mag_ep = mag_ep.reset_index()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        pd.set_option('display.expand_frame_repr', False)

        # print(df[(df['episode_id'] == 6) & (df['agent_run_id'] == 1)])
        print(mag_ep)

    if barplot:
        ordered_goals = mag_ep.goal.value_counts().index
        g = sns.FacetGrid(data=mag_ep, col="goal")
        g.map(sns.barplot, data=mag_ep, x="agent", y="abs_dist", ci="sd", hue=hue)
        plt.legend()

        # sns.barplot(data=mag_ep, x='agent', y='abs_dist', hue=hue, ci="sd", ax=ax)
        # sns.barplot(data=mag_ep, x='goal', y='abs_dist', hue=hue, ci="sd", ax=ax)
    else:
        sns.lineplot(data=mag_ep, x='goal', y='abs_dist', ci="sd", ax=ax, hue=hue)
    plt.show()


def plot_mean_distance_after_goal_overall(df, barplot=True, ax=None, hue="agent"):
    mag_ep = df.groupby(['agent', 'goal']).agg(
        {'abs_dist': ['mean', 'std']})

    mag_ep.columns = mag_ep.columns.map(lambda x: '_'.join([*map(str, x)]))
    mag_ep = mag_ep.reset_index()

    mag_ep_mean = mag_ep.groupby(['agent']).agg({'abs_dist_mean': 'mean'}).reset_index()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        pd.set_option('display.expand_frame_repr', False)

        print(mag_ep)

        print(mag_ep_mean)

    if barplot:

        sns.barplot(data=mag_ep_mean, x='abs_dist_mean', y='agent', hue=hue, ci="sd", ax=ax)
    else:
        sns.lineplot(data=mag_ep_mean, x='agent', y='abs_dist_mean', hue=hue, ci="sd", ax=ax)
    plt.show()


if __name__ == '__main__':
    path = "/Users/Dennis/Desktop/thesis/coding/mfd-rl/sim_analysis/experiments/colormix_3d_simple/param_experiments/results/batch_size/1001_162812_bsize=128/run_1/val_result_step_9999.csv"
    data = pd.read_csv(path)

    rp = RunProcessor()

    result_df, result, ep_array = rp.process(path)

    # print(result_df['goal'])

    print(result_df)

    target_ep = ep_array[1]

    sns.barplot(data=result_df, x='episode_id', y='mean_reward_after_goal')
    plt.show()

    # plot_mean_distance_after_goal(ag, barplot=False)
