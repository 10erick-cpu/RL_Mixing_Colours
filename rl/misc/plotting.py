import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from stable_baselines.bench import Monitor




def plot_goal_distribution(df):
    grouped = df.groupby(['goal', 'agent', 'agent_run_id']).agg({'s_r': 'first'}).reset_index()
    print(grouped.head())


def plot_agent_multirun(runs_xy, ax=None, show=False, color=None, label=None):
    grouped = runs_xy.groupby(['agent', 'agent_run_id', 'episode']).agg({'reward': 'mean'}).reset_index()
    print(grouped.head())

    if ax is None:
        ax = plt.gca()
    # data[:, 1] = smooth_dataframe(data[:, 1], 5)

    # sns.lineplot(data=grouped, x='episode', y='reward', hue="agent_run_id", alpha=0.3,palette=sns.color_palette("hls", len(grouped['agent_run_id'].unique())))
    grouped['reward'] = grouped['reward'].rolling(5, min_periods=1).mean()
    sns.lineplot(data=grouped, x='episode', y='reward', hue="agent", palette=sns.color_palette("hls", len(grouped['agent'].unique())))

    # means = data[:, 1].mean(axis=0)
    # stds = data[:, 1].std(axis=0)

    # means_minus_std = means - stds
    # means_plus_std = means + stds
    # ax.plot(means, color=color, label=label)
    # ax.plot(means_minus_std, alpha=0.1, color=color)
    # ax.plot(means_plus_std, alpha=0.1, color=color)

    # ax.fill_between(data[0, 0], means_minus_std, means_plus_std, alpha=0.3, color=color)
    if show:
        plt.show()


def load_result_file(filename):
    """
    Load results from a given file

    :param path: (str) the path to the log file
    :return: (Pandas DataFrame) the logged data
    """
    # get both csv and (old) json files

    if not filename.endswith(Monitor.EXT):
        if os.path.isdir(filename):
            filename = os.path.join(filename, Monitor.EXT)
        else:
            filename = filename + "." + Monitor.EXT
    monitor_files = [filename]
    data_frames = []
    headers = []
    for file_name in monitor_files:
        with open(file_name, 'rt') as file_handler:
            if file_name.endswith('csv'):
                first_line = file_handler.readline()
                assert first_line[0] == '#'
                header = json.loads(first_line[1:])
                data_frame = pandas.read_csv(file_handler, index_col=None)
                headers.append(header)
            elif file_name.endswith('json'):  # Deprecated json format
                episodes = []
                lines = file_handler.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                data_frame = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            data_frame['t'] += header['t_start']
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    data_frame.sort_values('t', inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame['t'] -= min(header['t_start'] for header in headers)
    # data_frame.headers = headers  # HACK to preserve backwards compatibility
    return data_frame
