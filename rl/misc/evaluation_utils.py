import datetime
import os

import imageio
import matplotlib
import numpy as np
import pandas as pd
from stable_baselines.her import HERGoalEnvWrapper

from utils.fluids.envs.env_criteria.goals import DeterministicGoalGenerator
from utils.fluids.envs.env_state.fluid_env_initializers import CustomInitializer
from utils.models.folder import Folder


def calculate_real_time(cycle_time, inf_s_per_cycle, episode_length, num_episodes):
    td = datetime.timedelta(seconds=cycle_time * episode_length * num_episodes)
    return td


def record_agent(action_select_fn, env, num_episodes, out_file_name, max_steps=40, output_fps=5, dgg=None):
    backend = matplotlib.get_backend()

    matplotlib.use('agg')
    images = []
    for i in range(num_episodes):
        print(f"ep {i + 1}/{num_episodes}")
        if dgg is not None:
            dgg.advance_goal_idx()
        obs = env.reset()

        done = False
        step = 0
        while not done:

            action = action_select_fn(obs)
            print(f"step {step + 1}\n|state", obs.tolist(), "\n| action", action)
            obs, rewards, dones, info = env.step(action)
            # env.render("plt", wait_delay=0.1)
            # plt.show()
            if step == 0:
                for _ in range(8):
                    images.append(env.render("rgb_array", wait_delay=0.01, title="reset"))
            else:
                img = env.render("rgb_array", wait_delay=0.1)
                images.append(img)

            step += 1
            if dones or step > max_steps:
                break
    print("processing gif")
    imageio.mimsave(out_file_name, [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=output_fps)

    matplotlib.use(backend)


def load_validation_file(path):
    with open(path, 'rb') as f:
        data = pd.read_csv(f)
    return data


def load_final_validation_file(folder: Folder, fname="val_result_final.csv"):
    path = folder.get_file_path(fname)
    if not os.path.exists(path):
        raise ValueError("File does not exist", path)
    return load_validation_file(path)


def unwrap_goal_dict_to_flat_state(g_dict, is_her_env):
    if not isinstance(g_dict, dict):
        if is_her_env:
            obs = g_dict[:-2]
            goal = g_dict[-1]
            return np.concatenate((obs, [goal])), g_dict

        return g_dict, g_dict
    obs = g_dict['observation']
    goal = g_dict['desired_goal'].squeeze()
    return np.concatenate((obs, [goal])), g_dict


def eval_agent(src_env, action_selection_fn, num_runs=1, view_eps=[], use_random_goals=False,
               goals=None):
    if goals is None:
        goals = [[110], [120], [130], [140], [160], [170], [180], [190]]

    def init_fn(n_ch, g):
        return np.asarray([150, 150, 150])

    if isinstance(src_env, HERGoalEnvWrapper):
        is_her_env = True
        env = src_env.env
    else:
        is_her_env = False
        env = src_env

    old_gg = env.goal_generator()

    dgg = DeterministicGoalGenerator(goals, allow_index_overflow=True)

    env.goal_generator(dgg)

    env.initializer(CustomInitializer(init_fn))

    env = src_env

    result = []

    for eval_iter_id in range(num_runs):
        print(f"\reval run {eval_iter_id + 1}/{num_runs}", end="")

        for episode_id in range(len(goals)):

            dgg.advance_goal_idx()
            s_flat, state = unwrap_goal_dict_to_flat_state(env.reset(), is_her_env)

            step = 1
            while True:

                action_result = action_selection_fn(state)
                if isinstance(action_result, tuple):
                    action_result = action_result[0]

                n_s, reward, done, info = env.step(action_result)
                n_s_flat, n_s = unwrap_goal_dict_to_flat_state(n_s, is_her_env)

                if s_flat.shape[0] == 4:

                    s_r, s_g, s_b, goal = s_flat
                else:
                    (s_r, s_g, s_b), goal = s_flat[:3], s_flat[-1]

                if n_s_flat.shape[0] == 4:

                    n_s_r, n_s_g, n_s_b, goal = n_s_flat
                else:
                    (n_s_r, n_s_g, n_s_b), goal = n_s_flat[:3], n_s_flat[-1]

                data = eval_iter_id, episode_id, step, s_r, s_g, s_b, n_s_r, n_s_g, n_s_b, action_result, reward, goal, done

                result.append(data)

                if reward > 0:
                    raise IndexError

                state = n_s
                s_flat = n_s_flat

                if episode_id in view_eps:
                    env.render("plt", wait_delay=0.01)
                if done:
                    break
                step += 1

    env.reset()

    data_frame = pd.DataFrame(result,
                              columns=['eval_iter_id', 'episode_id', 'step', 's_r', 's_g', 's_b', 'n_s_r', 'n_s_g', 'n_s_b', 'action',
                                       'reward',
                                       'goal', 'done'])

    return data_frame
