import pickle

import matplotlib.pyplot as plt
import numpy as np

file_sim = "./ColorMix3D_color_range.pkl"
file_real = "./real_env_test_v1.pkl"
file_real_white_red_blue = "./real_env_test_09-17 19:01:47.pkl"
file_real_blue_red = "./real_env_test_09-18 12:37:07.pkl"
file_real_blue_red_2 = "./real_env_test_09-18 13:56:58.pkl"
file_real_water_blue = "./real_env_test_09-18 14:19:47_flush_blue.pkl"
file_real_blue_red_low_conc = "./real_env_test_09-18 18:46:34_flush_blue.pkl"
file_real_blue_red_two_drop_conc = "./real_env_test_09-18 19:26:09_flush_blue.pkl"
file_real_blue_red_two_drop_conc_v2_no_liquid = "./real_env_test_09-18 21:11:49_flush_blue.pkl"
file_real_blue_red_two_drop_conc_v2_morning = "./real_env_test_09-19 10:14:15_flush_blue.pkl"


# from envs.gym_envs.color_mix_env_3d import run, ColorMix3D


def load(file):
    with open(file, 'rb')as f:
        return pickle.load(f)


def plot_data(action_data, idx, ax, alpha=1.0, max_steps=100):
    data = np.asarray(action_data[idx])
    ax.set_title(f"{idx}")
    ax.plot(data[:max_steps, 0], color="r", alpha=alpha)
    ax.plot(data[:max_steps, 1], color="g", alpha=alpha)
    ax.plot(data[:max_steps, 2], color="b", alpha=alpha)
    # ax.set_ylim(bottom=0, top=265)


# env = gym.make('envs:ColorMix3D-v0')

# action_data = ColorMix3D.perform_range_test(env, save_path="./ColorMix3D_color_range.pkl")

def process_batch(data):
    num_exp = len(data)

    result = []

    for d in data:
        r = np.asarray(d)
        result.append(r)

    exp_a = result[::2]
    exp_b = result[1::2]

    return np.asarray(exp_a).mean(axis=0), np.asarray(exp_b).mean(axis=0)


real_data = load(file_real_blue_red_two_drop_conc_v2_morning)

exp_b2r, exp_r2b = process_batch(real_data)

sim_b2r, sim_r2b = process_batch(load(file_sim))
max_steps = 60

f, ax = plt.subplots(1, 2, figsize=(16, 8))
alpha_sim = 0.5

ax[0].plot(exp_b2r[:max_steps, 0], color="r", alpha=1, label="exp_b2r")
ax[0].plot(exp_b2r[:max_steps, 1], color="g", alpha=1)
ax[0].plot(exp_b2r[:max_steps, 2], color="b", alpha=1)

ax[0].plot(sim_b2r[:max_steps, 0], '--', color="r", alpha=alpha_sim, label="sim_b2r")
ax[0].plot(sim_b2r[:max_steps, 1], '--', color="g", alpha=alpha_sim)
ax[0].plot(sim_b2r[:max_steps, 2], '--', color="b", alpha=alpha_sim)
ax[0].legend()
ax[0].set_title("blue to red")

ax[1].plot(exp_r2b[:max_steps, 0], color="r", alpha=1, label="exp_r2b")
ax[1].plot(exp_r2b[:max_steps, 1], color="g", alpha=1)
ax[1].plot(exp_r2b[:max_steps, 2], color="b", alpha=1)

ax[1].plot(sim_r2b[:max_steps, 0], '--', color="r", alpha=alpha_sim, label="sim_r2b")
ax[1].plot(sim_r2b[:max_steps, 1], '--', color="g", alpha=alpha_sim)
ax[1].plot(sim_r2b[:max_steps, 2], '--', color="b", alpha=alpha_sim)
ax[1].legend()
ax[1].set_title("red to blue")

plt.show()

labels = ['b2r', 'r2b']

for i in range(len(real_data)):
    f, ax = plt.subplots(1, 1)

    data_sim = plot_data(load(file_sim), i, ax, alpha=0.3, max_steps=max_steps)
    data_real = plot_data(real_data, i, ax, alpha=1, max_steps=max_steps)
    ax.set_title(labels[i % 2])

    plt.show()
