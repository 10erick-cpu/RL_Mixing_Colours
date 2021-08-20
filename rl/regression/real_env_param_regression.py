import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import L1Loss
from torch.nn.modules.loss import SmoothL1Loss
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from envs.gym_envs import ColorMix3D
from fluids.simulation_devices import ColorMixer


def get_data_loader(ds, idx=None, batch_size=1, num_workers=4, shuffle=True):
    if idx is None:
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
    else:
        return DataLoader(ds, num_workers=num_workers, batch_size=batch_size, sampler=SubsetRandomSampler(idx))


def get_range_data():
    env = gym.make("ColorMix3D-v0")

    return ColorMix3D.perform_range_test(env), env.env.sim.device.vol_per_pix()
    # return env


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 60)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(60, 9)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        out = self.linear(x)
        out = self.relu1(out)
        out = self.linear2(out)
        return out

    def get_m(self):
        return self(torch.tensor([1]).float()).clamp(min=-1, max=1)

    def color_mixer_forward(self, distribution, intensity, M):
        m = M.float().repeat(distribution.shape[0], 1, 1)
        # m = M
        dist = torch.bmm(distribution, m)
        return intensity - intensity * dist

    def simulate_for_output(self, m_hat, sim_data):
        return self.color_mixer_forward(sim_data, 180, m_hat.view(-1, 3, 3))

    def learn(self, real_data, sim_data, criterion, optimizer):
        # get loss for the predicted output

        m_hat = self.get_m()

        sim_output = self.simulate_for_output(m_hat, sim_data)
        loss = criterion(sim_output, real_data)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()


def load_real_file(path, plot=False, num_steps=70):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    data = np.asarray(data)
    print(data.shape)
    import matplotlib.pyplot as plt

    mean_b2r = data[::2, :].mean(axis=0)
    mean_r2b = data[1::2, :].mean(axis=0)

    mean_b2r = mean_b2r[:num_steps]
    mean_r2b = mean_r2b[:num_steps]

    if plot:
        print(mean_r2b)
        plt.plot(data[0, :, 0], c="r")
        plt.plot(data[0, :, 1], c="g")
        plt.plot(data[0, :, 2], c="b")

        plt.plot(mean_b2r[:, 0], c="r", linestyle="dashed")
        plt.plot(mean_b2r[:, 1], c="g", linestyle="dashed")
        plt.plot(mean_b2r[:, 2], c="b", linestyle="dashed")
        plt.show()
    return mean_b2r, mean_r2b


def old():
    load = False

    if not load:
        data_b2r, data_r2b = load_real_file(
            "/Users/Dennis/Desktop/thesis/coding/mfd-rl/envs/gym_envs/real_env_test_09-19 10:14:15_flush_blue.pkl")

        data, grid_states = get_range_data()

        sim_data = np.asarray(grid_states)
        sim_data = sim_data[:, :, :3]

        sim_data_b2r = sim_data[0, :]
        sim_data_r2b = sim_data[1, :]

        real_data = data_r2b
        ##########

        real_data = np.ascontiguousarray(real_data[:60])
        sim_data = np.ascontiguousarray(sim_data_r2b)
        print(real_data.shape)
        print(sim_data_b2r.shape)
        with open("sim_data.pkl", 'wb') as f:
            pickle.dump(sim_data, f)
        with open("real_data.pkl", 'wb') as f:
            pickle.dump(real_data, f)
    else:
        with open("./sim_data.pkl", 'rb') as f:
            sim_data = pickle.load(f)
        with open("./real_data.pkl", 'rb') as f:
            real_data = pickle.load(f)

    sim_data = torch.from_numpy(sim_data).float().unsqueeze(1)
    real_data = torch.from_numpy(real_data).float().unsqueeze(1)

    print(sim_data.shape)
    print(real_data.shape)

    test_M = torch.tensor([
        [-0.2, 1, 1],
        [1, -0.2, 1],
        [1, 1, -0.2]
    ])

    model = LinearRegression()

    criterion = SmoothL1Loss()
    num_epochs = 500

    print(test_M.shape)

    for i in range(sim_data.shape[0]):
        result = model.color_mixer_forward(sim_data[i][None], 200, test_M)
        print(result)
        print(real_data[i])
        print()

    print(result.shape)
    print(result)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    last_loss = 1000

    while True:
        loss = model.learn(real_data, sim_data, criterion, optimizer)
        print(loss)
        break
        if abs(last_loss - loss) <= 0.1 and loss <= 0.99:
            break
        last_loss = loss

    print(model(real_data.view(-1)).view(3, 3))

    M = np.asarray([[-0.1096, 0.9114, 0.6606],
                    [0.6285, 0.0719, -0.1047],
                    [0.5207, 0.2459, 0.0463]])

    c = ColorMixer()
    c.M = M

    data = []
    for i in range(sim_data.shape[0]):
        result = c.forward(sim_data[i].squeeze().numpy(), 180)
        data.append(result)
        print("sim:", result.tolist())
        print("real:", real_data[i].numpy().squeeze().tolist())
        print()

    import matplotlib.pyplot as plt

    data = np.asarray(data)
    print(data.shape)
    plt.plot(data[:, 0], c='r', linestyle="dashed")
    plt.plot(data[:, 1], c='g', linestyle="dashed")
    plt.plot(data[:, 2], c='b', linestyle="dashed")
    real_data = real_data.numpy().squeeze()
    print(real_data.shape)
    plt.plot(real_data[:, 0], c='r')
    plt.plot(real_data[:, 1], c='g')
    plt.plot(real_data[:, 2], c='b')
    plt.show()


def create_data():
    # real_data = pd.read_csv("/Users/Dennis/Desktop/thesis/coding/mfd-rl/envs/gym_envs/real_env_test_all.csv")

    sim_file = "/Users/Dennis/Desktop/thesis/coding/mfd-rl/regression/sim_data.pkl"
    with open(sim_file, 'rb') as f:
        sim_data = pickle.load(f)

    (_, dist), vol_per_pix = get_range_data()

    b2r, r2b = dist

    data = []
    for idx, step in enumerate(b2r):
        print(step)

        r, g, b, w = step
        data.append({'type': 'b2r', 'ep_id': 0, 'step': idx, 'r': r, 'g': g, 'b': b, 'src': 'sim'})

    for idx, step in enumerate(r2b):
        r, g, b, w = step
        data.append({'type': 'r2b', 'ep_id': 0, 'step': idx, 'r': r, 'g': g, 'b': b, 'src': 'sim'})

    df = pd.DataFrame(data)
    df.to_csv("./df_sim.csv")


def read_data():
    real_data = pd.read_csv("../envs/gym_envs/real_env_test_all.csv")
    real_data = real_data[real_data['step'] < 60]

    sim_data = pd.read_csv("./df_sim.csv")

    b2r = enhance_real_data(real_data, sim_data, mode="b2r")
    r2b = enhance_real_data(real_data, sim_data, mode="r2b")

    print(b2r)

    data = pd.concat([b2r, r2b])

    return data

    # real_b2r = real_data[real_data['type'] == "b2r"]
    # sim_steps_b2r = real_b2r['step']
    # steps_numpy = sim_steps_b2r.to_numpy()
    #
    # sim_dist_b2r = sim_data[sim_data['type'] == "b2r"]
    # b2r_rgb = sim_dist_b2r[['r', 'g', 'b']].to_numpy()
    #
    # b2r = real_b2r[['r', 'g', 'b']].copy()
    #
    # b2r['target_r'] = b2r_rgb[steps_numpy][:, 0]
    # b2r['target_g'] = b2r_rgb[steps_numpy][:, 1]
    # b2r['target_b'] = b2r_rgb[steps_numpy][:, 2]


def enhance_real_data(df_real, df_sim, mode="r2b"):
    real_mode_filtered = df_real[df_real['type'] == mode]
    steps_mode_filtered = real_mode_filtered['step']
    steps_numpy = steps_mode_filtered.to_numpy()

    sim_dist = df_sim[df_sim['type'] == mode]
    rgb_sim_dist_numpy = sim_dist[['r', 'g', 'b']].to_numpy()

    mode_filtered = real_mode_filtered[['r', 'g', 'b']].copy()

    mode_filtered['target_r'] = rgb_sim_dist_numpy[steps_numpy][:, 0]
    mode_filtered['target_g'] = rgb_sim_dist_numpy[steps_numpy][:, 1]
    mode_filtered['target_b'] = rgb_sim_dist_numpy[steps_numpy][:, 2]
    return mode_filtered


class SimpleDataset(Dataset):
    def __init__(self, pd_frame):
        real_vals = torch.from_numpy(pd_frame[['r', 'g', 'b']].to_numpy()).float()
        sim_dists = torch.from_numpy(pd_frame[['target_r', 'target_g', 'target_b']].to_numpy()).float()
        self.data = list(zip(real_vals, sim_dists))

    def __getitem__(self, item):
        real_val, sim_dist = self.data[item]

        return real_val.unsqueeze(0), sim_dist.unsqueeze(0)

    def __len__(self):
        return len(self.data)


class EpisodeDataset(Dataset):
    def __init__(self, pd_frame):
        real_vals = torch.from_numpy(pd_frame.groupby(['type'])[['r', 'g', 'b']].to_numpy()).float()
        sim_dists = torch.from_numpy(pd_frame[['target_r', 'target_g', 'target_b']].to_numpy()).float()
        self.data = list(zip(real_vals, sim_dists))

    def __getitem__(self, item):
        real_val, sim_dist = self.data[item]

        return real_val.unsqueeze(0), sim_dist.unsqueeze(0)

    def __len__(self):
        return len(self.data)


def fit_model(dataframe):
    real_vals = torch.from_numpy(dataframe[['r', 'g', 'b']].to_numpy()).float()
    sim_dists = torch.from_numpy(dataframe[['target_r', 'target_g', 'target_b']].to_numpy())

    dataset = list(zip(real_vals, sim_dists))

    dataset = SimpleDataset(dataframe)

    model = LinearRegression()

    criterion = L1Loss()
    num_epochs = 600

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train = False
    if train:
        last_loss = 1000
        batch_size = len(dataset)

        for i in range(num_epochs):
            print(i + 1, "/", num_epochs)
            data_loader = get_data_loader(dataset, batch_size=batch_size, shuffle=True)
            for batch_id, (real_data, sim_data) in enumerate(data_loader):
                print(real_data.shape)
                loss = model.learn(real_data, sim_data, criterion, optimizer)
                print(loss)

                if abs(last_loss - loss) <= 0.1 and loss <= 0.99:
                    break
                last_loss = loss
        with open("./reg_model.model", "wb") as f:
            torch.save(model.state_dict(), f)
    else:
        with open("./reg_model.model", 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)

    with torch.no_grad():
        real = []
        prediction = []

        print(model.get_m().view(-1, 3, 3
                                 ))

        for idx, (real_data, sim_data) in enumerate(dataset):

            #m = model.get_m()
            #print(m.view(-1, 3, 3).numpy())
            m = torch.from_numpy(np.asarray([[-0.0984, 0.7386, 0.5415],
                            [0.7000, -.1000, .5000],
                            [0.6502, 0.3318, -0.1884]]))

            d= torch.from_numpy(np.asarray([[-0.2, 1, 1],
                                             [1, -0.2, 1],
                                             [1, 1, -0.2]]))


            predicted = model.simulate_for_output(m, sim_data.unsqueeze(0))
            print("real", real_data.numpy(), "pred", predicted.numpy())
            real.append(real_data.numpy())
            prediction.append(predicted.numpy())
            print()

            if idx and (idx + 1) % 60 == 0:
                real = np.asarray(real).squeeze()
                prediction = np.asarray(prediction).squeeze()
                plt.plot(real[:, 0], c="r")
                plt.plot(real[:, 1], c="g")
                plt.plot(real[:, 2], c="b")

                plt.plot(prediction[:, 0], c="r", linestyle="dashed")
                plt.plot(prediction[:, 1], c="g", linestyle="dashed")
                plt.plot(prediction[:, 2], c="b", linestyle="dashed")
                plt.show()
                real = []
                prediction = []


fit_model(read_data())
