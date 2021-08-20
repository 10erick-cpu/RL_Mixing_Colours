import matplotlib.pyplot as plt
import numpy as np


class FluidEnvInitializer(object):

    def init_env(self, env, goal):
        raise NotImplementedError("base")

    def test_setup(self, num_channels=3):
        goals = []
        num_samples = 10000
        for i in range(num_samples):
            print("{}/{}".format(i + 1, num_samples))
            for i in self.grid_val(num_channels).flatten():
                goals.append(round(i))
            print(goals[-1])

        plt.ioff()
        plt.hist(goals)

        plt.show()

    def grid_val(self, num_channels, goal):
        raise NotImplementedError("")


class CustomInitializer(FluidEnvInitializer):

    def __init__(self, val_fn):
        self.val_fn = val_fn

    def init_env(self, env, goal):
        env.sim.device_state.reset_fixed_channel_wise(val=self.grid_val(env.channel_count(), goal))

    def grid_val(self, num_channels, goal):
        return self.val_fn(num_channels, goal)


class SameInitializer(FluidEnvInitializer):
    def grid_val(self, num_channels, goal):
        return self.value

    def __init__(self, value=0):
        self.value = value

    def init_env(self, env, goal):
        env.sim.device_state.reset_fixed(val=self.grid_val(env.channel_count(), goal))


class RandomInitializer(FluidEnvInitializer):

    def __init__(self, grid_min=0, grid_max=None, pump_min=0, pump_max=280):
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.pump_min = pump_min
        self.pump_max = pump_max

    def grid_val(self, num_channels, goal):
        if self.grid_max:
            if self.grid_min:
                return np.random.uniform(self.grid_min, self.grid_max, size=num_channels)
            return np.random.uniform(0, self.grid_max, size=num_channels)
        return np.random.uniform(0, goal, size=num_channels)

    def init_env(self, env, goal):
        # env.sim.device_state.reset_random(min_val=self.grid_min, max_val=env.sim.device.vol_per_pix())

        grid_vals = self.grid_val(env.channel_count(), goal)

        env.sim.device_state.reset_fixed_channel_wise(grid_vals)
        # for ctrl in env.sim.ctrl.controllers:
        # ctrl.pump.start_transfusion(np.random.uniform(self.pump_min, self.pump_max))


class RandomSameInitializer(RandomInitializer):

    def init_env(self, env, goal):
        # grid_val = np.random.uniform(self.grid_min, env.sim.device.vol_per_pix())
        grid_val = self.grid_val(env.channel_count(), goal)
        # print("RandomSame:", grid_val)
        env.sim.device_state.reset_fixed_channel_wise(grid_val)
        # env.sim.device_state.reset_fixed(grid_val)

        # for ctrl_name, ctrl in env.sim.ctrl.controllers.items():
        #   ctrl.pump.start_transfusion(np.random.choice(env.action_space()))
