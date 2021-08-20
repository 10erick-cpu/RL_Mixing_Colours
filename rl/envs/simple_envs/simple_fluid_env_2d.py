import gym
import numpy as np


class FixedAction(object):
    def __init__(self, chan_id, val):
        self.chan_id = chan_id
        self.val = val

    def apply(self, inf_vals):
        inf_vals[self.chan_id] = self.val
        return inf_vals

    def __repr__(self):
        return "c" + str(self.chan_id) + ":" + str(self.val)


class IncrementalAction(FixedAction):
    def apply(self, inf_vals):
        inf_vals[self.chan_id] += self.val
        return inf_vals


def reward(goal, state_curr_val, action_idx, next_state_curr_val):
    dist = abs(goal - next_state_curr_val[-2:])

    return -dist.mean()


class SimpleFluidEnv2D(gym.Env):
    def __init__(self, max_steps=None, incremental=True):
        self.state = None
        self.goal = None
        self.actions = []

        if incremental:
            for chan in [0, 1]:
                for a_val in [-2, -1, 0, 1, 2]:
                    self.actions.append(IncrementalAction(chan, a_val))
        else:
            for chan in [0, 1]:
                for a_val in [0, 1, 5, 10, 20]:
                    self.actions.append(FixedAction(chan, a_val))

        self.drain = 3
        self.inf_vals = None
        self.updates_per_step = 1
        self.curr_steps = 0
        self.max_steps = max_steps

    def num_observations(self):
        return 2 + 2 + 2

    def action_space(self):
        return self.actions

    def done(self, state):
        return any(state >= 255)

    def to_state(self, state):
        return np.asarray([*self.goal, *self.inf_vals, *state])

    def update(self):
        for i in range(self.updates_per_step):
            up = self.inf_vals - np.random.randint(self.drain, size=2)
            self.state += up / self.updates_per_step
        self.state = np.clip(self.state, a_min=0, a_max=None)
        return self.state

    def step(self, action):
        state = self.state
        self.apply_action(action)
        next_state = self.update()
        r = reward(self.goal, state, action, next_state)
        self.curr_steps += 1
        if self.max_steps and self.curr_steps > self.max_steps:
            return self.to_state(next_state), r, True, None
        return self.to_state(next_state), r, self.done(next_state), None

    def apply_action(self, action_idx):
        self.inf_vals = self.actions[action_idx].apply(self.inf_vals)
        self.inf_vals = np.clip(self.inf_vals, 0, 80)

    def reset(self):
        self.goal = np.random.randint(80, 210, 2)
        self.state = np.random.randint(0, 80, 2).astype(np.float)
        self.inf_vals = np.zeros_like(self.state)
        self.inf_vals = np.random.choice(self.actions).apply(self.inf_vals)
        self.curr_steps = 0
        return self.to_state(self.state)

    def render(self, mode='human'):
        if mode == "mat":
            mat = np.zeros((100, 300, 3))
            mat[:, :, 0] = self.state[0]
            mat[:, :, 1] = self.state[1]
            mat = mat.round().astype(np.uint8)

            return mat
