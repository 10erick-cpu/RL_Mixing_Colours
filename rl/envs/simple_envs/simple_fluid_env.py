import gym
import numpy as np

from agents.replay_buffer import HerBuffer


def reward(goal, state_curr_val, action_idx, next_state_curr_val):
    dist = abs(goal - next_state_curr_val)

    # if 0 <= dist <= 5:
    #     return 1 + (1 - (dist / 5))
    # return -1

    return 1 if dist <= 1 else 0


class SimpleFluidEnv(gym.Env):
    def __init__(self, max_steps=None):
        self.val = None
        self.goal = None
        self.actions = [0, 1, 2, 4, 5, 10, 15, 20]
        self.drain = 5
        self.inf_val = None
        self.updates_per_step = 1
        self.inf_damp_factor = 1
        self.curr_steps = 0
        self.max_steps = max_steps

    def num_observations(self):
        return 3

    def action_space(self):
        return self.actions

    def done(self, state):
        return state >= 255

    def to_state(self, state):
        return np.asarray([self.goal, np.asarray(state), self.inf_val])

    def update(self):
        for i in range(self.updates_per_step):
            up = self.inf_val - np.random.randint(0, self.drain)
            up = up * self.inf_damp_factor
            self.val += up / self.updates_per_step

        self.val = max(self.val, 0)
        return self.val

    def step(self, action):
        state = self.val
        self.apply_action(action)
        next_state = self.update()
        r = reward(self.goal, state, action, next_state)
        self.curr_steps += 1
        if self.max_steps and self.curr_steps > self.max_steps:
            return self.to_state(next_state), r, True, None

        done = self.done(next_state)

        return self.to_state(next_state), r, done, None

    def apply_action(self, action_idx):
        self.inf_val = self.actions[action_idx]
        self.inf_val = np.clip(self.inf_val, 0, 30)

    def reset(self):
        self.goal = np.random.randint(80, 210)
        self.val = np.random.randint(0, self.goal)
        self.curr_steps = 0
        self.inf_val = np.random.choice(self.actions)
        return self.to_state(self.val)

    def render(self, mode='human'):
        if mode == "mat":
            mat = np.zeros((100, 300, 3))
            mat[:, :, 0] = self.val
            mat = mat.round().astype(np.uint8)
            return mat
        if mode == "human":
            return True
        raise ValueError("Unknown mode %s" % mode)
