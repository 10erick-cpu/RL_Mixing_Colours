import numpy as np
from utils.fluids.envs.env_criteria.base_criterion import Criterion
from utils.fluids.envs.env_state.fluid_env_state import State
from utils.fluids.envs.env_stepper import StepData


class RewardProvider(Criterion):
    MAX_REWARD = 100

    def __init__(self):
        super(RewardProvider, self).__init__()

    def get_reward_from_step_data(self, data: StepData):
        return self.get_reward(data.curr_state.numpy(), data.action_idx, data.next_state.numpy())

    def get_reward(self, state, action, next_state):
        raise NotImplementedError("base")

    def invalid_action_reward(self):
        self._unhandled_call()

    def simulator_overflow_reward(self):
        self._unhandled_call()

    def terminal_reward(self):
        self._unhandled_call()

    def terminal_reward_max_step_count(self, data: StepData):
        self._unhandled_call()

    def debug_reward_info(self):
        # requires update
        data = [State(d_state=np.asarray(s)) for s in range(254)]
        rewards = []
        for i, s_n in enumerate(data):
            s = data[i - 1] if 0 <= i <= len(data) else data[i]
            rewards.append(self.get_reward(None, None, s, s_n))
        return rewards
