from utils.fluids.envs.env_criteria.base_criterion import Criterion
from utils.fluids.envs.env_stepper import StepData


class StepLimitCounter(Criterion):
    def __init__(self, max_steps):
        super().__init__()
        self.max_steps = max_steps
        self.counter = None

    def is_infinite_counter(self):
        return not self.max_steps or self.max_steps <= 0

    def reset(self):
        self.counter = 0

    def step(self):
        self.counter += 1

    def is_limit_reached(self):
        return not self.is_infinite_counter() and self.max_steps <= self.counter + 1

    def max_step_abort_penalty(self, data: StepData):
        return self.parent().reward_provider.terminal_reward_max_step_count(data)
