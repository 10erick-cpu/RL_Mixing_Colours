import copy

import matplotlib.pyplot as plt
import numpy as np

from utils.fluids.envs.env_criteria.base_criterion import Criterion
from utils.fluids.envs.env_state.fluid_env_state import State
from utils.fluids.envs.env_stepper import StepData


class NoMoreGoalsError(ValueError):
    def __init__(self):
        super(NoMoreGoalsError, self).__init__("No more goals available")


class GoalGenerator(object):

    def reset_goal(self, state, num_channels):
        raise NotImplementedError("base")

    def test_setup(self):
        goals = []

        num_samples = 256

        num_runs = 5

        for run in range(num_runs):

            for i in range(num_samples):

                s = State()
                s.device_state = np.asarray([i])

                print("{}/{}".format(i + 1, num_samples))
                for i in self.reset_goal(s, 1):
                    goals.append(i)
                print(goals[-1])

            goals = sorted(goals)
            plt.ioff()
            plt.hist(goals, density=False)

        plt.show()


class RandomGoalGenerator(GoalGenerator):

    def __init__(self, min_goal_val=40, max_goal_val=250):
        self.min_goal = min_goal_val
        self.max_goal = max_goal_val

    def min_goal_val(self, state):
        # return np.ceil(state.device_state) + 1
        return self.min_goal

    def max_goal_val(self, state):
        return self.max_goal

    def reset_goal(self, state, num_channels):
        goal = np.random.uniform(self.min_goal_val(state), self.max_goal_val(state), size=num_channels)
        goal = np.round(goal)
        return goal


class DeterministicGoalGenerator(GoalGenerator):
    def __init__(self, goal_list, allow_index_overflow=False):
        self.goal_list = goal_list
        self.current_idx = None
        self.allow_overflow = allow_index_overflow

    def advance_goal_idx(self):
        if self.current_idx is None:
            self.current_idx = 0
        else:
            self.current_idx += 1
        if self.current_idx >= len(self.goal_list):
            if self.allow_overflow:
                self.current_idx = 0
            else:
                raise NoMoreGoalsError()

    def _get_goal_instance(self, idx):
        goal = copy.deepcopy(self.goal_list[idx])
        if not isinstance(goal, np.ndarray):
            goal = np.asarray(goal)
        return goal

    def reset_goal(self, state, num_channels, auto_advance=False):

        if self.current_idx is None:
            raise IndexError("Goal idx was not advanced yet, call advance_goal_idx() first")
        goal = self._get_goal_instance(self.current_idx)
        if auto_advance:
            self.advance_goal_idx()
        return goal


class GoalCriterion(Criterion):
    def __init__(self, num_channels, goal_generator=RandomGoalGenerator()):
        super().__init__()
        self.__goal = None
        self.num_channels = num_channels
        self.goal_generator = goal_generator
        if isinstance(self.goal_generator, DeterministicGoalGenerator):
            print("WARNING: Deterministic goal generator active")

    @property
    def goal(self):
        return self.__goal

    def __set_goal(self, g):
        self.__goal = g
        if not isinstance(self.__goal, np.ndarray):
            self.__goal = np.asarray([self.goal])

    @goal.setter
    def goal(self, val):
        self.__set_goal(val)

    def goal_achieved_numpy(self, goal, state):
        raise NotImplementedError("base")

    def goal_from_state(self, state_numpy):
        if self.num_channels == 1:
            return state_numpy[:1]
        return state_numpy[:self.num_channels]

    def set_goal_of_state(self, state_numpy, goal):
        if self.num_channels == 1:
            state_numpy[0] = goal
        state_numpy[:self.num_channels] = goal
        return state_numpy

    def state_as_goal(self, state_numpy):
        offset = self.num_channels
        return state_numpy[-offset:]

    def goal_achieved(self, goal, data: StepData) -> bool:
        return self.goal_achieved_numpy(goal, data.next_state.numpy())

    def reset(self, state, num_channels=None):
        if not num_channels:
            num_channels = self.num_channels
        self.__set_goal(self.goal_generator.reset_goal(state, num_channels))

        return self.goal


class FixedGoal(GoalCriterion):

    def __init__(self, goal, num_channels):
        super().__init__(num_channels)
        self.goal = goal

    def goal_achieved_numpy(self, goal, data) -> bool:
        raise NotImplementedError("base")

    def reset(self, state, num_channels=None):
        return self.goal


class RandomLowerBoundMeanGoal(GoalCriterion):

    def goal_achieved_numpy(self, goal, data) -> bool:
        raise NotImplementedError("base")

    def min_goal_val(self, state):
        return state.device_state.mean() + 1
        # return 40

    def max_goal_val(self, state):
        return 240

    def reset(self, state, num_channels=None):
        if num_channels == 1:
            self.goal = np.random.uniform(self.min_goal_val(state), self.max_goal_val(state))
        else:
            self.goal = np.random.uniform(self.min_goal_val(state), self.max_goal_val(state), num_channels)
        self.goal = np.round(self.goal)
        return self.goal
