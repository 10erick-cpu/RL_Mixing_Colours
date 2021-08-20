import numpy as np

from utils.fluids.envs.env_criteria.goals import GoalCriterion, RandomGoalGenerator


class AbsDistGoal(GoalCriterion):
    def __init__(self, goal_generator=RandomGoalGenerator()):
        super().__init__(1, goal_generator=goal_generator)

    def distance_to_goal(self, goal, state):
        return goal - self.state_as_goal(state)

    def goal_achieved_numpy(self, goal, state) -> bool:
        err = np.abs(self.distance_to_goal(goal, state))
        success = all(err <= 1)

        if success:
            # print(" (G) ")
            pass
        return success
