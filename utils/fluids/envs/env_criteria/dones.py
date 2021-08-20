from utils.fluids.envs.env_criteria.base_criterion import Criterion
from utils.fluids.envs.env_stepper import StepData


class DoneCriterion(Criterion):

    def is_done(self, data: StepData):
        raise NotImplementedError("base")


class Never(DoneCriterion):

    def is_done(self, data: StepData):
        return False


class GoalReached(DoneCriterion):
    def is_done(self, data: StepData):
        return self.parent().goal_criterion.goal_achieved(data)


class CustomFn(DoneCriterion):

    def is_done(self, data: StepData):
        pass
