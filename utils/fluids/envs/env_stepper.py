import numpy as np

from utils.fluids.envs.env_state.fluid_env_state import State


class StepData(object):
    def __init__(self):
        self.step_count: int = None
        self.max_steps: int = None
        self.action_idx: int = None
        self.action_value: float = None
        self.action_valid: bool = None
        self.action_ch: int = None
        self.reward: float = None
        self.curr_state: State = None
        self.next_state: State = None
        self.goal_achieved: bool = None
        self.raw_obs: np.ndarray = None

        self.sim_success: bool = True
        self.reset_required: bool = False

    def log(self, logger):
        logger.debug("env step: action %f | reward: %f | state: %s" % (self.action_value, self.reward, str(
            self.curr_state.numpy().tolist())))

    def as_env_return(self):
        return self.next_state.numpy(), self.reward, self.reset_required, self


class EnvStepper(object):

    def __init__(self, env):

        self.env = env

    def do_step(self, action_idx):
        env = self.env
        if env.reset_required:
            raise ValueError("Calling step() on env which requires reset")
        data = StepData()

        data.action_idx = action_idx
        data.action_value, data.action_ch, data.action_valid = env.action_handler.handle_action(action_idx)
        data.curr_state = env.state
        data.step_count = env.max_step_counter.counter
        data.max_steps = env.max_step_counter.max_steps
        try:
            env.sim.step()
            data.sim_success = True
        except ValueError as e:
            # self.logger.debug("Overflow catched, resetting environment %s" % str(e))
            # print("env overflow, state", data.curr_state.numpy(), "action", env.action_space()[action_idx])
            return self._on_simulator_error(data)
        data.raw_obs, data.next_state = self._get_state()

        data.goal_achieved = env.goal_criterion.goal_achieved(env.goal_criterion.goal, data)

        if env.done_criterion.is_done(data):
            data.reset_required = True
            # self.logger.debug("done criteria met")

        if env.max_step_counter.is_limit_reached():
            data.reset_required = True
            # self.logger.debug("Max steps reached")
            data.reward = env.max_step_counter.max_step_abort_penalty(data)
        else:
            data.reward = env.reward_provider.get_reward_from_step_data(data)

        env.max_step_counter.step()
        return data

    def _get_state(self):
        return self.env.state_extractor.get_state(self.env.sim)

    def _on_simulator_error(self, data):
        data.reward = self.env.reward_provider.simulator_overflow_reward()
        data.reset_required = True
        data.next_state = self._get_state()
        return data
