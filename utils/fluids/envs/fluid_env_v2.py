import gym

from utils.fluids.envs.env_criteria.dones import DoneCriterion
from utils.fluids.envs.env_criteria.goals import GoalCriterion, GoalGenerator
from utils.fluids.envs.env_criteria.rewards import RewardProvider
from utils.fluids.envs.env_criteria.step_limit import StepLimitCounter
from utils.fluids.envs.env_state.fluid_env_initializers import FluidEnvInitializer
from utils.fluids.envs.env_state.state_extractors import SimStateExtractor
from utils.fluids.envs.env_stepper import EnvStepper
from utils.fluids.envs.env_utilities import ActionHandler, PumpManager
from utils.fluids.envs.fluid_simulator import FluidSimulator
from utils.helper_functions.torch_utils import training_device


class FluidEnvBaseV2(gym.Env):

    def step(self, action_idx):
        # start = time.time()
        result = self.env_stepper.do_step(action_idx)
        # print("step iteration: %.4f" % (time.time() - start))

        self._state = result.next_state

        self.reset_required = result.reset_required

        if self.return_raw_state:
            return result

        return result.raw_obs, result.as_env_return()

    def reset(self, **kwargs):
        self.sim.reset(**kwargs)
        self.reset_required = False
        self.max_step_counter.reset()

        self.goal_criterion.reset(self._state, )
        self.initializer.init_env(self, self.goal_criterion.goal)

        raw_obs, self._state = self.state_extractor.get_state(self.sim)

        if self.return_raw_state:
            return raw_obs, self.state

        return raw_obs, self.state.numpy()

    def render(self, mode='human'):
        result = self.sim.render(mode)
        return result

    def set_goal_generator(self, generator: GoalGenerator):
        self.goal_criterion.goal_generator = generator

    def num_observations(self):
        goal_count = self.goal_criterion.num_channels
        pump_count = len(self.pump_manager.get_pump_states())
        num_inlets = len(self.sim.device.inlets)
        if self.sim.device.water_channel_id:
            num_inlets -= 1
        outlet = 1 if self.sim.device.outlet else 0

        return goal_count + pump_count + outlet + num_inlets

    def observation_space(self):
        return 1, self.num_observations()

    def action_space(self):
        return self.action_handler.actions

    def channel_count(self):
        return self.sim.config.NUM_CHANNELS

    @property
    def state(self):
        return self._state

    def init_simulator(self, sim_config):
        return FluidSimulator(sim_config)

    def set_pump_manager(self, manager=None):
        if manager is None:
            return self.pump_manager
        self.pump_manager = manager
        self._do_setup()
        return self.pump_manager

    def __init__(self, sim_config, pump_manager: PumpManager,
                 state_extractor: SimStateExtractor,
                 done_criterion: DoneCriterion,
                 reward_provider: RewardProvider,
                 goal_criterion: GoalCriterion,
                 action_handler: ActionHandler,
                 max_step_counter: StepLimitCounter,
                 initializer: FluidEnvInitializer,
                 train_device=training_device(),
                 return_raw_state=False):
        self.action_handler = action_handler
        self.return_raw_state = return_raw_state
        self.env_stepper = EnvStepper(self)
        self._state = None
        self.reset_required = True
        self.pump_manager = pump_manager
        self.sim = self.init_simulator(sim_config)
        self.state_extractor = state_extractor
        self.done_criterion = done_criterion
        self.reward_provider = reward_provider
        self.goal_criterion = goal_criterion

        self.train_device = train_device
        self.max_step_counter = max_step_counter
        self.initializer = initializer

        self._do_setup()

    def _do_setup(self):
        self.done_criterion.set_parent_env(self)
        self.reward_provider.set_parent_env(self)
        self.state_extractor.set_parent_env(self)
        self.goal_criterion.set_parent_env(self)
        self.max_step_counter.set_parent_env(self)

        self.action_handler.init(self.pump_manager)
        self.pump_manager.connect_device(self.sim.device, self.sim.tp)
        self.sim.set_controller(self.pump_manager)
