from utils.experiment_control.pumps.activa_a_22 import AdoxActivaA22
from utils.experiment_control.serial_comm.serial_devices import SerialDevice
from utils.fluids.envs.env_criteria.dones import DoneCriterion
from utils.fluids.envs.env_criteria.goals import GoalCriterion
from utils.fluids.envs.env_criteria.rewards import RewardProvider
from utils.fluids.envs.env_criteria.step_limit import StepLimitCounter
from utils.fluids.envs.env_state.fluid_env_initializers import FluidEnvInitializer
from utils.fluids.envs.env_state.state_extractors import MultiChanSimStateExtractor
from utils.fluids.envs.env_utilities import ActionHandler, IntervalPumpManager
from utils.fluids.envs.fluid_env_v2 import FluidEnvBaseV2
from utils.fluids.envs.real_env.real_exp import RealExperiment
from utils.fluids.envs.real_env.real_exp_observer import CameraObserver



def init_pump_fn(port):
    print("Initializing pump @", port)
    return AdoxActivaA22(serial_device=SerialDevice(port))


class RealEnvInitializer(FluidEnvInitializer):

    def init_env(self, env, goal):
        pass

    def grid_val(self, num_channels, goal):
        pass


class RealEnvironment(FluidEnvBaseV2):

    def init_simulator(self, sim_config):
        return RealExperiment(sim_config, self.pump_manager, self.action_handler, self.flush_port, self.camera_observer)

    @staticmethod
    def build_pump_manager(pump_mapping, inf_duration_per_cycle, pump_init_fn=None):
        pump_mngr = IntervalPumpManager(pump_mapping,
                                        schedule_on_time=inf_duration_per_cycle,
                                        pump_init_fn=pump_init_fn if pump_init_fn is not None else init_pump_fn)
        return pump_mngr

    def __init__(self, sim_config, done_criterion: DoneCriterion, pump_mapping, flush_port,
                 reward_provider: RewardProvider, goal_criterion: GoalCriterion, action_handler: ActionHandler,
                 max_step_counter: StepLimitCounter, inf_duration_per_cycle, real_mode=True, return_raw_state=True,
                 camera_obs_id=0):

        self.camera_observer = CameraObserver(camera_obs_id)
        self.flush_port = flush_port

        initializer = RealEnvInitializer()

        state_extractor = MultiChanSimStateExtractor(3)

        if real_mode:
            pump_mngr = self.build_pump_manager(pump_mapping, inf_duration_per_cycle, init_pump_fn)
        else:
            pump_mngr = IntervalPumpManager(pump_mapping,
                                            schedule_on_time=inf_duration_per_cycle)

        super().__init__(sim_config, pump_mngr, state_extractor, done_criterion, reward_provider, goal_criterion,
                         action_handler,
                         max_step_counter, initializer, return_raw_state=return_raw_state)

    def step(self, action_idx):

        r = super(RealEnvironment, self).step(action_idx)
        return r

    def reset(self, **kwargs):
        return super(RealEnvironment, self).reset(**kwargs)
