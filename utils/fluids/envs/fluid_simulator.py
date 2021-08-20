import time

from utils.fluids.simulation_devices import ColorMixTestDevice, MfdState
from utils.fluids.simulation_engines import BlurEngine, BasicMeanEngine
from utils.fluids.time_providers import SimulatedTime
from utils.models.dot_dict import DotDict


class SimulatorConfig(object):
    NUM_CHANNELS = 1
    TIME_PROVIDER = SimulatedTime()

    DEVICE = ColorMixTestDevice()

    SIMULATION_SECONDS_STEPS = 60

    ENGINE = DotDict({'type': BlurEngine(127), 'overflow_smooth': True})

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


class FluidSimulator(object):
    def __init__(self, sim_config=SimulatorConfig()):
        super(FluidSimulator, self).__init__()
        self.config = sim_config
        self.device = sim_config.DEVICE
        self.device_state = MfdState(self.device)
        self.engine = sim_config.ENGINE['type']

        self.overflow_smooth = sim_config.ENGINE['overflow_smooth']

        self.tp = sim_config.TIME_PROVIDER
        self.tp.set_parent_env(self)

        self.sim_steps_per_second = sim_config.SIMULATION_SECONDS_STEPS
        self.ctrl = None
        self.steps = None
        self.step_start = None

    def set_controller(self, controller):
        self.ctrl = controller

    def reset(self, reset_device=True, **kwargs):
        self.steps = 0
        self.ctrl.reset()
        if reset_device or self.device_state.grid is None:
            self.device_state.reset_fixed(0)

    def step(self, output=False, delta_time=None):
        self.step_start = time.time()

        self._update(self.sim_steps_per_second if delta_time is None else delta_time)

    def _on_advance_time(self, delta_time=1):
        self.ctrl.run_step()
        self.device_state.update(self.engine, delta_time)

    def _update(self, delta_time=1):
        self.ctrl.run_step()
        self.tp.advance_time_s(delta_time, progress_callback=self._on_advance_time)
        self.steps += 1

    def render(self, mode='real'):
        return self.device_state.render(mode)

    def close(self):
        pass
