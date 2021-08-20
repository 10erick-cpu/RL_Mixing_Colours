import numpy as np

from utils.fluids.envs.env_criteria.base_criterion import Criterion
from utils.fluids.envs.env_state.fluid_env_state import State
from utils.fluids.envs.fluid_simulator import FluidSimulator


class SimStateExtractorBase(Criterion):

    def _get_pump_states(self):
        state = self.parent().pump_manager.get_pump_states()

        return np.array(state)

    def _get_device_state(self, sim: FluidSimulator):
        mat = sim.device_state.render('channel_mean')
        return mat

    def _get_pump_history(self, sim:FluidSimulator):
        history_dict = sim.device_state.get_inf_history(window=60)
        flat = [np.sum(history_dict[key]) for key in sorted(history_dict.keys())]
        return flat

    def _get_outlet_state(self, sim: FluidSimulator):
        return sim.device.outlet.ug_min

    def get_state(self, sim: FluidSimulator) -> tuple:
        raise NotImplementedError("base")

    def get_goal(self):
        return self.parent().goal_criterion.goal


class SimStateExtractor(SimStateExtractorBase):

    def get_state(self, sim: FluidSimulator) -> tuple:
        pump_states = self._get_pump_states()
        device_state = self._get_device_state(sim)
        outlet_state = self._get_outlet_state(sim)
        goal_state = self.get_goal()
        pump_history = self._get_pump_history(sim)
        return None, State(pump_states, device_state, outlet_state, goal_state, pump_history=pump_history)


class MultiChanSimStateExtractor(SimStateExtractor):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
