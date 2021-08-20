import numpy as np
import torch


class State(object):

    def __init__(self, p_state=None, d_state=None, o_state=None, goal_state=None, pump_history=None):
        self.__pump_states = p_state
        self.__device_state = d_state
        self.__outlet_state = o_state
        self.__goal_state = goal_state
        self.__pump_history = pump_history
        self.__numpy = None


    def numpy(self):
        if self.__numpy is None:



            # self.__numpy = np.asarray(
            #     [*self.goal_state, *self.pump_states, self.outlet_state, *self.device_state])

            self.__numpy = np.asarray(
                [*self.goal_state, *self.device_state])
        return self.__numpy

    def torch(self):
        return torch.from_numpy(self.numpy())

    @property
    def pump_history(self):
        return self.__pump_history

    @pump_history.setter
    def pump_history(self, val):
        self.__pump_history = val

    @property
    def pump_states(self):
        return self.__pump_states

    @pump_states.setter
    def pump_states(self, val):
        self.__pump_states = val

    @property
    def device_state(self):
        return self.__device_state

    @device_state.setter
    def device_state(self, val):
        self.__device_state = val

    @property
    def outlet_state(self):
        return self.__outlet_state

    @outlet_state.setter
    def outlet_state(self, val):
        self.__outlet_state = val

    @property
    def goal_state(self):
        return self.__goal_state

    @goal_state.setter
    def goal_state(self, val):
        self.__goal_state = val
