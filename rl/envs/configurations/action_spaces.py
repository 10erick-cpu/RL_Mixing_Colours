import numpy as np

from utils.fluids.envs.env_utilities import ActionHandler, PredefinedActionSpace


class SimpleDiscrete(ActionHandler):
    APP = [0, 4, 8, 16, 32, 64, 128, 256, 300]

    def __init__(self):
        super().__init__(self.APP, is_incremental_actions=False)


class SimpleDiscreteReal(PredefinedActionSpace):
    ACTION_DICT = {
        0: {"channel": 0, "inf": 0},
        1: {"channel": 0, "inf": 8},
        2: {"channel": 0, "inf": 16},
        3: {"channel": 0, "inf": 32},
        4: {"channel": 0, "inf": 64},
        5: {"channel": 0, "inf": 128},
        6: {"channel": 0, "inf": 256},
        7: {"channel": 0, "inf": 300},
        8: {"channel": 1, "inf": 0},
        9: {"channel": 1, "inf": 8},
        10: {"channel": 1, "inf": 16},
        11: {"channel": 1, "inf": 32},
        12: {"channel": 1, "inf": 64},
        13: {"channel": 1, "inf": 128},
        14: {"channel": 1, "inf": 256},
        15: {"channel": 1, "inf": 300}
    }

    def __init__(self):
        super().__init__(self.ACTION_DICT, is_incremental_actions=False)


class Discrete(ActionHandler):
    APP = [0, 32, 120, 280]

    def __init__(self):
        super().__init__(self.APP, is_incremental_actions=False)


class DiscreteIncremental(ActionHandler):
    APP = [-15, -5, -2, -1, 0, 1, 2, 5, 15]

    def __init__(self):
        super().__init__(self.APP, is_incremental_actions=True)


class Continuous(ActionHandler):

    def __init__(self):
        super().__init__({})

    def handle_action(self, action_idx):
        assert isinstance(action_idx, np.ndarray)
        valid = []
        for ch, action in enumerate(action_idx):
            self.adjust_channel(ch, action)
            valid.append(0 <= action <= 300)

        return action_idx, np.arange(len(action_idx)), np.asarray(valid)
