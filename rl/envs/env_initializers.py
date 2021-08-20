import gym
from gym.wrappers import FlattenDictWrapper


class Envs:
    class Default2Fluids(object):

        def __init__(self, sub_env="v0"):
            self.sub_env = sub_env

        def get_plain_env(self):
            return gym.make(f'rl.envs.gym_envs:ColorMix3D-{self.sub_env}')

        def get_flatten_env(self):
            return FlattenDictWrapper(gym.make(f'rl.envs.gym_envs:ColorMix3D-{self.sub_env}'), dict_keys=["observation", "desired_goal"])

    class Default7Fluids(Default2Fluids):
        def __init__(self):
            super().__init__("v1")

    class Default14Fluids(Default2Fluids):
        def __init__(self):
            super().__init__("v2")

    class Saturated3Fluids(object):

        def __init__(self):
            self.sub_env = "v0"

        def get_plain_env(self):
            return gym.make(f'rl.envs.gym_envs:ColorMix3DSaturated-{self.sub_env}')

        def get_flatten_env(self):
            return FlattenDictWrapper(gym.make(f'rl.envs.gym_envs:ColorMix3DSaturated-{self.sub_env}'), dict_keys=["observation", "desired_goal"])

    class Saturated7Fluids(Saturated3Fluids):
        def __init__(self):
            super().__init__()
            self.sub_env = "v1"

    class Saturated14Fluids(Saturated3Fluids):
        def __init__(self):
            super().__init__()
            self.sub_env = "v2"

    class ContinuousDefault3Fluids(object):

        def __init__(self, sub_env="v0"):
            self.sub_env = sub_env

        def get_plain_env(self):
            return gym.make(f'rl.envs.gym_envs:ColorMix3DContinuous-{self.sub_env}')

        def get_flatten_env(self):
            return FlattenDictWrapper(gym.make(f'rl.envs.gym_envs:ColorMix3DContinuous-{self.sub_env}'), dict_keys=["observation", "desired_goal"])

    class Real2Fluids(object):

        def __init__(self, sub_env="v0"):
            self.sub_env = sub_env

        def get_plain_env(self):
            return gym.make(f'rl.envs.gym_envs:ColorMixReal3D-{self.sub_env}')

        def get_flatten_env(self):
            return FlattenDictWrapper(gym.make(f'rl.envs.gym_envs:ColorMixReal3D-{self.sub_env}'), dict_keys=["observation", "desired_goal"])


class EnvBuilder:
    _MODE_DEFAULT = "default"
    _MODE_SAT = "saturated"
    _MODE_REAL = "real"
    _ASPACE_DISC = "discrete"
    _ASPACE_CONT = "cont"

    def __init__(self):
        self.mode = self._MODE_DEFAULT
        self.num_fluids = 2
        self.a_space = self._ASPACE_DISC

    def is_continuous(self):
        return self.a_space == self._ASPACE_CONT

    def is_real(self):
        return self.mode == self._MODE_REAL

    def is_default(self):
        return self.mode == self._MODE_DEFAULT

    def is_discrete(self):
        return self.a_space == self._ASPACE_DISC

    def is_saturated(self):
        return self.mode == self._MODE_SAT

    def real(self):
        self.mode = self._MODE_REAL
        return self

    def default(self):
        self.mode = self._MODE_DEFAULT
        return self

    def saturated(self):
        self.mode = self._MODE_SAT
        return self

    def fluids(self, num_fluids):
        self.num_fluids = num_fluids
        return self

    def continuous(self):
        self.a_space = self._ASPACE_CONT
        return self

    def build(self):
        if self.is_default():
            if self.is_continuous():
                return Envs.ContinuousDefault3Fluids()
            else:
                if self.num_fluids == 2:
                    return Envs.Default2Fluids()
                elif self.num_fluids == 7:
                    return Envs.Default7Fluids()
                elif self.num_fluids == 14:
                    return Envs.Default14Fluids()
                else:
                    raise NotImplementedError()
        elif self.is_real():
            return Envs.Real2Fluids()

        elif self.is_saturated():
            if self.num_fluids == 3:
                return Envs.Saturated3Fluids()
            elif self.num_fluids == 7:
                return Envs.Saturated7Fluids()
            elif self.num_fluids == 14:
                return Envs.Saturated14Fluids()
            else:
                raise NotImplementedError()

        raise NotImplementedError()
