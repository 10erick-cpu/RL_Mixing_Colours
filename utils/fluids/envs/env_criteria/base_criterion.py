class Criterion(object):

    def __init__(self):
        self.__parent_env = None

    def set_parent_env(self, parent_env):
        self.__parent_env = parent_env

    def parent(self):
        return self.__parent_env

    def _unhandled_call(self):
        raise ValueError("Unhandled call")

    def _unexpected_call(self):
        raise ValueError("Not expected to be called")
