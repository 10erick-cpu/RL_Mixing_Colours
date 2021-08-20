
class RepeatStrategy(object):

    def apply_strategy(self, schedule):
        raise NotImplementedError("base")

    def reset(self):
        pass


class Once(RepeatStrategy):
    def apply_strategy(self, schedule):
        pass


class Infinite(RepeatStrategy):

    def apply_strategy(self, schedule):
        schedule.current_idx = 0


class Times(RepeatStrategy):
    def __init__(self, times):
        self.max_repeats = times
        self.num_repeats = 1

    def reset(self):
        self.num_repeats = 1

    def apply_strategy(self, schedule):
        if self.num_repeats < self.max_repeats:
            schedule.current_idx = 0
            self.num_repeats += 1
