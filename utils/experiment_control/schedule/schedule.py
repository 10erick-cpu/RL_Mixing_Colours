from utils.experiment_control.schedule.repeat_strategy import Once
from utils.experiment_control.schedule.schedule_items import ScheduleItem, ScheduleItemException
from utils.fluids.time_providers import RealTime


class Schedule(ScheduleItem):

    def __init__(self, schedulables, repeat_strategy=Once(), time_provider=RealTime()):
        super().__init__()
        self.events = list(schedulables)
        self.current_idx = None
        self.start_time = None
        self.stop_time = None
        self.num_events = len(self.events)
        self.repeat_handler = repeat_strategy
        self.tp = time_provider

    def init(self):
        self.start_time = None
        self.stop_time = None
        self.current_idx = 0
        for elem in self.events:
            elem.register_callback(self)
            elem.reset()

    def set_time_provider(self, tp):
        self.tp = tp

    def is_initialized(self):
        return self.current_idx is not None

    def is_done(self):
        if self.current_idx == self.num_events:
            self.repeat_handler.apply_strategy(self)
        if self.current_idx == self.num_events:
            self.stop_time = self.get_time()
            return True
        return False

    def reset(self):
        self.init()
        self.repeat_handler.reset()

    def on_start(self):
        self.init()
        self.start_time = self.get_time()
        self._advance_current_ptr(advance_to_idx=self.current_idx)

    def on_stop(self):
        self.stop_time = self.get_time()
        print("schedule finished")

    def on_schedulable_finished(self, schedulable):
        self._advance_current_ptr()

    def on_schedulable_error(self, schedulable, err):
        print(err)
        raise err

    def get_time(self):
        return self.tp.get_time()

    def _advance_current_ptr(self, advance_to_idx=None):
        self.current().on_stop()

        if advance_to_idx is not None:
            self.current_idx = advance_to_idx
        else:
            self.current_idx += 1

        if not self.is_done():
            self.current().on_start()

    def current(self):
        if self.current_idx >= len(self.events):
            return self.events[-1]
        return self.events[self.current_idx]

    def __repr__(self):
        data = [repr(event) for event in self.events]
        if self.current_idx is not None:
            data[self.current_idx] += "**"
        return " ->> ".join(data)

    def execute_schedule_blocking(self):
        if not self.is_initialized():
            if not self.is_initialized():
                print("Initializing schedule", self)
                self.on_start()

        while not self.is_done():
            self.current().on_update(self)
        self.stop_time = self.get_time()

    def update_step(self):
        if not self.is_initialized():
            self.on_start()
        if not self.is_done():
            while self.current().on_update(self):
                pass
        else:
            self.parent.on_schedulable_error(self, ScheduleItemException("end of schedule reached"))
        return self.is_done()

    def _run(self, **kwargs):
        self.update_step()
        if self.is_done():
            self.parent.on_schedulable_finished()
        return True
