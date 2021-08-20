from utils.experiment_control.schedule.pump_actions import Transfuse, Stop, Idle
from utils.experiment_control.schedule.repeat_strategy import Once
from utils.experiment_control.schedule.schedule import Schedule
from utils.experiment_control.schedule.schedule_items import ScheduleItem
from utils.fluids.time_providers import RealTime


class ScheduleBuilder(object):
    def __init__(self):
        self.__events = []
        self.__repeat_strategy = Once()
        self.__tp = RealTime()

    def with_time(self, tp):
        self.__tp = tp

    def _add_event(self, event):
        self.__events.append(event)

    def add_schedule(self, schedule):
        self._add_event(schedule)
        return self

    def start_transfuse(self, ul_min):
        self._add_event(Transfuse(ul_min))
        return self

    def stop_transfuse(self):
        self._add_event(Stop())
        return self

    def wait(self, seconds):
        self._add_event(Idle(seconds))
        return self

    def with_schedulable(self, schedulable: ScheduleItem):
        self._add_event(schedulable)
        return self

    def repeat(self, repeat_strategy):
        self.__repeat_strategy = repeat_strategy
        return self

    def __check(self):
        assert self.__repeat_strategy is not None
        assert len(self.__events) > 0, "empty schedule queue"

    def build(self):
        self.__check()

        # return Schedule(copy.deepcopy(self.__events), copy.deepcopy(self.__repeat_strategy), copy.deepcopy(self.__tp))
        return Schedule(self.__events, self.__repeat_strategy, self.__tp)

    # def build_for_pump(self, pump):
    #     self.__check()
    #     events = list(self.__events)
    #
    #     for event in events:
    #         if isinstance(event, PumpSchedulable):
    #             event.pump = pump
    #     return Schedule(self.__events, self.__repeat_strategy, time_provider=self.tp)
