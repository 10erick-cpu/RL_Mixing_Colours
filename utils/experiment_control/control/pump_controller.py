from utils.experiment_control.pumps.pump_interface import PumpInterface
from utils.experiment_control.schedule.schedule import Schedule
from utils.experiment_control.schedule.schedule_items import PumpScheduleItem


class PumpController(object):

    def __init__(self, pump: PumpInterface, schedule: Schedule):
        self.pump = pump
        self.schedule = self.set_schedule(schedule)

    def set_schedule(self, schedule):
        self.schedule = schedule

        for event in self.schedule.events:
            if isinstance(event, PumpScheduleItem):
                event.pump = self.pump
        self.schedule.register_callback(self)
        return self.schedule

    def reset(self):
        self.schedule.reset()
        self.pump.stop_transfusion()

    def on_schedulable_error(self, sched, err):
        print("err:", sched, err)

    def on_schedulable_finished(self, sched):
        print("finished:", sched)

    def execute_schedule_blocking(self):
        self.schedule.execute_schedule_blocking()

    def is_active(self):
        return not self.schedule.is_done()

    def update_schedule(self):
        self.schedule.update_step()
