from utils.experiment_control.schedule.schedule_items import PumpScheduleItem, ScheduleItemException


class Transfuse(PumpScheduleItem):
    def __init__(self, ul_min):
        super(Transfuse, self).__init__()
        self.ul_min = ul_min

    def _run(self, **kwargs):
        self.pump.start_transfusion(self.ul_min)
        print("Start pump")
        return True


class Stop(PumpScheduleItem):
    def __init__(self):
        super(Stop, self).__init__()

    def _run(self, **kwargs):
        self.pump.stop_transfusion()
        return True

    def _is_run_required(self):
        return self.pump.enabled


class Idle(PumpScheduleItem):
    def __init__(self, seconds):
        super(Idle, self).__init__()

        self.seconds = seconds
        self.start_trigger = None

    def is_done(self):
        if self.start_trigger is None:
            return False
        return self.parent.get_time() - self.start_trigger > self.seconds

    def _is_run_required(self):
        return self.is_done()

    def on_start(self):
        self.start_trigger = self.parent.get_time()

    def on_stop(self):
        self.start_trigger = None

    def _run(self, **kwargs):
        if self.start_trigger is None:
            raise ScheduleItemException("start trigger not set")
        return self.is_done()
