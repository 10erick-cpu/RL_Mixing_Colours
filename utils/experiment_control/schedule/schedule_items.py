import sys


class ScheduleItemException(Exception):
    def __init__(self, e):
        super(ScheduleItemException, self).__init__(e)


class ScheduleItem(object):
    def __init__(self):
        self.parent = None

    def trigger_time(self):
        return self.trigger_time

    def register_callback(self, parent):
        self.parent = parent

    def _is_run_required(self):
        return True

    def on_start(self):
        pass

    def on_stop(self):
        pass

    def reset(self):
        self.on_stop()

    def __repr__(self):
        return self.__class__.__name__

    def notify_parent(self, err=None):
        if not self.parent:
            print("no schedule to notify")
            return False
        if err:
            return self.parent.on_schedulable_error(self, err)
        return self.parent.on_schedulable_finished(self)

    def on_update(self, caller):
        if not self._is_run_required():
            return False

        try:
            if self._run():
                self.notify_parent()
            return True
        except Exception:
            s = ScheduleItemException(sys.exc_info())
            self.notify_parent(s)

    def _run(self, **kwargs) -> bool:
        raise NotImplementedError("base")


class PumpScheduleItem(ScheduleItem):
    def __init__(self):
        super().__init__()
        self.pump = None

    def _run(self, **kwargs) -> bool:
        raise NotImplementedError("base")


class LambdaScheduleItem(ScheduleItem):
    def reset(self):
        pass

    def __init__(self, lambda_fn, **kwargs):
        super().__init__()
        self.fn = lambda_fn
        self.kwargs = kwargs

    def _run(self, **kwargs):
        self.fn(self.kwargs)
