import datetime
import time


class TimeProvider(object):

    def __init__(self):
        self.env = None

    def get_time(self):
        raise NotImplementedError("base")

    def advance_time_s(self, seconds, sleep_time=0.2, progress_callback=None):
        done_time = self.get_time() + seconds
        remain = done_time - self.get_time()
        while remain > 0:
            s_time = min(sleep_time, remain)
            start_sleep = self.get_time()
            self.sleep(s_time)
            if progress_callback:
                progress_callback(delta_time=self.get_time() - start_sleep)
            remain = done_time - self.get_time()

    def get_human_time_str(self):
        return str(datetime.timedelta(seconds=self.get_time()))

    def sleep(self, seconds):
        raise NotImplementedError()

    def set_parent_env(self, env):
        self.env = env


class RealTime(TimeProvider):

    def sleep(self, seconds):
        time.sleep(seconds)

    def get_time(self):
        return time.time()

    def advance_time_s(self, seconds, sleep_time=0.2, progress_callback=None):

        if self.env.step_start:
            diff = time.time() - self.env.step_start

            rel_advance = seconds - diff
            #print("Advance realtime: %f seconds" % rel_advance)
            if rel_advance >= 0:
                super(RealTime, self).advance_time_s(rel_advance, sleep_time, progress_callback)
        else:
            #print("Advance realtime: %d seconds" % seconds)
            super(RealTime, self).advance_time_s(seconds, sleep_time, progress_callback)


class SimulatedTime(TimeProvider):
    def sleep(self, seconds):
        self.time += 1
        #print("SimTime", self.get_time(), self.get_time() / 60 / 60, "hrs")

    def __init__(self):
        super().__init__()
        self.time = 0

    def get_time(self):
        return self.time
