import time

LOG_TEMPL = "{0}: {1:.2f}{2}"


class TimedBlock(object):
    def __init__(self, key, stopwatch):
        self.key = key
        self.sw = stopwatch

    def __enter__(self):
        if not self.sw.enabled:
            return
        self.sw.current_block_count += 1
        self.sw.reset(self.key)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.sw.enabled:
            return
        self.sw.current_block_count -= 1
        self.sw.log_and_reset(self.key)


class MultiStopWatch(object):

    def __init__(self, use_millis=True, enabled=True):
        self.log_dict = dict()
        self.use_millis = True
        self.enabled = enabled
        self.current_block_count = 0

    def set_enabled(self, enabled):
        self.enabled = enabled
        
    def get_val(self, key):
        val = None
        if key in self.log_dict:
            val = self.log_dict[key]
        return val

    def reset(self, key):
        val = self.get_val(key)
        self.log_dict[key] = time.time()
        if self.use_millis:
            return val * 1000 if val is not None else None
        return val
    
    def delta(self, key):
        last_val = self.get_val(key)
        if last_val is None:
            return None
        now = time.time()
        if self.use_millis:
            now *= 1000
        return now - last_val

    def delta_reset(self, key):
        delta = self.delta(key)
        if delta:
            self.reset(key)
        return delta
    
    def _seconds_to_millis(self, seconds):
        return seconds*1000
    
    def reset_if_seconds_passed(self, key, seconds):
        val = self.delta(key)
        if val and val > (self._seconds_to_millis(seconds) if self.use_millis else seconds):
            self.reset(key)
            return True
        return False
        

    def _log(self, data):
        if not self.enabled:
            return data
        data = data.capitalize()
        out = "MSW | " + "\t" * (self.current_block_count - 1) + "{}".format(data)

        print(out)
        return out

    def log(self, key, time_delta):
        scale = "ms" if self.use_millis else "s"
        time_delta = time_delta * 1000 if self.use_millis else time_delta
        log_data = LOG_TEMPL.format(key, time_delta, scale)
        self._log(log_data)

    def log_and_reset(self, key):
        delta = self.delta_reset(key)
        self.log(key, delta)
        return delta

    def timed_block(self, key):

        return TimedBlock(key, self)
