import time

from utils.experiment_control.control.pump_controller import PumpController


class ExperimentExecutor(object):
    def __init__(self):
        self.controllers = dict()
        self.finished = dict()

    def register_pump(self, pump, schedule, time_provider, allow_override=False):
        if not allow_override:
            assert pump.name not in self.controllers
        schedule.set_time_provider(time_provider)
        self.controllers[pump.name] = PumpController(pump, schedule)

    def reset(self):
        for ctrl_name, ctrl in self.controllers.items():
            ctrl.reset()

        for ctrl_name, ctrl in self.finished.items():
            ctrl.reset()
            self.controllers[ctrl_name] = ctrl
        self.finished.clear()

    def clear_data(self):
        self.controllers.clear()
        self.finished.clear()

    def run_step(self):

        if len(self.controllers) > 0:
            ctrls = list(self.controllers.items())
            for ctrl_name, ctrl in ctrls:
                if ctrl_name not in self.finished:

                    if ctrl.is_active():
                        ctrl.update_schedule()

                    else:

                        self.on_schedule_finished(ctrl_name, ctrl)
            return True
        return False

    def on_schedule_finished(self, ctrl_name, pump_ctrl):

        self.finished[ctrl_name] = pump_ctrl
        self.controllers.pop(ctrl_name)

    def run(self, time_provider, update_per_second=1, progress_interval=None):
        last_update = time_provider.get_time()

        while self.run_step():

            time.sleep(1 / update_per_second)
            if progress_interval is not None and time_provider.get_time() - last_update >= progress_interval:
                last_update = time_provider.get_time()