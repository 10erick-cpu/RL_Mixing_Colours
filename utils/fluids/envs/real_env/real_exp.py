import random
import time

from utils.experiment_control.schedule.repeat_strategy import Infinite
from utils.experiment_control.schedule.schedule_builder import ScheduleBuilder
from utils.fluids.envs.fluid_simulator import FluidSimulator
from utils.fluids.simulation_devices import MfdState, ColorMixTestDevice



class RealDevice(ColorMixTestDevice):
    pass


class RealDeviceState(MfdState):
    def __init__(self, observer):
        super().__init__(RealDevice(0))
        self.exp_observer = observer

    def update(self, engine, delta_time=1):
        pass

    def _mean_rgb(self, mat):
        return mat.mean(axis=(0, 1))

    def render(self, mode="channel_mean"):
        frame, target_area = self.exp_observer.capture_observation(draw_rect=False)

        if mode == "channel_mean":
            return self._mean_rgb(target_area)

        elif mode == "raw":
            return frame, target_area
        else:
            raise KeyError("unknown mode", mode)


class RealExperiment(FluidSimulator):

    def __init__(self, sim_config, pump_manager, action_handler, water_pump_port, observer):
        super().__init__(sim_config=sim_config)

        self.p_manager = pump_manager
        self.water_pump_port = water_pump_port
        self.device_state = RealDeviceState(observer)
        self.device = self.device_state.device
        self.action_handler = action_handler
        self.step_start = None

    def reset(self, reset_device=False, **kwargs):
        super(RealExperiment, self).reset(reset_device=False)

        # self.p_manager.channel_to_pump_mapping[1].pump.start_transfusion(300)
        pump_schedule = self.ctrl.controllers[self.water_pump_port].schedule
        water_pump = self.p_manager.pump_data[self.water_pump_port][0]

        reset_sched = ScheduleBuilder().start_transfuse(300).wait(2000).stop_transfuse().repeat(
            Infinite()).build()
        self.ctrl.register_pump(water_pump, reset_sched, pump_schedule.tp, allow_override=True)

        duration = kwargs.get("reset_duration", None)
        if duration is None:
            duration = random.randint(0, 8)

        if duration > 0:
            print(self.action_handler.actions)
            print(f"Real reset for {duration} seconds")
            start = self.tp.get_time()

            reset_start = self.device_state.render('channel_mean')
            print()
            while self.tp.get_time() - start < duration:
                self.step(delta_time=1)
                print(f"\rreset running, remaining: {round(start + duration - self.tp.get_time())}s", end="")
            print()

            print("Reset complete | before", reset_start, "now", self.device_state.render('channel_mean'))
        else:
            print("No real reset performed")

        self.ctrl.register_pump(water_pump, pump_schedule, pump_schedule.tp, allow_override=True)

    def step(self, output=False, delta_time=None):
        self.step_start = time.time()
        return super(RealExperiment, self).step(output, delta_time)

    def render(self, mode='real', draw_target_area=False):
        return super(RealExperiment, self).render(mode)
