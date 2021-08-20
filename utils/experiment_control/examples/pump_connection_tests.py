from utils.fluids.time_providers import RealTime
from utils.pump_control.pump_interface import ExperimentController, AdoxActivaA22
from utils.pump_control.pump_utils import list_ports
from utils.pump_control.schedule.schedule_builder import ScheduleBuilder
from utils.pump_control.schedule import Infinite
from utils.pump_control.serial_comm.serial_devices import SerialDevice

print("Ports:")
print(list_ports())
print()

port_1 = "/dev/ttyUSB0"
port_2 = "/dev/ttyUSB1"
port_3 = "/dev/USB2"


def run_schedules(port_1=None, port_2=None, port_3=None):
    exp_ctrl = ExperimentController()

    if port_1:
        pump1 = AdoxActivaA22(SerialDevice(port_1))

        s_1 = ScheduleBuilder().wait(0).start_transfuse(120).wait(1).stop_transfuse().wait(1).repeat(
            Infinite()).build()

        exp_ctrl.register_pump(pump1, s_1, RealTime())

    if port_2:
        pump2 = AdoxActivaA22(SerialDevice(port_2))
        s_2 = ScheduleBuilder().wait(0).start_transfuse(150).wait(1).stop_transfuse().wait(1).repeat(
            Infinite()).build()
        exp_ctrl.register_pump(pump2, s_2, RealTime())

    if port_3:
        pump3 = AdoxActivaA22(SerialDevice(port_3))

        pump3 = AdoxActivaA22(SerialDevice(pump3))
        s_3 = ScheduleBuilder().wait(0).start_transfuse(150).wait(1).stop_transfuse().wait(1).repeat(
            Infinite()).build()
        exp_ctrl.register_pump(pump3, s_3, RealTime())


    exp_ctrl.run(20, progress_interval=None)


if __name__ == '__main__':
    run_schedules(port_1, port_2, port_3)
