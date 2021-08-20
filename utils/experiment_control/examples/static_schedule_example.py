import datetime
import logging
import time

from utils.experiment_control.control.experiment_executor import ExperimentExecutor
from utils.experiment_control.pumps.activa_a_22 import AdoxActivaA22
from utils.experiment_control.schedule.repeat_strategy import Infinite, Times
from utils.experiment_control.schedule.schedule_builder import ScheduleBuilder
from utils.experiment_control.serial_comm.serial_devices import SerialDevice
from utils.experiment_control.setup_helpers import list_ports, check_pump_assignments
from utils.fluids.time_providers import RealTime


logging.basicConfig(filename='static_experiment_log',
                    filemode='a',
                    format='%(asctime)s | %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logging.info("init")





# NO EDIT REQUIRED: Shows all available serial devices connected to the laptop
print("Available devices:")
print(list_ports())
print()

# specify the devices you want to use here (e.g. COM1, COM2 for Windows, /dev/ttyUSB0 for unix systems)
port_1 = "COM5"
port_2 = "COM6"

# NO EDIT REQUIRED: create the connection to the serial device / pump
pump_1 = AdoxActivaA22(SerialDevice(port_1), print_changes=True)
pump_2 = AdoxActivaA22(SerialDevice(port_2), print_changes=True)

# NO EDIT REQUIRED: Enable pumps for a short time one after another to ensure the ports are assigned correctly
check_pump_assignments(pump_1, "pump1")
time.sleep(3)
check_pump_assignments(pump_2, "pump2")

#########################################################################################################
#################################### CUSTOMIZATION PART #################################################
#########################################################################################################
# define your schedules here, possible options are:
# wait(seconds),
# start_transfuse(ug_min),
# stop_transfuse,
# repeat(Once() / Times(n) / Infinite())
#########################################################################################################
#########################################################################################################

# SCHEDULE A: start immediately, infuse 120ug/min for 1s, stop the pump after that and wait another second, repeat infinite times
schedule_A = ScheduleBuilder() \
    .wait(0) \
    .start_transfuse(120) \
    .wait(1) \
    .stop_transfuse() \
    .wait(1) \
    .repeat(Infinite()).build()

# SCHEDULE B: start immediately, infuse 100ug/min for 10s, increase infusion to 200 for additional 10s, increase even more to 300ug/min for
# another 10s, stop afterwards and wait one minute until repetition, repeated a total of 5 times
schedule_B = ScheduleBuilder() \
    .wait(0) \
    .start_transfuse(100) \
    .wait(10) \
    .start_transfuse(200) \
    .wait(10) \
    .start_transfuse(300) \
    .wait(10) \
    .stop_transfuse() \
    .wait(60) \
    .repeat(Times(5)) \
    .build()

#########################################################################################################
#########################################################################################################
#########################################################################################################

# NO EDIT REQUIRED: Create the experiment controller, it will execute the schedules assigned to each pump
experiment_controller = ExperimentExecutor()

# Assign a schedule to each of the pumps

# Let pump 1 execute schedule A
experiment_controller.register_pump(pump_1, schedule_A, RealTime())

# Let pump 2 execute schedule B
experiment_controller.register_pump(pump_2, schedule_B, RealTime())

# NO EDIT REQUIRED:
# Start the experiment, it will run until all schedules are finished or infinite if one of them has the repeat(Infinite()) option set
# update_per_second parameter defines how often per second the controller checks for a change in the schedule (change dose, stop, etc)
# for longer schedules, checking once per second should be fine

delay_hours = 0
delay_minutes = 0
delay_seconds = 10

experiment_countdown = datetime.timedelta(hours=delay_hours, minutes=delay_minutes, seconds=delay_seconds)
start_time = datetime.datetime.now().replace(microsecond=0) + experiment_countdown

current_time = datetime.datetime.now().replace(microsecond=0)

while start_time > current_time:
    delta = start_time - current_time

    print("\r", "Experiment will start in", str(delta), end="")
    time.sleep(1)
    current_time = datetime.datetime.now().replace(microsecond=0)
experiment_controller.run(update_per_second=1, progress_interval=None)
