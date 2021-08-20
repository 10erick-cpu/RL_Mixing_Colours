import time

import cv2

from utils.experiment_control.pumps.activa_a_22 import AdoxActivaA22
from utils.experiment_control.serial_comm.serial_devices import SerialDevice
from utils.experiment_control.setup_helpers import check_pump_assignments, list_ports


def check_video_devices():
    index = 1
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


def test_activate_pumps(ports):
    valid = []
    for name, port in ports:

        if name is None and "COM" not in port:
            print("Warning Invalid pump setting: ", (name, port))
            continue
        try:
            pump_1 = AdoxActivaA22(SerialDevice(port), print_changes=True)

            check_pump_assignments(pump_1, name)
            time.sleep(3)
            valid.append((name, port))
        except Exception as e:
            print("Pump activation failed", e)
    return valid


available_ports = list_ports()

print("Available serial ports")
print(available_ports)

print("Testing pump connection and assignment")
usable_ports = test_activate_pumps(available_ports)

print("Available video devices")
video_devices = check_video_devices()
print(video_devices)

print("Summary")
print("Detected pumps:")
for name, port in usable_ports:
    print(name, "@", port)

print()
print("Camera devices:")
for idx in video_devices:
    print("Index", idx)
