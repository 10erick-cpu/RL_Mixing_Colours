import time


def check_pump_assignments(pump, name):
    print("Pump", name, "@ port", pump.device.port, "will activate")
    pump.start_transfusion(50)
    time.sleep(5)
    pump.stop_transfusion()


def list_ports():
    import serial.tools.list_ports

    ports = serial.tools.list_ports.comports()

    return [(port.name, port.device) for port in ports]
