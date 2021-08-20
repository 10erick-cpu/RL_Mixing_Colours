from serial.tools import list_ports


class ValidationException(ValueError):
    def __init__(self, e):
        super(ValidationException, self).__init__(e)


class PumpInterface(object):
    def __init__(self, device, name=None):
        self.device = device
        self.name = name if name is not None else self.device.port

    def start_transfusion(self, ul_min):
        raise NotImplementedError("base")

    def stop_transfusion(self):
        raise NotImplementedError("base")

    def current_inf_lvl(self):
        raise NotImplementedError("base")


def find_usb_devices(name_filter=None):
    if name_filter:
        ports = [(p.description, p.device) for p in list_ports.grep(name_filter)]
    else:
        ports = [(p.description, p.device) for p in list_ports.comports(True)]
    print("{} devices found".format(len(ports)))
    return ports
