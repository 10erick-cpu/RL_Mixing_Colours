from utils.experiment_control.pumps.activa_a_22 import AdoxActivaA22


class SimulatedEnvironmentCmdReceiver(object):
    def __init__(self):
        self.conn_established = False

    def on_open(self, port):
        self.conn_established = True

    def on_close(self, port):
        self.conn_established = False

    def decode_data(self, port, raw_data):
        raise NotImplementedError("base")

    def on_handle_command(self, port, decoded_data):
        raise NotImplementedError("base")

    def on_write(self, port, bits):
        self.on_handle_command(port, self.decode_data(port, bits))


class VirtualAdoxActivaA22(AdoxActivaA22):

    def __init__(self, serial_device):
        super().__init__(serial_device)

    def __repr__(self):
        return "VirtualAdoxActivaA22@" + self.device.port

    class CommandReceiver(SimulatedEnvironmentCmdReceiver):
        def __init__(self):
            super().__init__()
            self.handlers = {VirtualAdoxActivaA22._CMD_START: self.handle_start_cmd,
                             VirtualAdoxActivaA22._CMD_STOP: self.handle_stop_cmd,
                             VirtualAdoxActivaA22._CMD_SET_INF_LVL: self.handle_set_inf_lvl_cmd}

        def decode_data(self, port, raw_data):
            return raw_data.decode("utf-8")

        @staticmethod
        def decode_inf_cmd(set_inf_lvl_cmd):
            return AdoxActivaA22.cmd_to_value(set_inf_lvl_cmd)

        def handle_start_cmd(self, port, cmd):
            pass

        def handle_stop_cmd(self, port, cmd):
            pass

        def handle_set_inf_lvl_cmd(self, port, inf_lvl):
            pass

        def register_handler(self, cmd, handler):
            self.handlers[cmd] = handler

        def on_handle_command(self, port, data):

            if len(self.handlers) == 0:
                print(data)
                return

            for handler_key, handler in self.handlers.items():
                if handler_key in data:
                    handler(port, data)
                    return

            raise ValueError("No handler for port/data", port, data)

        @staticmethod
        def default():
            return VirtualAdoxActivaA22.CommandReceiver()
