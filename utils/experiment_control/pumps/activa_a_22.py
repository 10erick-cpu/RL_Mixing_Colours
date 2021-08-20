import math

from utils.experiment_control.pumps.pump_interface import PumpInterface, ValidationException


class AdoxActivaA22(PumpInterface):
    MAX_INF_LEVEL = 500
    _CMD_START = '<<J000R>\n'
    _CMD_STOP = '<<J000S>\n'
    _CMD_SET_INF_LVL = '<<J000F>\n'
    _CMD_SET_INF_LEVEL_TEMPL = "<<J000F{0:04d}.{1:s}>\n"

    def __init__(self, serial_device, print_changes=True):
        PumpInterface.__init__(self, serial_device)

        self.enabled = False
        self.curr_ul_min = 0
        self.print_changes = print_changes

    def __enable(self):
        self.device.send_data(self._CMD_START)
        self.enabled = True
        return self.enabled

    def __disable(self):
        self.device.send_data(self._CMD_STOP)
        self.enabled = False
        return not self.enabled

    def start_transfusion(self, ul_min):
        self.__set_transfusion_param(ul_min)
        self.__enable()

    def stop_transfusion(self):
        self.__disable()
        self.__set_transfusion_param(0)

    def __set_transfusion_param(self, level):
        if level < 0 or level > self.MAX_INF_LEVEL:
            print("WARNING: setting invalid inf level", level)
        cmd = self.infusion_val_to_cmd(level)
        self._validate_cmd(cmd, level)

        if self.print_changes:
            param_changed = level != self.curr_ul_min
            # TODO
            # if param_changed:
            #    data = " ".join([self.device.port, "changed from", str(self.curr_ul_min), "to", str(level), "ug/min"])

        self.curr_ul_min = level

        r = self.device.send_data(cmd, receive_data=True, receive_eol="F")
        print(r)
        return r

    def get_current_inf_param(self):
        inf_idx = 0
        data = self.device.send_data("<J000>", receive_data=True, receive_eol=">")
        while "F" not in data:
            data = self.device.send_data("<J000>", receive_data=True, receive_eol=">")
            inf_idx += 1
        print(data)
        print(inf_idx)
        inf_lvl = self.cmd_to_value(data)
        return inf_lvl

    def _validate_cmd(self, cmd, cmd_input_val):
        try:
            if cmd_input_val < 0 or cmd_input_val > self.MAX_INF_LEVEL:
                print("WARNING: setting invalid inf level", cmd_input_val)

            val = self.cmd_to_value(cmd)

            if not val == cmd_input_val:
                raise ValidationException(
                    "Validation failed for output cmd {} with input val {}".format(cmd, cmd_input_val))
            return val
        except Exception as e:
            print(e)
            raise ValidationException(
                "Validation failed for output cmd {} with input val {}".format(cmd, cmd_input_val))

    def infusion_val_to_cmd(self, val):
        # ul/min
        # <J000F0085.5000>
        if val < 0 or val > self.MAX_INF_LEVEL:
            # raise ValidationException("Illegal val {}, max is {}".format(val, self.MAX_INF_LEVEL))
            print("WARNING: setting invalid inf level", val)
        val = float(val)
        decimals, integer = math.modf(val)
        result = self._CMD_SET_INF_LEVEL_TEMPL.format(int(integer), "{0:.4f}".format(decimals).split(".")[1])
        self._validate_cmd(result, val)
        return result

    @staticmethod
    def cmd_to_value(cmd):
        return float(cmd.replace(">", "").split('F')[1][:-1])

    def current_inf_lvl(self):
        return self.curr_ul_min
