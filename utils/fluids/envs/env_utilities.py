from utils.experiment_control.control.experiment_executor import ExperimentExecutor
from utils.experiment_control.schedule.repeat_strategy import Infinite, Once
from utils.experiment_control.schedule.schedule_builder import ScheduleBuilder
from utils.experiment_control.schedule.schedule_items import PumpScheduleItem
from utils.experiment_control.serial_comm.serial_devices import VirtualSerialDevice
from utils.experiment_control.simulation.virtual_pump import VirtualAdoxActivaA22
from utils.models.dot_dict import DotDict


class AgentProvidedSchedule(PumpScheduleItem):
    def __init__(self):
        super().__init__()
        self._transfuse = 0
        self.update_required = True

    def _is_run_required(self):
        return self.update_required

    def on_stop(self):
        pass

    def reset(self):
        self._transfuse = 0
        self.pump.stop_transfusion()
        self.update_required = True
        # print("Pump Reset")

    @property
    def transfuse(self):
        return self._transfuse

    @transfuse.setter
    def transfuse(self, val):
        # self.update_required = self._transfuse != val
        self.update_required = True

        self._transfuse = val
        # print("AgentCtrl: Set pump val %s" % val)

    # @Logged()
    def _run(self, **kwargs) -> bool:

        if self.transfuse == 0:

            self.pump.stop_transfusion()
            # print("agentschedule: _run stop", self.pump)
        else:

            self.pump.start_transfusion(self.transfuse)
            # print("AgentCtrl: Start transfusion %s" % self.transfuse, self.pump)
            # print("agentschedule: _run %f" % self.transfuse)
        self.update_required = False
        return True


def _init_pump(port):
    return VirtualAdoxActivaA22(VirtualSerialDevice(port))


class PumpManager(ExperimentExecutor):

    def __init__(self, pump_mapping, pump_init_fn=_init_pump):
        super().__init__()
        self.pump_data = DotDict()
        self.__p_internal = DotDict()
        self.channel_to_pump_mapping = DotDict()

        for p_name, p_mapping in pump_mapping.items():
            port = p_mapping['port']
            channel = p_mapping['channel']
            self.__p_internal[port] = DotDict()
            pump_dict = self.__p_internal[port]
            pump_dict.agent_ctrl = AgentProvidedSchedule()
            pump_dict.pump = pump_init_fn(port)
            pump_dict.pump_data = (pump_dict.pump, self.build_pump_schedule(pump_dict.agent_ctrl),
                                   channel, p_mapping['fluid_type'])
            self.pump_data[port] = pump_dict.pump_data
            self.channel_to_pump_mapping[channel] = self.__p_internal[port]

    def build_pump_schedule(self, agent_ctrl):
        return ScheduleBuilder().with_schedulable(agent_ctrl).repeat(Infinite()).build()

    def get_pump_states(self):
        state = []
        for i in range(len(self.channel_to_pump_mapping)):
            state.append(self.current_inf_val(i))

        return state

    def connect_device(self, mfd_device, time_provider):
        print("Connecting device to inlets")

        for pump, schedule, p_channel, fluid_type in self.pump_data.values():
            pump.device.receiver = mfd_device.configure_inlet(p_channel, fluid_type)
            self.register_pump(pump, schedule, time_provider)
        print()

    def pump_count(self):
        return len(self.__p_internal)

    def on_schedule_finished(self, ctrl_name, pump_ctrl):
        # super(PumpManager, self).on_schedule_finished(ctrl_name, pump_ctrl)
        # print("sched finished", pump_ctrl)
        pass

    def current_inf_val(self, channel_idx):
        return self.channel_to_pump_mapping[channel_idx].pump.curr_ul_min

    def apply_action(self, channel_idx, action):
        raise NotImplementedError()

    def update_channel(self, channel_idx, action, update_incremental=False):
        self.apply_action(channel_idx, action)
        return True

    def stop(self):

        for p in self.pump_data:
            p_data = self.pump_data[p]
            print("Stopping", p_data)
            p_data[0].stop_transfusion()


class DirectPumpManager(PumpManager):
    def apply_action(self, channel_idx, action):
        pump = self.channel_to_pump_mapping[channel_idx].pump
        pump.start_transfusion(action)


class IntervalPumpManager(PumpManager):

    def __init__(self, pump_mapping, schedule_on_time, pump_init_fn=_init_pump):
        self.schedule_on_time = schedule_on_time
        super().__init__(pump_mapping, pump_init_fn)

    def build_pump_schedule(self, agent_ctrl):
        return ScheduleBuilder().with_schedulable(agent_ctrl).wait(self.schedule_on_time).stop_transfuse().repeat(
            Once()).build()

    def on_schedule_finished(self, ctrl_name, pump_ctrl):
        # print("schedule finished", ctrl_name)
        pass

    def apply_action(self, channel_idx, action):
        schedule = self.channel_to_pump_mapping[channel_idx].pump_data[1]
        schedule.reset()
        self.channel_to_pump_mapping[channel_idx].agent_ctrl.transfuse = action

        # time_p = self.channel_to_pump_mapping[channel_idx].pump_data[1].tp
        # pump = self.channel_to_pump_mapping[channel_idx].pump
        # self.register_pump()


class IncrementalPumpManager(PumpManager):

    def apply_action(self, channel_idx, action):
        ctrl = self.channel_to_pump_mapping[channel_idx].agent_ctrl

        new_val = ctrl.transfuse + action
        if new_val < 0:
            new_val = 0
        if new_val > 300:
            new_val = 300
        ctrl.transfuse = new_val
        if self.channel_to_pump_mapping[channel_idx].agent_ctrl.update_required:
            pass


class ActionHandler(object):
    def __init__(self, actions_per_pump, is_incremental_actions=False):
        self.pumps = None
        self.pump_count = None
        self.actions_per_pump = actions_per_pump
        self.actions = None
        self.incremental = is_incremental_actions
        self.action_dict = {}

    def init(self, pump_manager: PumpManager):
        self.pumps = pump_manager
        self.pump_count = pump_manager.pump_count()
        self.actions = self.build_action_space()

    def build_action_space(self):
        actions = []
        action_idx = 0
        for channel in range(self.pump_count):

            for action in self.actions_per_pump:
                actions.append(action)
                self.action_dict[action_idx] = {'channel': channel, 'inf': action}
                action_idx += 1
        return actions

    def adjust_channel(self, channel, action):

        return self.pumps.update_channel(channel, action, self.incremental)

    def handle_action(self, action_idx):

        channel = self.action_dict[action_idx]['channel']
        inf_val = self.action_dict[action_idx]['inf']
        # print("ActionHandler: adjust channel %d - action_idx %d - inf_val %d" % (channel, action_idx, inf_val))
        return self.actions[action_idx], channel, self.adjust_channel(channel, inf_val)


class PredefinedActionSpace(ActionHandler):
    def __init__(self, action_dict, is_incremental_actions=False):
        super().__init__([], is_incremental_actions=is_incremental_actions)
        self.action_dict = action_dict

    def build_action_space(self):
        actions = []
        for action_idx in self.action_dict:
            channel = self.action_dict[action_idx]['channel']
            inf_val = self.action_dict[action_idx]['inf']
            actions.append(inf_val)
        return actions
