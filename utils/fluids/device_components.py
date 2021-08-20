from collections import deque

import numpy as np

from utils.experiment_control.simulation.virtual_pump import VirtualAdoxActivaA22
from utils.helper_functions.mat_utils import calc_inverse_normed_distance_map



class Component(object):
    def __init__(self):
        self._width = None
        self._height = None
        self._x = None
        self._y = None

    @staticmethod
    def create(x, y, width, height):
        c = Component()
        c.x = x
        c.y = y
        c.width = width
        c.height = height
        return c

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value


class Outlet(Component):
    def __init__(self):
        super().__init__()
        self.levitation = None
        self.ug_min = None
        self.ug_s = None
        self.drain_pixels_per_second = None
        self.dist_map = None

    @staticmethod
    def from_config(config):
        self = Outlet()
        self.config = config
        self.position = config.OUTLET['position']
        self.shape = config.OUTLET['shape']
        self.levitation = config.INLET['levitation']
        self.ug_min = config.OUTLET['ug_min']
        self.ug_s = self.ug_min / 60
        self.drain_pixels_per_second = self.ug_s
        self.init(local_center=config.OUTLET.local_center, num_channels=config.NUM_CHANNELS)
        return self

    def init(self, local_center=None):

        if local_center is not None:
            index = (local_center[1], local_center[0])
        else:
            index = None
        self.dist_map = calc_inverse_normed_distance_map(self.height, self.width, index=index)

        # self.dist_map = softmax(self.dist_map)
        # TODO: back to softmax if non-clipping outlet is used
        self.dist_map = self.dist_map / self.dist_map.sum()

        self.ug_s = self.ug_min / 60
        self.drain_pixels_per_second = self.ug_s

    def calc_inf_update(self, region, drain_val):
        assert region.shape == self.dist_map.shape

        update = np.multiply(self.dist_map, drain_val)
        return update

    def _apply_update(self, target_area, update_mat, levitation=None):
        if levitation:
            selector = target_area > levitation
            target_area[selector] -= update_mat[selector]
        else:
            target_area -= update_mat
        target_area[target_area < 0] = 0

    def update(self, well_mat, delta_time=1):
        target_area = well_mat[self.y:self.y + self.height, self.x - self.width: self.x, :]

        h, w, c = target_area.shape

        drain_val = self.drain_pixels_per_second * delta_time
        # evenly spread drain val
        drain_val = drain_val / target_area.shape[2]

        drain_distr = np.random.uniform(0, .7, size=c)

        drain_distr = drain_distr / drain_distr.sum()
        drain_distr = drain_distr * drain_val

        for i in range(c):
            up = self.calc_inf_update(target_area[:, :, i], drain_distr[i])

            # if isinstance(target_area, torch.Tensor):
            #    up = torch.from_numpy(up).to("cuda:0")

            self._apply_update(target_area[:, :, i], up, self.levitation)


class ClippingOutlet(Outlet):
    def _apply_update(self, target_area, update_mat, levitation=None):
        if not levitation:
            raise ValueError("ClippingOutlet without levitation")
        selector = target_area > levitation
        drained = target_area[selector]
        if drained > 0:
            print("Clipping outlet drained", drained)

        target_area[selector] = levitation

    def update(self, well_mat, delta_time=1):
        try:

            target_area = well_mat[self.y:self.y + self.height, self.x - self.width: self.x]
            target_area = well_mat
            h, w, c = target_area.shape

            drain_val = self.drain_pixels_per_second * delta_time
            # evenly spread drain val

            curr_pix_volume = target_area.sum(axis=2, keepdims=True)
            delta_vol = self.levitation - curr_pix_volume.squeeze()

            drain_targets = target_area[delta_vol < 0]

            if len(drain_targets) == 0:
                return False

            liquid_dist = target_area / curr_pix_volume

            # liquid_dist[:, :] += liquid_dist[:, :, 3][:, :, None] / 4
            # liquid_dist[:, :, 3] /= 4

            # liquid_dist /=liquid_dist.sum(axis=2, keepdims=True)
            # liquid_dist = liquid_dist
            # print("ldist", liquid_dist[0,0])
            drain_targets = drain_targets + (delta_vol[:, :, None] * liquid_dist * 1)[delta_vol < 0]
            # print(drain_targets)
            target_area[delta_vol < 0] = drain_targets
        except Exception as e:

            print(e)
            raise e


class Inlet(VirtualAdoxActivaA22.CommandReceiver, Component):
    def __init__(self, fluid):
        VirtualAdoxActivaA22.CommandReceiver.__init__(self)
        Component.__init__(self)
        self.enabled = False
        self.levitation = 0
        self.dist_map = None
        self._inf_per_s = 0
        self.fluid = fluid
        self.delay_q = deque()
        self.inlet_delay = 0
        self.infusion_history = []

    @property
    def inf_per_s(self):
        return self._inf_per_s

    @inf_per_s.setter
    def inf_per_s(self, val):
        if val != self.inf_per_s:
            # print("set inf_per_s, old", self.inf_per_s, "new", val, self.fluid_type, self)
            pass
        self._inf_per_s = val

    def init(self, local_center=None):

        if local_center is not None:
            index = (local_center[1], local_center[0])
        else:
            index = None
        self.dist_map = calc_inverse_normed_distance_map(self.height, self.width, index=index)
        dist_map_score = self.dist_map.sum()
        # self.dist_map = softmax2d(self.dist_map)
        self.dist_map = self.dist_map / self.dist_map.sum()

    def handle_set_inf_lvl_cmd(self, port, inf_lvl):
        target_inf_lvl = self.decode_inf_cmd(inf_lvl)

        # ug/min -> ug/s
        self.inf_per_s = target_inf_lvl / 60

        # logging.debug(" ".join(["t: ", self.config.TIME_PROVIDER.get_human_time_str(), "| SimPump@", port, "inf lvl",
        #                        str(self.pump_data[port].inf_lvl)]))

    def handle_start_cmd(self, port, cmd):
        # logging.debug(" ".join(["t: ", self.config.TIME_PROVIDER.get_human_time_str(), "| SimPump@", port, "Start"]))
        self.enabled = True

    def handle_stop_cmd(self, port, cmd):
        # logging.debug(" ".join(["t: ", self.config.TIME_PROVIDER.get_human_time_str(), "| SimPump@", port, "Stop"]))
        self.enabled = False
        # self.pump_data[port].inf_lvl = 0

    def calc_inf_update(self, region, inf_lvl):
        assert region.shape[:2] == self.dist_map.shape
        # update = np.zeros_like(region)
        # update[self.dist_map < 0.4] = np.multiply(self.dist_map[self.dist_map < 0.7], inf_lvl)
        # update[self.dist_map >= 0.4] = inf_lvl

        update = np.multiply(self.dist_map[:, :, None], inf_lvl)

        return update

    def reset(self):
        self.infusion_history.clear()

    def correct_update_by_height(self, update_mat):
        if self.levitation == 0:
            return update_mat
        update_mat -= self.levitation
        update_mat[update_mat < 0] = 0
        return update_mat

    def update(self, ui, delta_time=1):
        self.delay_q.append((self.enabled, self.inf_per_s, delta_time))

        if len(self.delay_q) >= self.inlet_delay * delta_time:
            self._update_step(ui)

    def _update_step(self, ui):

        enabled, inf, delta_time = self.delay_q.popleft()

        assert len(self.delay_q) == 0

        if not enabled and inf != 0:
            print("Warning: Pump not enabled but infusion value not 0")
            return
        scale = 1

        inf_lvl = inf * delta_time * scale

        # ui[self.y, self.x] += inf_lvl

        target_area = ui[self.y:self.y + self.height, self.x: self.x + self.width]

        update_val = self.calc_inf_update(target_area, inf_lvl)

        update_val = self.correct_update_by_height(update_val)

        self.infusion_history.append(inf_lvl)
        infusion = update_val * self.fluid.intensity_update(self.infusion_history)

        target_area += infusion
