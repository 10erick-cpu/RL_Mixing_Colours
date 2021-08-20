import numpy as np

from utils.fluids.device_components import Inlet, ClippingOutlet


def mm_to_um(mm):
    return int(mm * 1000)


class Fluid(object):

    def calc_effectivity(self, curr_saturation, saturation_threshold):
        x = min(curr_saturation / saturation_threshold, 1)
        assert 0 <= x <= 1
        return np.e ** -(3 * x ** 3)

    def __init__(self, color_dist, f_type, saturation_window=None, saturation_inf_sum=240):
        self.particles = np.asarray(color_dist)
        self.type = f_type
        self.saturation_window = saturation_window
        self.max_inf = saturation_inf_sum / 60

    def intensity_update(self, infusion_history):
        # print(self, len(infusion_history), "total", sum(infusion_history))

        eff = self.effectivity(
            sum(infusion_history[-self.saturation_window:])) if self.saturation_window is not None else 1
        # print("effectivity", eff)
        return self.particles * eff

    def effectivity(self, sum_history):
        if self.saturation_window is None:
            return 1
        max_damp = self.max_inf * self.saturation_window // 3

        # print("% damp", x, "sum", curr_damp, "max", max_damp, "damp window", damp_window)

        return self.calc_effectivity(sum_history, max_damp)

    def __repr__(self):
        return f"Fluid | type: {self.type} | dist: {self.particles.tolist()}"


class Water(Fluid):
    def __init__(self):
        super(Water, self).__init__([.0, 0, 0, 1], 'water')


class MicroFluidicDevice(object):
    PIX_PER_MM = 10

    def __init__(self, height_mm, width_mm, depth_mm):
        self.height = height_mm
        self.width = width_mm
        self.depth = depth_mm
        self.inlets = dict()
        self.outlet = None
        self.max_volume = 1
        self.water_channel_id = 3

    def mm_to_pix(self, val):
        return int(val * self.PIX_PER_MM)

    def mm_to_um(self, val):
        return val * 1e3

    def height_pix(self):
        return self.height * self.PIX_PER_MM

    def width_pix(self):
        return self.width * self.PIX_PER_MM

    def total_vol_pix(self):
        return self.width_pix() * self.height_pix() * self.depth_pix()

    def vol_per_pix(self, num_channels=1):
        return self.total_vol_pix() / (self.width_pix() * self.height_pix()) / num_channels

    def depth_pix(self):
        return self.depth
        # return self.mm_to_um(self.depth / (self.PIX_PER_MM ** 2))

    def configure_inlet(self, channel, fluid_type):
        raise NotImplementedError("base")

    def configure_outlet(self, ug_min):
        raise NotImplementedError("base")


class ColorMixTestDevice(MicroFluidicDevice):
    def __init__(self, outlet_ug_min=0):
        super().__init__(11, 22, 0.029 / 2)
        self.max_volume = 60
        self.outlet = self.configure_outlet(outlet_ug_min)
        self.water_channel_id = 3

    def configure_inlet(self, channel, fluid: Fluid):
        inlet = Inlet(fluid)
        inlet.height = self.mm_to_pix(0.2)
        inlet.width = self.mm_to_pix(0.2)
        inlet.x = 0
        inlet.y = self.height_pix() // 2 - inlet.height // 2
        inlet.init(local_center=(0, int(inlet.height // 2)))
        self.inlets[channel] = inlet

        print(str(self)[-41:], ": Configure inlet | color", fluid.particles, "| channel:", channel, "| type:",
              fluid.type)
        # traceback.print_stack(limit=4, file=sys.stdout)
        return self.inlets[channel]

    def configure_outlet(self, ug_min):
        outlet = ClippingOutlet()
        outlet.levitation = self.depth_pix()
        outlet.height = self.mm_to_pix(1)
        outlet.width = self.mm_to_pix(3)
        outlet.x = self.width_pix()
        outlet.y = self.height_pix() // 2 - outlet.height // 2
        outlet.ug_min = ug_min
        outlet.init(local_center=(outlet.width - 1, int(outlet.height // 2)))
        return outlet


class ColorMixer(object):
    @staticmethod
    def from_matrix(M):
        try:
            c = ColorMixer(0, 0)
            c.M = M
            c.M_inv = np.linalg.inv(M)
            return c
        except Exception as e:
            print("Failed to set M:", e)
            return None

    def __init__(self, m=None, absorbance=1, reflection=-0.2):
        self.M = None
        self.M_inv = None

        if m is None:
            m = np.asarray([[reflection, absorbance, absorbance],
                            [absorbance, reflection, absorbance],
                            [absorbance, absorbance, reflection]], dtype=np.float32)
        self.set_m(m)

    def set_m(self, m):
        self.M = m
        self.M_inv = np.linalg.inv(self.M)

    def forward(self, distribution, intensity):

        dist = (distribution.dot(self.M))
        return intensity - intensity * dist

    def backward(self, rgb_color, src_intensity):
        # TODO: negative values if rgb values too far from another
        rest_dist = (src_intensity - rgb_color) / src_intensity
        result = self.M_inv.dot(rest_dist)
        valid = not (result < 0).any()
        return result, valid


class MfdState(object):
    def __init__(self, device: MicroFluidicDevice, base_intensity=180):
        self.device = device
        self.grid = None
        self.step_count = None
        self.mixer = None
        self.base_intensity = base_intensity

        m = np.asarray([[-0.0984, 0.7386, 0.5415],
                        [.36, -.2, .6000],
                        [0.6502, 0.3318, -0.1884]])

        m = np.asarray([
            [-0.3, 1., 1.],
            [1., -.3, 1.],
            [1., 1., -0.3]
        ])

        self.set_color_mixer(ColorMixer(m=m))

    def set_color_mixer(self, mixer: ColorMixer):
        print("Set color mixer", mixer.M)
        self.mixer = mixer

    def num_channels(self, include_water=True):
        chans = 4

        if self.device.water_channel_id and not include_water:
            chans -= 1
        return chans

    def _generic_reset(self):
        for inl_key, inl in self.device.inlets.items():
            inl.reset()

    def reset_fixed(self, val=0):
        self._generic_reset()
        num_channels = self.num_channels(include_water=True)
        self.step_count = 0
        if num_channels <= 0:
            raise ValueError("No inlet provided")

        val = ((val / 255) * self.device.vol_per_pix(self.num_channels(include_water=False)))

        self.grid = np.full((self.device.height_pix(), self.device.width_pix(), num_channels),
                            fill_value=val,
                            dtype=np.float32)

        self.grid[:, :, self.device.water_channel_id] = self.device.vol_per_pix() - \
                                                        self.grid[:, :, :self.device.water_channel_id].sum(
                                                            axis=2)

    def reset_fixed_channel_wise(self, val):
        self._generic_reset()
        num_channels = self.num_channels(include_water=True)
        self.step_count = 0
        if num_channels <= 0:
            raise ValueError("No inlet provided")

        assert len(val) == num_channels - 1

        alphas, valid = self.mixer.backward(val, self.base_intensity)
        if not valid:
            alphas, valid = self.mixer.backward(self.mixer.forward(alphas, self.base_intensity), self.base_intensity)

        if not valid:
            alphas = np.clip(alphas, a_min=0, a_max=None)

        alphas *= self.device.vol_per_pix()

        self.grid = np.zeros((self.device.height_pix(), self.device.width_pix(), num_channels),
                             dtype=np.float32)

        water_fill = max(self.device.vol_per_pix() - alphas[:self.device.water_channel_id].sum(), 0)

        self.grid[:, :, :self.device.water_channel_id] = alphas
        self.grid[:, :, self.device.water_channel_id] = water_fill

    def reset_random(self, min_val=0, max_val=250):
        self._generic_reset()
        num_channels = self.num_channels(include_water=True)
        self.step_count = 0
        if num_channels <= 0:
            raise ValueError("No inlet provided")

        shape = (self.device.height_pix(), self.device.width_pix(), num_channels)

        self.grid = np.random.uniform(min_val, max_val, shape)

    def update(self, engine, delta_time=1):

        for inl_key, inl in self.device.inlets.items():
            inl.update(self.grid, delta_time=delta_time)

        self.grid = engine.update(self.grid,
                                  lim=self.device.outlet.levitation,
                                  max_iter=1)

        if self.device.outlet:
            self.device.outlet.update(self.grid, delta_time)

        self.step_count += delta_time

    def intensity_noise_rgb(self, fill_values, rgb_noise_scalar=0.75):
        noise = np.random.normal(0, 0.2, size=fill_values.shape)

        noise *= rgb_noise_scalar
        return fill_values + noise

    def compute_intensity(self, fluid_dist, intensity):
        # print(fluid_dist)

        result = self.mixer.forward(fluid_dist[:self.device.water_channel_id], intensity)
        # print("alphas", fluid_dist, "result", result)
        return self.intensity_noise_rgb(result)

    def render_intensity_values(self):

        alphas = self.grid[:, :, :].mean(axis=(0, 1))
        # alphas /= alphas.sum()
        alphas /= (self.device.vol_per_pix())

        return self.compute_intensity(alphas, self.base_intensity)

    def get_inf_history(self, window=60):
        result = dict()
        for inl_key, inl in self.device.inlets.items():
            result[inl_key] = inl.infusion_history[-window:]
        return result

    def render(self, mode="real"):
        if mode == "real":
            mat = self.render_intensity_values()

            # if num_channels == 1:
            # mat = gray2rgb(mat)
            return mat.round().astype(np.uint8)
        if mode == "channel_mean":
            # result = (self.grid.mean(axis=(0, 1)) / self.device.vol_per_pix(self.num_channels())) * 255
            # result = self.render_intensity_values().mean(axis=(0, 1))

            result = self.render_intensity_values()
            return result
        raise ValueError("Unknown mode to render", mode)
