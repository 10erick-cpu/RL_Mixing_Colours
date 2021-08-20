import numpy as np
import seaborn as sns

from utils.helper_functions.misc_utils import sort_natural_keys

COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

# These are the "Tableau 20" colors as RGB.
tableau20 = np.asarray([[31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
                        [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150],
                        [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
                        [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199],
                        [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229]], dtype=np.float32)
tableau20 /= 255


class ColorIterator(object):
    def __init__(self, colors):
        self.colors = colors

        self.idx = None

    def reset(self):
        self.idx = None
        return self

    def get(self):
        if self.idx >= len(self.colors):
            raise ValueError(f"Out of colors idx {self.idx} of {len(self.colors)}")
        return self.colors[self.idx]

    def next(self):
        self.idx = self.idx + 1 if self.idx is not None else 0

        return self.get()

    def to_palette(self, hue_names):
        assert len(hue_names) == len(self.colors)
        return dict(zip(hue_names, self.colors))


class ColorGenerator(object):
    def __init__(self, palette_keys):
        self._keys = sorted(palette_keys, key=sort_natural_keys)
        self._palette = self.get_palette()

    def _get_colors(self, count):
        return sns.color_palette("husl", n_colors=count)

    def keys(self):
        return self._keys

    def __getitem__(self, item):
        return self._palette[item]

    def as_dict(self):
        return self._palette

    def get_palette(self):
        colors = self._get_colors(len(self._keys))
        assert len(self._keys) == len(colors)
        return dict(zip(self._keys, colors))
