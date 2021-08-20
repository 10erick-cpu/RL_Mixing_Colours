import datetime
import re
import time

import matplotlib.pyplot as plt
import numpy as np

ENABLE_TIMED_EXECUTION_LOGS = True

IS_NOTEBOOK = False


def atoi(text):
    return int(text) if text.isdigit() else text


def sort_natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)] if isinstance(text, str) else text


def draw_interactive_plot(fig=None, fps=30):
    if not IS_NOTEBOOK:
        plt.ion()
        plt.draw()
        plt.pause(1 / fps)
    else:
        plt.show()


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def smooth_dataframe(data, window=5):
    return [np.asarray(data[max(0, i - window):i + 1]).mean() for i in range(len(data))]


def progress_string(percentage, num_chars=10):
    percentage = percentage * 100
    return ">" * int(np.floor(percentage / num_chars)) + "-" * int(np.ceil((100 - percentage) / num_chars))


def format_float_array_values(array):
    return ["{0:.3f}".format(x) for x in array]


def fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__class__.__name__


def timed_execution(func):
    def wrapper(*args, **kwargs):
        if not ENABLE_TIMED_EXECUTION_LOGS:
            return func(*args, **kwargs)

        start = time.time()
        ret = func(*args, **kwargs)
        dur = time.time() - start

        print("\n", "Func", func.__name__, "took", "{0:.5f}".format(dur * 1000), "ms")

        return ret

    return wrapper


def flatten_dataframe_index(df):
    def join_cols(inp):
        if len(inp[1]) == 0:
            return inp[0]
        return "_".join(inp)

    df.columns = df.columns.map(join_cols)
    return df


def enumerate_with_step(xs, start=0, step=1):
    for x in xs:
        yield (start, x)
        start += step


def timestamp_now_str():
    return timestamp_to_str(time.time(), include_micro=False)


def timestamp_to_str(ts, include_micro=False, max_reduce=True):
    if include_micro:
        return datetime.datetime.fromtimestamp(ts).strftime('%m-%d %H:%M:%S.%f')

    if max_reduce:
        return datetime.datetime.fromtimestamp(ts).strftime('%m%d_%H%M%S')

    return datetime.datetime.fromtimestamp(ts).strftime('%m-%d %H:%M:%S')


class MultiRunningMean(object):
    def __init__(self):
        self._mean_dict = dict()
        self._last_dict = dict()

    def update(self, key, val):
        self._last_dict[key] = val
        if key not in self._mean_dict:
            self._mean_dict[key] = val
        else:
            self._mean_dict[key] = (self._mean_dict[key] + val) / 2

    def reset(self):
        self._mean_dict = dict()
        self._last_dict = dict()

    def keys(self):
        return sorted(list(self._mean_dict.keys()))

    def value(self, key, strict=False):
        if key not in self._mean_dict:
            return 0 if not strict else None
        return self._mean_dict[key]

    def last(self, key):
        return self._last_dict[key]


class RunningMean(object):
    def __init__(self, id=None):
        self._running_mean = None
        self.id = id

    def update(self, val):
        if not self._running_mean:
            self._running_mean = val
        else:
            self._running_mean = (self._running_mean + val) / 2

    def reset(self):
        self._running_mean = None

    def value(self):
        if self._running_mean is None:
            return 0
        return self._running_mean
