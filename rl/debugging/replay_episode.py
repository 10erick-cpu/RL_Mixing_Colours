import pickle

import cv2

from rl.envs.configurations import action_spaces
from utils.fluids.envs.real_env.real_exp_observer import CameraObserver
from utils.helper_functions.misc_utils import format_float_array_values

from utils.models.file_providers import FilteredFileProvider
from utils.models.folder import Folder


def file_sort_fn(filename):
    filename = filename[filename.rfind('/'):filename.rfind('.')]
    split = filename.split("step=")
    return int(split[1])


class EpisodePlayer(object):
    def __init__(self, observer):
        self.ms = 500
        self.observer = observer
        self.color = (255, 255, 0)

    def write_text(self, img, txt, size=0.7, v_offset=0):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (5, 30 + v_offset)
        fontScale = size
        fontColor = self.color
        lineType = 2

        cv2.putText(img, txt,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType, bottomLeftOrigin=False)

    def play(self, folder, action_array=None, show_img=False, ep_delay_s=0.5):
        with open(folder.get_file_path("episode_buffer"), 'rb') as f:
            ep_buffer = pickle.load(f)
        imgs = FilteredFileProvider(folder.path(), "png")
        imgs = sorted(imgs, key=file_sort_fn)

        measured_colors = []
        ep_colors = []
        distances = []
        goals = []

        for step, img in enumerate(imgs):
            s, a, n_s, r = ep_buffer[step]
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            target, _, _ = self.observer.get_target_area(img)
            measure = target.mean(axis=(0, 1))
            measured_colors.append(measure[0])
            measure_str = format_float_array_values(measure.tolist())
            ep_str = s.tolist()[-1]
            ep_colors.append(ep_str)
            distances.append(ep_str - s[0])
            goals.append(s[0])

            self.write_text(img, str(measure_str))
            self.write_text(img, "{0:.3f}".format(ep_str), v_offset=25)
            self.write_text(img, "g:{0:.3f}".format(s[0]), v_offset=45)
            self.write_text(img, str(format_float_array_values(s[1:3].tolist())), v_offset=65)
            if action_array is not None:
                if a < len(action_array):
                    channel, action = action_array[a]['channel'], action_array[a]['inf']
                    self.write_text(img, "c" + str(channel) + " | " + str(action), v_offset=85)

            if show_img:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("", img)
                cv2.waitKey(self.ms)

        import matplotlib.pyplot as plt
        plt.plot(measured_colors, color="r")
        plt.plot(ep_colors, color="k")
        plt.plot(distances, color="b")
        plt.plot(goals, color="g")
        plt.plot([0 for i in range(len(distances))], color="g")
        plt.ion()
        plt.pause(ep_delay_s)
        plt.cla()


if __name__ == '__main__':
    obs_storage = Folder("/mnt/unix_data/datastorage/experiment_experience/color_red_blue/obs_storage/")
    folders = obs_storage.get_folders()
    player = EpisodePlayer(CameraObserver(-1))
    aspaces = action_spaces.SimpleDiscrete()
    aspaces.pump_count = 2
    aspaces.build_action_space()
    actions = list(aspaces.action_dict.values())

    inspect = Folder("/mnt/unix_data/datastorage/experiment_experience/color_red_blue/obs_storage/time=20190802_1431,agent=future_medium_no_bn,ep=16")
    inspect=None
    if inspect:
        player.play(folder=inspect, action_array=actions, show_img=True)

    for folder in folders:
        folder = Folder(folder)
        player.play(folder=folder, action_array=actions, ep_delay_s=0.2)

