import threading
from queue import Queue

import cv2
import matplotlib.pyplot as plt
import numpy as np


class CameraObserver(object):
    def __init__(self, target_camera_id=0, output_scale=0.5, flip_image=True):
        super().__init__()
        self.camera_id = target_camera_id
        self.obs_window_h = 150
        self.obs_window_w = 150
        self.frame_scale = output_scale
        self.capture_device = None
        self.do_flip = flip_image
        self.measurements = np.zeros((100, 3), dtype=np.float32)
        self.measurement_idx = 0

    def num_observations(self):
        return None

    def write_text(self, img, txt):

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (5, 30)
        fontScale = 1
        fontColor = (0, 0, 255)
        lineType = 3

        cv2.putText(img, txt,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType, bottomLeftOrigin=False)

    def capture_frame(self, gray=False):
        if not self.capture_device:
            self.capture_device = VideoCapture(self.camera_id)

        device = self.capture_device

        frame = device.read()
        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.do_flip:
            # frame = np.flip(frame, axis=(0, 1))
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)
            pass

        return frame

    def get_target_rect(self, frame):
        obs_height = self.obs_window_w
        obs_width = self.obs_window_h

        w, h, c = frame.shape

        rect_lo = (h // 2 - obs_height // 2, w // 2 - obs_width // 2)
        rect_hi = (h // 2 + obs_height // 2, w // 2 + obs_width // 2)
        return rect_lo, rect_hi

    def get_target_area(self, frame):

        rect_lo, rect_hi = self.get_target_rect(frame)

        thickness = 1

        target_area = frame[rect_lo[1] + thickness:rect_hi[1] + thickness - 1,
                      rect_lo[0] + thickness:rect_hi[0] + thickness - 1]
        return target_area, rect_lo, rect_hi

    def capture_observation(self, draw_rect=False):
        frame = self.capture_frame()

        target_area, rect_lo, rect_hi = self.get_target_area(frame)

        if draw_rect:
            cv2.rectangle(frame, rect_lo, rect_hi, (255, 0, 0))

        return frame, target_area

    def view(self):
        import cv2

        cap = cv2.VideoCapture(self.camera_id)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = frame

            # Display the resulting frame
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def measure(self, frame):
        means = frame.mean(axis=(0, 1))
        self.measurements[self.measurement_idx, :] = means
        self.measurement_idx += 1
        if self.measurement_idx >= len(self.measurements):
            self.measurement_idx = 0

        return means[0], means[1], means[2]

    def debug_view(self, show_full_img=False, show_stats=True, show_raw_input=False):

        plt.ion()
        if show_stats:
            f, ax = plt.subplots(1, 2)
        while True:

            frame, area = self.capture_observation(draw_rect=True)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r, g, b = self.measure(area)
            if show_stats:
                ax[1].cla()

                ax[1].plot(self.measurements[:, 0], color="r", label="{0:.2f}".format(r))
                ax[1].plot(self.measurements[:, 1], color="g", label="{0:.2f}".format(g))
                ax[1].plot(self.measurements[:, 2], color="b", label="{0:.2f}".format(b))
                ax[1].axvline(x=self.measurement_idx - 1)
                ax[1].legend()

            if show_full_img is not None:
                if show_full_img:
                    ax[0].imshow(frame, interpolation=None)
                else:
                    ax[0].imshow(area, interpolation=None)
            else:
                mat = np.zeros((50, 50, 3), dtype=np.float)
                mat[:, :] = [r, g, b]
                # ax[0].imshow(mat.round().astype(np.uint8), interpolation=None)

            plt.pause(0.000001)
            if show_raw_input:
                # obs.view()
                vals = ["{0:.3f}".format(x) for x in [r, g, b]]
                self.write_text(frame, str(vals))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("frame", frame)
                k = cv2.waitKey(1)


class VideoCapture:

    def __init__(self, name, adapt_params=True):
        self.cap = cv2.VideoCapture(name)
        if adapt_params:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)

            self.cap.set(cv2.CAP_PROP_CONTRAST, 50)
            self.cap.set(cv2.CAP_PROP_BACKLIGHT, 0)
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)
            self.cap.set(cv2.CAP_PROP_SATURATION, 50)
            self.cap.set(cv2.CAP_PROP_GAIN, 0)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 0)
            self.cap.set(cv2.CAP_PROP_GAMMA, 0)
        self.q = Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            while not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except Queue.Empty:
                    pass
            self.q.put(frame)
            # cv2.imshow("", frame)
            # cv2.waitKey(1)

    def read(self):
        return self.q.get()
