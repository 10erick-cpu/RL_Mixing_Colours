import time

from utils.fluids.envs.real_env.real_exp_observer import CameraObserver


def view(camera_id=0, frame_delay=0.5):
    viewer = CameraObserver(camera_id)

    while True:
        viewer.debug_view(show_full_img=True, show_stats=True, show_raw_input=True)
        time.sleep(frame_delay)


if __name__ == '__main__':

    # for available camera ids see setup_checks.py "Available video devices", should be continuous integers, i.e. 0, 1, 2, ...
    view(camera_id=1)

