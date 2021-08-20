import tensorflow as tf

# disable tf logging warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from rl.misc.seeds import Seeded

import multiprocessing

from utils.helper_functions.misc_utils import timestamp_now_str
from utils.models.dot_dict import DotDict
from rl.misc.monitor import CustomMonitor


def parallel_train_func(data):
    run_folder = data.run_folder
    print("start job id", data.cfg.short_name(), data.run_id, "tb log dir:", data.tb_log_dir)

    with Seeded(np=True, pytorch=True, tf=True, seed=data.run_id * 100) as s:
        monitor = CustomMonitor(env=data.cfg.build_env(data.env_init_cls),
                                data=data,
                                log_dir=run_folder,
                                log_f_name="train",
                                print_interval=data.num_episodes_per_log * data.env_max_steps,

                                eval_interval=data.num_episodes_per_eval * data.env_max_steps)

        model = data.cfg.get_model(monitor, tensorboard_log_dir=data.tb_log_dir.path() if data.tb_log_dir else None)

        model.learn(total_timesteps=data.num_episodes_to_learn * data.env_max_steps,
                    tb_log_name=f"{data.experiment_name}_{data.cfg.tensorboard_id()}_run={data.run_id}",
                    callback=monitor, seed=s.user_seed)

        monitor.eval_agent(model, data.num_episodes_to_learn * data.env_max_steps, run_folder, "val_result_final.csv")

    model.save(run_folder.get_file_path("final_model.pkl"))


class ExperimentRunner(object):
    def __init__(self, output_dir,
                 num_episodes_per_log=5,
                 num_episodes_per_eval=10,
                 env_max_steps=100,
                 runs_per_config=3,
                 tb_log_dir=None
                 ):
        self.output_dir = output_dir
        self.tb_log_dir = tb_log_dir
        self.runs_per_config = runs_per_config
        self.num_episodes_per_log = num_episodes_per_log
        self.num_episodes_per_eval = num_episodes_per_eval
        self.env_max_steps = env_max_steps

    def __build_jobs(self, experiment_name, configurations, env_init_cls, num_episodes_to_learn=100):

        exp_root_folder = self.output_dir.make_sub_folder(experiment_name)

        if self.tb_log_dir is None:
            #tb_log_dir = self.output_dir.make_sub_folder("tb_logs")
            tb_log_dir = None

        else:
            tb_log_dir = self.tb_log_dir


        jobs = []

        for cfg_no, cfg in enumerate(configurations):
            timestamp = timestamp_now_str()
            cfg_id = f"{timestamp}_{cfg.short_name()}"
            config_output_folder = exp_root_folder.make_sub_folder(cfg_id)
            cfg.save(config_output_folder.get_file_path("config.json"))

            for run_id in range(self.runs_per_config):
                data = DotDict()
                data.run_id_str = f"run_{run_id}"

                run_folder = config_output_folder.make_sub_folder(data.run_id_str)
                data.env_init_cls = env_init_cls
                data.cfg = cfg
                data.num_episodes_to_learn = num_episodes_to_learn
                data.num_episodes_per_log = self.num_episodes_per_log
                data.num_episodes_per_eval = self.num_episodes_per_eval
                data.env_max_steps = self.env_max_steps
                data.tb_log_dir = tb_log_dir
                data.run_folder = run_folder
                data.experiment_name = experiment_name
                data.run_id = run_id
                jobs.append(data)

        return jobs

    def execute(self, experiment_name, configurations, env_init_cls, num_episodes_to_learn=100, num_workers=4):

        jobs = self.__build_jobs(experiment_name, configurations, env_init_cls, num_episodes_to_learn)

        if not num_workers:
            for job in jobs:
                parallel_train_func(job)
        else:
            pool = multiprocessing.Pool(num_workers)
            pool.map(parallel_train_func, jobs)
            pool.close()
            pool.join()
            pool.terminate()
