import multiprocessing
import os

from rl.envs.env_initializers import EnvBuilder
from rl.misc.model_params import DQNParams, PPOv1Params
from rl.visualization.experiment_runner import ExperimentRunner
from rl.visualization.results_viewer import ResultsViewer, ExperimentViewer
from utils.models.dot_dict import DotDict
from utils.models.folder import Folder


class BaselineParamsHpSearch:
    params = DotDict()
    params.batch_size = 32
    params.layers = [64, 64]
    params.lr = 5e-4
    params.dueling = False
    params.layer_norm = False

    params.target_network_updates = 5 * 100

    @classmethod
    def to_kwargs(cls):
        return cls.params


class BaselineParamsAlgComparison:
    params = DotDict()
    params.batch_size = 256
    params.layers = [64, 128]
    params.lr = 0.001
    params.buffer_size = 50 * 1000
    params.dueling = False
    params.layer_norm = False

    params.target_network_updates = 8 * 100

    @classmethod
    def to_kwargs(cls):
        return cls.params


def get_dqn_baseline(target_param, baseline_params):
    bl = DQNParams(**baseline_params.to_kwargs())

    default_param = bl[target_param]
    name = f"BL={default_param}"
    bl.description = name
    return bl


def get_ppo_baseline(target_param, baseline_params):
    bl = PPOv1Params(**baseline_params.to_kwargs())

    default_param = bl[target_param]
    name = f"BL={default_param}"
    bl.description = name
    return bl


class ExperimentResultLogger(object):

    def __init__(self, base_folder: Folder, enable_tb_logs=False):
        self.base_folder = base_folder

        self.runner_setup = {'num_episodes_per_eval': 20, 'runs_per_config': 1}


        self.runner = ExperimentRunner(self.base_folder, **self.runner_setup,
                                       tb_log_dir=self.base_folder.make_sub_folder("tb_logs") if enable_tb_logs else None)
        self.results = ResultsViewer(self.base_folder)

    def run(self, experiment, configs, environment_init, num_episodes=100, num_workers=multiprocessing.cpu_count()):

        self.runner.execute(experiment, configs, environment_init, num_episodes_to_learn=num_episodes,
                            num_workers=num_workers)
        return self.results


class RunnableExperiment(object):
    def __init__(self, result_logger: ExperimentResultLogger):
        self.logger = result_logger

    @classmethod
    def get(cls):
        raise NotImplementedError()

    def run(self, env_init_cls, num_episodes=100):
        exp_name = f"{self.__class__.__name__}@{env_init_cls.__class__.__name__}"
        print("Run experiment", exp_name)
        return self.logger.run(exp_name, self.get(), env_init_cls, num_episodes)

    def view(self, target_env_cls, show=True):
        exp_name = f"{self.__class__.__name__}@{target_env_cls.__class__.__name__}"
        return self.logger.results.view_experiment(exp_name)


class DQN:

    base_dir = Folder(f"{os.path.dirname(__file__)}/experiments/dqn_experiments")
    exp_result_logger = ExperimentResultLogger(base_folder=base_dir)

    class Experiments:
        class LearningRate(RunnableExperiment):

            @classmethod
            def logger(cls):
                return cls(DQN.exp_result_logger)

            @classmethod
            def get(cls):
                default = get_dqn_baseline("lr", BaselineParamsHpSearch)

                rates = [0.01, 0.001, 0.002, 0.005, 0.0001]

                configs = [default] + [DQNParams(description=f"lr={bsize}", lr=bsize) for bsize in rates]
                return configs

        class Layers(RunnableExperiment):
            @classmethod
            def logger(cls):
                return cls(DQN.exp_result_logger)

            @classmethod
            def get(cls):
                default = get_dqn_baseline("layers", BaselineParamsHpSearch)

                layers = [[32, 32], [32, 64], [128, 128], [64, 128], [128, 64]]

                configs = [default] + [DQNParams(description=f"layers={ly}", layers=ly) for ly in layers]
                return configs

        class BatchSize(RunnableExperiment):
            @classmethod
            def logger(cls):
                return cls(DQN.exp_result_logger)

            @classmethod
            def get(cls):
                default = get_dqn_baseline("batch_size", BaselineParamsHpSearch)

                batch_sizes = [64, 128, 256, 512]

                configs = [default] + [DQNParams(description=f"bsize={bsize}", batch_size=bsize) for bsize in
                                       batch_sizes]
                return configs

        class BufferSize(RunnableExperiment):
            @classmethod
            def logger(cls):
                return cls(DQN.exp_result_logger)

            @classmethod
            def get(cls):
                default = get_dqn_baseline("buffer_size", BaselineParamsHpSearch)

                buffer_sizes = [20, 80, 100]

                configs = [default] + [DQNParams(description=f"bf_size={bsize}k", buffer_size=bsize * 1000) for bsize in
                                       buffer_sizes]
                return configs

        class TargetUpdates(RunnableExperiment):
            @classmethod
            def logger(cls):
                return cls(DQN.exp_result_logger)

            @classmethod
            def get(cls):
                default = get_dqn_baseline("target_network_updates", BaselineParamsHpSearch)

                intervals = [1, 2, 4, 6, 8, 10]

                configs = [default] + [DQNParams(description=f"interval={up_interval}",
                                                 target_network_updates=up_interval * 100) for
                                       up_interval in intervals]
                return configs

        class LayerNorm(RunnableExperiment):
            @classmethod
            def logger(cls):
                return cls(DQN.exp_result_logger)

            @classmethod
            def get(cls):
                default = get_dqn_baseline("layer_norm", BaselineParamsAlgComparison)

                configs = [default, DQNParams(description=f"layer_norm=True", layer_norm=True)]
                return configs

        class DuelingDQN(RunnableExperiment):
            @classmethod
            def logger(cls):
                return cls(DQN.exp_result_logger)

            @classmethod
            def get(cls):
                default = get_dqn_baseline("dueling", BaselineParamsAlgComparison)

                configs = [default, DQNParams(description=f"dueling=True", dueling=True)]

                return configs

        class Her(RunnableExperiment):
            @classmethod
            def logger(cls):
                return cls(DQN.exp_result_logger)

            @classmethod
            def get(cls):
                default = get_dqn_baseline("replay", BaselineParamsAlgComparison)
                default.description = "BL=ExperienceReplay"
                her = DQNParams(description=f"HER_future")
                her.use_her(mode="future", k=4)
                her_2 = DQNParams(description=f"HER_future_2")
                her_2.use_her(mode="future", k=2)
                her_episode = DQNParams(description=f"HER_episode")
                her_episode.use_her(mode="episode", k=4)

                her_random = DQNParams(description=f"HER_random")
                her_random.use_her(mode="random", k=4)

                configs = [default, her, her_2, her_episode, her_random]

                return configs

        class Per(RunnableExperiment):
            @classmethod
            def logger(cls):
                return cls(DQN.exp_result_logger)

            @classmethod
            def get(cls):
                default = get_dqn_baseline("replay", BaselineParamsAlgComparison)
                default.description = "BL=ExperienceReplay"
                cfg_1 = DQNParams(description=f"alpha=0.2_beta0=0.4")
                cfg_1.use_exp_replay(alpha=0.2, beta0=0.4)
                cfg_2 = DQNParams(description=f"alpha=0.4_beta0=0.4")
                cfg_2.use_exp_replay(alpha=0.4, beta0=0.4)

                cfg_3 = DQNParams(description=f"alpha=0.6_beta0=0.4")
                cfg_3.use_exp_replay(alpha=0.6, beta0=0.4)
                cfg_4 = DQNParams(description=f"alpha=0.8_beta0=0.4")
                cfg_4.use_exp_replay(alpha=0.8, beta0=0.4)

                cfg_5 = DQNParams(description=f"alpha=0.6_beta0=0.2")
                cfg_5.use_exp_replay(alpha=0.6, beta0=0.2)

                cfg_6 = DQNParams(description=f"alpha=0.6_beta0=0.6")
                cfg_6.use_exp_replay(alpha=0.6, beta0=0.6)

                configs = [default, cfg_1, cfg_2,
                           cfg_3, cfg_4, cfg_5, cfg_6]

                return configs

        class Replay(RunnableExperiment):
            @classmethod
            def logger(cls):
                return cls(DQN.exp_result_logger)

            @classmethod
            def get(cls):
                default = get_dqn_baseline("replay", BaselineParamsAlgComparison)
                default.description = "BL=ExperienceReplay"
                her = DQNParams(description=f"HER_future_4")
                her.use_her(mode="future", k=4)

                exp = DQNParams(description=f"EXP_a=0.6_b=0.4_eps=1e-6")
                exp.use_exp_replay(alpha=0.6, beta0=0.4, eps=1e-6)

                configs = [default, her, exp]

                return configs


class PPO:
    class BaselineParams:
        params = DotDict()
        params.batch_size = 32
        params.layers = [64, 64]
        params.lr = 5e-4
        params.dueling = False
        params.layer_norm = False

        params.target_network_updates = 5 * 100

        @classmethod
        def to_kwargs(cls):
            return cls.params

    base_dir = Folder(f"{os.path.dirname(__file__)}/experiments/ppo_experiments")
    exp_result_logger = ExperimentResultLogger(base_folder=base_dir)

    class Experiments:
        class LearningRate(RunnableExperiment):

            @classmethod
            def logger(cls):
                return cls(PPO.exp_result_logger)

            @classmethod
            def get(cls):
                default = get_ppo_baseline("lr", PPO.BaselineParams)
                return [default]

                rates = [0.01, 0.001, 0.002, 0.005, 0.0001]

                configs = [default] + [PPOv2Params(description=f"lr={bsize}", lr=bsize) for bsize in rates]
                return configs


def list_setup():
    print("DQN")
    print("Learning rate")
    print(DQN.Experiments.LearningRate.get())
    print("Batch Size")
    print(DQN.Experiments.BatchSize.get())
    print("Layers")
    print(DQN.Experiments.Layers.get())
    print("Target updates")
    print(DQN.Experiments.TargetUpdates.get())
    print("Buffer size")
    print(DQN.Experiments.BufferSize.get())
    print("LayerNorm")
    print(DQN.Experiments.LayerNorm.get())
    print("Dueling")
    print(DQN.Experiments.DuelingDQN.get())
    print("Replay")
    print(DQN.Experiments.Replay.get())


def run_experiment(env, experiments=None):
    target_env = env

    if experiments is None:
        DQN.Experiments.LearningRate.logger().run(target_env)

        DQN.Experiments.BatchSize.logger().run(target_env)
        DQN.Experiments.BufferSize.logger().run(target_env)
        DQN.Experiments.Layers.logger().run(target_env)
        DQN.Experiments.TargetUpdates.logger().run(target_env)
        DQN.Experiments.DuelingDQN.logger().run(target_env)
        DQN.Experiments.LayerNorm.logger().run(target_env)
        DQN.Experiments.Replay.logger().run(target_env)
    else:
        for experiment in experiments:
            experiment.logger().run(target_env)


def update_charts(env, experiments=None):
    target_env = env
    if experiments is None:
        DQN.Experiments.LearningRate.logger().view(target_env).save_overview()
        DQN.Experiments.BatchSize.logger().view(target_env).save_overview()
        DQN.Experiments.BufferSize.logger().view(target_env).save_overview()
        DQN.Experiments.Layers.logger().view(target_env).save_overview()
        DQN.Experiments.TargetUpdates.logger().view(target_env).save_overview()
        DQN.Experiments.DuelingDQN.logger().view(target_env).save_overview()
        DQN.Experiments.LayerNorm.logger().view(target_env).save_overview()
        DQN.Experiments.Replay.logger().view(target_env).save_overview()
        DQN.Experiments.Her.logger().view(target_env).save_overview()
        DQN.Experiments.Per.logger().view(target_env).save_overview()
    else:
        for experiment in experiments:
            experiment.logger().view(target_env).save_overview()


def run(env_list, train=False, experiments=None):
    for env in env_list:
        if train:
            run_experiment(env, experiments)
        update_charts(env, experiments)


def compare(env_list, exp_list, agent_id):
    agent_collection = dict()
    for exp in exp_list:
        for env, env_name in env_list:
            agent = exp.logger().view(env).agents[agent_id]
            agent.rename(agent.name() + "@" + env_name)
            agent_collection[agent.name()] = agent

    ExperimentViewer(None, agent_collection).show_overview(smooth=10)

    print(agent_collection)


if __name__ == '__main__':
    list_setup()

    experiments = [DQN.Experiments.Replay]
    #experiments = [DQN.Experiments.Her, DQN.Experiments.Per]

    envs = [
        EnvBuilder().default().fluids(2).build(),
        #EnvBuilder().default().fluids(7).build(),
        #EnvBuilder().default().fluids(14).build(),
        #EnvBuilder().saturated().fluids(3).build(),
        #EnvBuilder().saturated().fluids(7).build(),
        #EnvBuilder().saturated().fluids(14).build()
    ]
    run(envs, train=True, experiments=experiments)

    # compare(list(zip(envs, ["2fluids", "7fluids", "14fluids"])), experiments, 'BL=ExperienceReplay')
