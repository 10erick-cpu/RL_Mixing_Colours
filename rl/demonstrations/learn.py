from rl.envs.env_initializers import Envs
from rl.misc.model_params import DQNParams
from rl.visualization.experiment_runner import ExperimentRunner
from rl.visualization.results_viewer import ResultsViewer
from utils.helper_functions.colors import ColorGenerator
from utils.models.folder import Folder

#### Folder setup ####
base_dir = Folder("./example_training_results", create=True)

agent_name = "example_agent"
experiment_name = "example_training"

#### Param setup ####


# the most basic DQN agent, see DQNParams class to play with parameters
default_agent = DQNParams(description=agent_name)
# e.g. default_agent.batch_size=512

# best configuration found by hyper-parameter analysis so far:
default_agent.batch_size = 256
default_agent.layers = [64, 128]
default_agent.lr = 0.001
default_agent.buffer_size = 50 * 1000
default_agent.dueling = False
default_agent.layer_norm = False

default_agent.target_network_updates = 8 * 100


#### Configs ####

configs = [default_agent]

# This is the simulation of the real world experiment with two fluids (red/blue)
init_helper = Envs.Default2Fluids()


def train_agent(num_episodes, num_runs_per_agent=1, intermediate_eval_interval=20, num_workers=4):
    runner = ExperimentRunner(base_dir,
                              num_episodes_per_eval=intermediate_eval_interval,
                              runs_per_config=num_runs_per_agent)

    #### Execution ####
    runner.execute(experiment_name, configs, init_helper, num_episodes_to_learn=num_episodes, num_workers=num_workers)


def show_results():
    #### Inspection ####

    rv = ResultsViewer(base_dir)
    print(rv.list_experiments())

    # get a viewer for the last experiment
    experiment_viewer = rv.view_experiment(experiment_name)

    # viewer can combine multiple runs of the same agent and multiple runs of multiple agents into one training plot
    #experiment_viewer.mr_training(palette=ColorGenerator([agent_name]), show=True)
    # show percentage of success steps (distance to goal <=1) after reaching the goal for the first time
    # (displays accuracy, ideally should be 1.0)
    #experiment_viewer.spag_training(palette=ColorGenerator([agent_name]), show=True)

    # get the training and validation history of a specific agent.
    # we can view individual training / validation stats here
    agent_history = experiment_viewer.get_agent(agent_name)

    # view the distribution of goals during training.
    #agent_history.plot_goal_distribution(show=True)

    # view episode 6 of the training procedure
    agent_history.plot_training_episode(episode_id=92)

    experiment_viewer.save_overview()
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    # set this to False to only see the results
    do_train = True

    if do_train:
        # train an agent once for 100 episodes
        train_agent(num_runs_per_agent=1, num_episodes=100)


    show_results()
