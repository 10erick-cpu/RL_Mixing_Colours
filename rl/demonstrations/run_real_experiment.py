from stable_baselines.her import HERGoalEnvWrapper

import numpy as np

from rl.envs.env_initializers import Envs
from rl.visualization.results_viewer import ResultsViewer
from utils.fluids.envs.env_criteria.goals import DeterministicGoalGenerator
from utils.fluids.envs.env_state.fluid_env_initializers import CustomInitializer


def play(model, action_selector, env=None, num_episodes=5, max_steps=50, goals=None, init=None):
    if env is None:
        env = model.env

    dgg = None
    # change from random goal generation to predefined goals
    if goals is not None:
        dgg = DeterministicGoalGenerator(goal_list=goals, allow_index_overflow=True)
        if isinstance(env, HERGoalEnvWrapper):
            env = env.env
        env.goal_generator(dgg)

    # change from random state initialization to deterministic initialization
    if init is not None:
        if isinstance(init, list):
            init = np.asarray(init)

        def init_fn(val, goal):
            return init

        env.initializer(CustomInitializer(init_fn))

    # typical interaction loop

    for i in range(num_episodes):
        if dgg is not None:
            dgg.advance_goal_idx()

        obs = env.reset()
        done = False
        step = 0
        while not done:
            step += 1
            action = action_selector(obs)
            if isinstance(action, tuple):
                action = action[0]

            obs, reward, done, info = env.step(action)

            env.render('plt', wait_delay=0.1)
            if done:
                break
            if step > max_steps:
                break


def initialize_agent(experiment_name, agent_name, agent_run_id=0, target_env_helper=None):
    from rl.demonstrations.learn import base_dir, init_helper
    rv = ResultsViewer(base_dir)
    print(rv.list_experiments())

    # get a viewer for the last experiment
    experiment_viewer = rv.view_experiment(experiment_name)

    # find our agent to play
    agent_history = experiment_viewer.agents[agent_name]

    # get the agent's configuration
    agent_params = agent_history.cfg
    # build the agent from the configuration and the environment

    env = agent_params.build_env(init_helper if target_env_helper is None else target_env_helper)
    agent = agent_params.get_model(env)

    # load the agent state
    checkpoint_name = "final_model.pkl"
    # usually for experiments agents are trained multiple times to determine the variability during individual runs
    # we therefore load a specific run, in a simple example there is only one run therefore run_id=0
    cp_path = agent_history.root_dir.make_sub_folder(f"run_{agent_run_id}", create=False).get_file_path(checkpoint_name)
    agent = agent.load(cp_path, env=agent.env)

    return agent


if __name__ == '__main__':

    env = Envs.Real2Fluids()

    # this agent has to be trained before by executing learn.py
    agent = initialize_agent("example_training", "example_agent", agent_run_id=0, target_env_helper=env)

    def action_selector_fn(obs):
        # agent prediction returns a tuple of (action_idx, q-values), we only need the idx
        return agent.predict(obs)[0]

    # mind the double brackets here
    goals = [[150]]

    # randomly sample goals
    goals = None


    play(agent, action_selector_fn, num_episodes=10, max_steps=100, goals=goals)

