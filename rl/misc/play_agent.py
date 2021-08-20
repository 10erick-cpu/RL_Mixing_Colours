import numpy as np
from stable_baselines.her import HERGoalEnvWrapper

from utils.fluids.envs.env_criteria.goals import DeterministicGoalGenerator
from utils.fluids.envs.env_state.fluid_env_initializers import CustomInitializer


def play(model, action_selector, env=None, num_episodes=5, max_steps=50, goals=None, init=None, episode_tracker=None):
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
        if episode_tracker:
            episode_tracker.reset()
        done = False
        step = 0
        while not done:
            step += 1
            action = action_selector(obs)
            if isinstance(action, tuple):
                action = action[0]

            data = env.step(action)
            obs, reward, done, info = data

            if episode_tracker:
                episode_tracker.step(action, data)

            env.render('plt', wait_delay=0.1)
            if done:
                break
            if step > max_steps:
                break
        if episode_tracker:
            episode_tracker.reset()
