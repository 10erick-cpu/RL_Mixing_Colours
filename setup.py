from setuptools import setup

setup(
    name='labmAIte',
    version='0.0.1',
    packages=['rl', 'rl.envs', 'rl.envs.gym_envs', 'rl.envs.simple_envs', 'rl.envs.configurations', 'rl.misc', 'rl.debugging',
              'rl.regression', 'rl.sim_analysis', 'rl.visualization', 'rl.demonstrations', 'utils', 'utils.nn', 'utils.fluids',
              'utils.fluids.envs', 'utils.fluids.envs.real_env', 'utils.fluids.envs.env_state', 'utils.fluids.envs.env_criteria',
              'utils.datasets', 'utils.experiment_control', 'utils.file_management', 'utils.helper_functions'],
    url='',
    license='',
    author='Dennis Raith',
    author_email='dennis.raith@gmail.com',
    description=''
)
