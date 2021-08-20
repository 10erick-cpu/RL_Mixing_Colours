from gym.envs.registration import register

register(
    id='ColorMix3D-v0',
    entry_point='rl.envs.gym_envs:ColorMix3D',
)
register(
    id='ColorMix3D-v1',
    entry_point='rl.envs.gym_envs:ColorMix3D7Fluid',
)
register(
    id='ColorMix3D-v2',
    entry_point='rl.envs.gym_envs:ColorMix3D14Fluid',
)
register(
    id='ColorMix3DSaturated-v0',
    entry_point='rl.envs.gym_envs:ColorMix3DSaturation',
)
register(
    id='ColorMix3DSaturated-v1',
    entry_point='rl.envs.gym_envs:ColorMix3DSaturation7Fluids',
)
register(
    id='ColorMix3DSaturated-v2',
    entry_point='rl.envs.gym_envs:ColorMix3DSaturation14Fluids',
)

register(
    id='ColorMix3DContinuous-v0',
    entry_point='rl.envs.gym_envs:ColorMix3DContinuous',
)

register(
    id='ColorMixReal3D-v0',
    entry_point='rl.envs.gym_envs:ColorMixReal3D',
)
