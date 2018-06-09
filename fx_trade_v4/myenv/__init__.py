from gym.envs.registration import register

register(
    id='myenv-v4',
    entry_point='myenv.env:MyEnv'
)

