from gymnasium.envs.registration import register

register(
    id='zzy_example_env/test-v0',
    entry_point='zzy_example_env.envs:TestEnv',
)