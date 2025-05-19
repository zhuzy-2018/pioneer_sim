from gymnasium.envs.registration import register

register(
    id='easiest_ad_env/ADEnv-v0',
    entry_point='easiest_ad_env.envs:ADEnv',
)