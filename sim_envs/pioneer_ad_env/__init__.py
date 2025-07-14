'''
Author zzy-office zhuzy_2013@163.com
Date 2025-07-11 15:53:51
LastEditors zzy-office zhuzy_2013@163.com
LastEditTime 2025-07-11 15:53:59
FilePath /pioneer_sim/sim_envs/pioneer_ad_env/__init__.py
Description 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
from gymnasium.envs.registration import register

register(
    id='pioneer_ad_env/ADEnv-v0',
    entry_point='pioneer_ad_env.envs:ADEnv',
)