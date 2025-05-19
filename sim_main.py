'''
Author zhuzy zhuzy_2013@163.com
Date 2025-05-19 18:56:07
LastEditors zhuzy zhuzy_2013@163.com
LastEditTime 2025-05-19 18:58:51
FilePath /pioneer_sim/sim_main.py
Description 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import sim_envs
import gymnasium


env = gymnasium.make('easiest_ad_env/ADEnv-v0', render_mode='human')
obs, _ = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
env.close()
