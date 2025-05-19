'''
Author zhuzy zhuzy_2013@163.com
Date 2025-05-19 18:02:11
LastEditors zhuzy zhuzy_2013@163.com
LastEditTime 2025-05-19 18:26:58
FilePath /pioneer_sim/sim_envs/zzy_example_env/envs/test.py
Description 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import gymnasium
import numpy as np

class TestEnv(gymnasium.Env):
    """
    Custom environment for testing purposes.
    """
    def __init__(self):
        super(TestEnv, self).__init__()
        self.action_space = gymnasium.spaces.Discrete(2)  # 两个离散动作：0和1
        self.observation_space = gymnasium.spaces.Box(
            low=0, 
            high=1, 
            shape=(1,), 
            dtype=float
        )
        # 添加环境状态变量
        self.state = 0.5  # 初始状态设为0.5

    def reset(self, seed=None, options=None):
        # 重置环境到初始状态
        self.state = 0.5
        return np.array([self.state]), {}  # 返回初始观测和info字典

    def step(self, action):
        # 根据动作更新环境状态
        if action == 0:
            self.state = max(0, self.state - 0.1)  # 减小状态值
        else:
            self.state = min(1, self.state + 0.1)  # 增加状态值
        
        # 计算奖励
        reward = self.state
        
        # 判断是否结束
        terminated = abs(self.state - 1.0) < 1e-6
        truncated = False
        info = {}
        
        return np.array([self.state]), reward, terminated, truncated, info