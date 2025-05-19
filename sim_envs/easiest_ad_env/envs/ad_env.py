'''
Author zhuzy zhuzy_2013@163.com
Date 2025-05-19 18:44:25
LastEditors zhuzy zhuzy_2013@163.com
LastEditTime 2025-05-19 18:44:25
FilePath /pioneer_sim/sim_envs/easiest_ad_env/envs/ad_env.py
Description 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import gymnasium
from gymnasium import spaces
import numpy as np
import pygame

class ADEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, render_mode=None):
        super().__init__()
        self.num_lanes = 3
        self.min_speed = 1
        self.max_speed = 10

        # 观测空间：车道编号(0,1,2)和速度
        self.observation_space = spaces.Box(
            low=np.array([0, self.min_speed], dtype=np.float32),
            high=np.array([self.num_lanes - 1, self.max_speed], dtype=np.float32),
            dtype=np.float32
        )
        # 动作空间
        self.action_space = spaces.Discrete(5)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.width = 300
        self.height = 500
        self.car_width = 40
        self.car_height = 60

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.lane = 1  # 初始中间车道
        self.speed = 5  # 初始速度
        obs = np.array([self.lane, self.speed], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # 动作处理
        if action == 0 and self.lane > 0:
            self.lane -= 1
        elif action == 1 and self.lane < self.num_lanes - 1:
            self.lane += 1
        elif action == 3 and self.speed < self.max_speed:
            self.speed += 1
        elif action == 4 and self.speed > self.min_speed:
            self.speed -= 1
        # 观测
        obs = np.array([self.lane, self.speed], dtype=np.float32)
        reward = 0  # 不设计 reward
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("ADEnv 三车道仿真")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))
        # 画三条道
        lane_width = self.width // self.num_lanes
        for i in range(1, self.num_lanes):
            pygame.draw.line(self.screen, (128, 128, 128), (i * lane_width, 0), (i * lane_width, self.height), 3)

        # 画车
        car_x = self.lane * lane_width + lane_width // 2 - self.car_width // 2
        car_y = self.height - self.car_height - 30
        pygame.draw.rect(self.screen, (0, 100, 255), (car_x, car_y, self.car_width, self.car_height))

        # 显示速度
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Speed: {self.speed}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

