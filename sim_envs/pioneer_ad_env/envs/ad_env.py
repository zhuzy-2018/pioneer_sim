# pioneer_ad_env/envs/ad_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random
from typing import Dict, Any, List, Tuple, Optional

class ADEnv(gym.Env):
    """自动驾驶仿真环境"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # 环境参数
        self.render_mode = render_mode
        self.window_width = 1200
        self.window_height = 800
        self.road_width = 10.5  # 3车道，每车道3.5米
        self.lane_width = 3.5
        self.max_other_vehicles = 15  # 增加最大车辆数
        self.max_episode_steps = 1000
        
        # 车辆生成参数
        self.vehicle_spawn_interval = 30  # 每30步生成一辆车
        self.vehicle_spawn_distance = 50  # 在自车后方50米生成
        self.vehicle_despawn_distance = 100  # 在自车前方100米消失
        
        # 状态变量
        self.ego_vehicle = None
        self.other_vehicles = []
        self.current_step = 0
        self.episode_reward = 0
        self.last_waypoints = []  # 存储最新的waypoints用于可视化
        
        # 渲染相关
        self.screen = None
        self.clock = None
        
        # 定义动作空间
        self.action_space = spaces.Dict({
            'waypoints': spaces.Box(
                low=np.array([[-1000, -50, -np.pi]] * 10),
                high=np.array([[1000, 50, np.pi]] * 10),
                dtype=np.float32
            ),
            'target_speed': spaces.Box(low=0.0, high=30.0, shape=(1,), dtype=np.float32)
        })
        
        # 定义观测空间
        self.observation_space = spaces.Dict({
            'ego_vehicle': spaces.Dict({
                'pose': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                'velocity': spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32),
                'steering_angle': spaces.Box(low=-np.pi/4, high=np.pi/4, shape=(1,), dtype=np.float32),
                'throttle': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                'brake': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            }),
            'other_vehicles': spaces.Box(
                low=np.array([[-1000, -50, -np.pi, -50, -50, -50]] * self.max_other_vehicles),
                high=np.array([[1000, 50, np.pi, 50, 50, 50]] * self.max_other_vehicles),
                dtype=np.float32
            ),
            'num_other_vehicles': spaces.Box(low=0, high=self.max_other_vehicles, shape=(1,), dtype=np.int32),
            'map_data': spaces.Dict({
                'lane_centers': spaces.Box(low=-1000, high=1000, shape=(3, 2), dtype=np.float32),
                'road_boundaries': spaces.Box(low=-1000, high=1000, shape=(2, 2), dtype=np.float32)
            }),
            'timestamp': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64),
            'weather': spaces.Dict({
                'condition': spaces.Discrete(4),
                'visibility': spaces.Box(low=10.0, high=1000.0, shape=(1,), dtype=np.float32),
                'precipitation': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
            }),
            'traffic_lights': spaces.Box(
                low=np.array([[-1000, -50, 0]] * 5),
                high=np.array([[1000, 50, 2]] * 5),
                dtype=np.float32
            ),
            'road_signs': spaces.Box(
                low=np.array([[-1000, -50, 0]] * 10),
                high=np.array([[1000, 50, 10]] * 10),
                dtype=np.float32
            )
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 重置状态
        self.current_step = 0
        self.episode_reward = 0
        self.last_waypoints = []
        
        # 初始化自车
        self.ego_vehicle = {
            'pose': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x, y, z, roll, pitch, yaw
            'velocity': [15.0, 0.0, 0.0],  # vx, vy, vz
            'steering_angle': 0.0,
            'throttle': 0.5,
            'brake': 0.0
        }
        
        # 初始化其他车辆（在前方生成一些初始车辆）
        self.other_vehicles = []
        self._spawn_initial_vehicles()
        
        # 补齐到最大数量
        while len(self.other_vehicles) < self.max_other_vehicles:
            self.other_vehicles.append([0.0] * 9)
        
        # 构建观测
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _spawn_initial_vehicles(self):
        """生成初始车辆"""
        num_initial = random.randint(3, 8)
        for i in range(num_initial):
            x = random.uniform(20, 80)  # 在前方20-80米
            y = random.choice([-3.5, 0.0, 3.5])  # 随机选择车道
            yaw = random.uniform(-0.1, 0.1)
            vx = random.uniform(8, 18)  # 速度略低于自车
            
            vehicle = [x, y, yaw, 0.0, 0.0, 0.0, vx, 0.0, 0.0]
            self.other_vehicles.append(vehicle)
    
    def _spawn_vehicle_from_behind(self):
        """从后方生成新车辆"""
        ego_x = self.ego_vehicle['pose'][0]
        
        # 在自车后方生成
        spawn_x = ego_x - self.vehicle_spawn_distance - random.uniform(0, 20)
        spawn_y = random.choice([-3.5, 0.0, 3.5])  # 随机车道
        spawn_yaw = random.uniform(-0.1, 0.1)
        spawn_vx = random.uniform(12, 22)  # 速度稍高，能够追上
        
        new_vehicle = [spawn_x, spawn_y, spawn_yaw, 0.0, 0.0, 0.0, spawn_vx, 0.0, 0.0]
        
        # 找到空位置插入
        for i, vehicle in enumerate(self.other_vehicles):
            if all(abs(x) < 1e-6 for x in vehicle[:6]):
                self.other_vehicles[i] = new_vehicle
                print(f"🚗 生成新车辆在位置 ({spawn_x:.1f}, {spawn_y:.1f})")
                break
    
    def _update_other_vehicles(self):
        """更新其他车辆状态"""
        ego_x = self.ego_vehicle['pose'][0]
        
        for i, vehicle in enumerate(self.other_vehicles):
            if len(vehicle) >= 9 and any(abs(x) > 1e-6 for x in vehicle[:6]):
                # 更新位置
                vehicle[0] += vehicle[6] * 0.1  # x += vx * dt
                
                # 添加一些随机的横向微调（模拟真实驾驶）
                if random.random() < 0.1:  # 10%概率微调
                    vehicle[1] += random.uniform(-0.2, 0.2)
                    # 限制在车道内
                    vehicle[1] = max(-5.0, min(5.0, vehicle[1]))
                
                # 检查是否需要移除（太远的车辆）
                if vehicle[0] > ego_x + self.vehicle_despawn_distance:
                    print(f"🗑️ 移除远距离车辆 ({vehicle[0]:.1f}, {vehicle[1]:.1f})")
                    self.other_vehicles[i] = [0.0] * 9
                elif vehicle[0] < ego_x - self.vehicle_despawn_distance:
                    print(f"🗑️ 移除后方车辆 ({vehicle[0]:.1f}, {vehicle[1]:.1f})")
                    self.other_vehicles[i] = [0.0] * 9
    
    def _get_observation(self) -> Dict[str, Any]:
        """获取当前观测"""
        # 构建其他车辆观测（只取前6个元素：pose信息）
        other_vehicles_obs = []
        for vehicle in self.other_vehicles:
            if len(vehicle) >= 6:
                other_vehicles_obs.append(vehicle[:6])
            else:
                other_vehicles_obs.append([0.0] * 6)
        
        # 生成天气信息
        weather_obs = {
            'condition': random.randint(0, 3),
            'visibility': np.array([random.uniform(100.0, 1000.0)], dtype=np.float32),
            'precipitation': np.array([random.uniform(0.0, 20.0)], dtype=np.float32)
        }
        
        # 生成交通灯信息（模拟）
        traffic_lights_obs = []
        num_lights = random.randint(0, 3)
        for i in range(num_lights):
            x = random.uniform(50, 200)
            y = random.choice([-3.5, 0.0, 3.5])
            state = random.randint(0, 2)
            traffic_lights_obs.append([x, y, state])
        
        # 补齐到最大数量
        while len(traffic_lights_obs) < 5:
            traffic_lights_obs.append([0.0, 0.0, 0.0])
        
        # 生成路标信息（模拟）
        road_signs_obs = []
        num_signs = random.randint(0, 5)
        for i in range(num_signs):
            x = random.uniform(10, 100)
            y = random.choice([-5.0, 5.0])
            sign_type = random.randint(0, 10)
            road_signs_obs.append([x, y, sign_type])
        
        # 补齐到最大数量
        while len(road_signs_obs) < 10:
            road_signs_obs.append([0.0, 0.0, 0.0])
        
        obs = {
            'ego_vehicle': {
                'pose': np.array(self.ego_vehicle['pose'], dtype=np.float32),
                'velocity': np.array(self.ego_vehicle['velocity'], dtype=np.float32),
                'steering_angle': np.array([self.ego_vehicle['steering_angle']], dtype=np.float32),
                'throttle': np.array([self.ego_vehicle['throttle']], dtype=np.float32),
                'brake': np.array([self.ego_vehicle['brake']], dtype=np.float32)
            },
            'other_vehicles': np.array(other_vehicles_obs, dtype=np.float32),
            'num_other_vehicles': np.array([len([v for v in self.other_vehicles if any(abs(x) > 1e-6 for x in v[:6])])], dtype=np.int32),
            'map_data': {
                'lane_centers': np.array([[-3.5, 0.0], [0.0, 0.0], [3.5, 0.0]], dtype=np.float32),
                'road_boundaries': np.array([[-5.25, 0.0], [5.25, 0.0]], dtype=np.float32)
            },
            'timestamp': np.array([self.current_step * 0.1], dtype=np.float64),
            'weather': weather_obs,
            'traffic_lights': np.array(traffic_lights_obs, dtype=np.float32),
            'road_signs': np.array(road_signs_obs, dtype=np.float32)
        }
        
        return obs
    
    def step(self, action: Dict[str, Any]):
        """执行一步仿真"""
        # 解析动作
        waypoints = action.get('waypoints', [])
        target_speed = action.get('target_speed', 15.0)
        
        # 保存waypoints用于可视化
        if isinstance(waypoints, np.ndarray):
            self.last_waypoints = waypoints.tolist()
        else:
            self.last_waypoints = waypoints
        
        # 更新自车状态
        if len(waypoints) > 0:
            # 简单的跟踪逻辑
            target_x, target_y = waypoints[0][:2]
            current_x, current_y = self.ego_vehicle['pose'][:2]
            
            # 计算朝向目标的角度
            dx = target_x - current_x
            dy = target_y - current_y
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                target_yaw = math.atan2(dy, dx)
            else:
                target_yaw = self.ego_vehicle['pose'][5]
            
            # 更新位置和朝向
            if isinstance(target_speed, np.ndarray):
                speed = target_speed[0]
            else:
                speed = target_speed
                
            self.ego_vehicle['pose'][0] += speed * 0.1 * math.cos(target_yaw)
            self.ego_vehicle['pose'][1] += speed * 0.1 * math.sin(target_yaw)
            self.ego_vehicle['pose'][5] = target_yaw  # yaw
            
            # 更新速度
            self.ego_vehicle['velocity'][0] = speed
        
        # 更新其他车辆
        self._update_other_vehicles()
        
        # 定期生成新车辆
        if self.current_step % self.vehicle_spawn_interval == 0:
            self._spawn_vehicle_from_behind()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查终止条件
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        self.current_step += 1
        self.episode_reward += reward
        
        # 获取观测和信息
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 速度奖励
        speed = self.ego_vehicle['velocity'][0]
        reward += min(speed / 20.0, 1.0)  # 鼓励合理速度
        
        # 车道保持奖励
        y_pos = self.ego_vehicle['pose'][1]
        lane_deviation = min(abs(y_pos + 3.5), abs(y_pos), abs(y_pos - 3.5))
        reward += max(0, 1.0 - lane_deviation / 1.75)  # 鼓励在车道中心
        
        # 碰撞惩罚
        if self._check_collision():
            reward -= 100.0
        
        # 前进奖励
        reward += 0.1  # 每步基础奖励
        
        return reward
    
    def _check_collision(self) -> bool:
        """检查碰撞"""
        ego_x, ego_y = self.ego_vehicle['pose'][:2]
        
        for vehicle in self.other_vehicles:
            if len(vehicle) >= 6 and any(abs(x) > 1e-6 for x in vehicle[:6]):
                other_x, other_y = vehicle[:2]
                distance = math.sqrt((ego_x - other_x)**2 + (ego_y - other_y)**2)
                if distance < 3.0:  # 碰撞阈值
                    return True
        
        return False
    
    def _check_termination(self) -> bool:
        """检查终止条件"""
        # 碰撞终止
        if self._check_collision():
            return True
        
        # 偏离道路终止
        y_pos = self.ego_vehicle['pose'][1]
        if abs(y_pos) > 6.0:  # 超出道路边界
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            'step_time': 0.1,
            'collision': self._check_collision(),
            'ego_lane': self._get_ego_lane(),
            'episode_reward': self.episode_reward,
            'speed': self.ego_vehicle['velocity'][0],
            'active_vehicles': len([v for v in self.other_vehicles if any(abs(x) > 1e-6 for x in v[:6])])
        }
    
    def _get_ego_lane(self) -> int:
        """获取自车所在车道"""
        y_pos = self.ego_vehicle['pose'][1]
        if y_pos < -1.75:
            return 0  # 左车道
        elif y_pos > 1.75:
            return 2  # 右车道
        else:
            return 1  # 中间车道
    
    def render(self):
        """渲染环境"""
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Pioneer AD Simulation")
            self.clock = pygame.time.Clock()
        
        # 清屏
        self.screen.fill((50, 50, 50))  # 深灰色背景
        
        # 绘制道路
        road_y = self.window_height // 2
        road_height = int(self.road_width * 20)  # 缩放因子
        
        # 道路背景
        pygame.draw.rect(self.screen, (80, 80, 80), 
                        (0, road_y - road_height//2, self.window_width, road_height))
        
        # 车道线
        lane_line_color = (255, 255, 255)
        for i in range(1, 3):  # 2条车道线
            y = road_y - road_height//2 + i * road_height//3
            for x in range(0, self.window_width, 40):
                pygame.draw.rect(self.screen, lane_line_color, (x, y-1, 20, 2))
        
        # 道路边界
        pygame.draw.line(self.screen, (255, 255, 0), 
                        (0, road_y - road_height//2), (self.window_width, road_y - road_height//2), 3)
        pygame.draw.line(self.screen, (255, 255, 0), 
                        (0, road_y + road_height//2), (self.window_width, road_y + road_height//2), 3)
        
        # 绘制自车
        ego_x = self.window_width // 4  # 固定在屏幕1/4处
        ego_y = road_y - int(self.ego_vehicle['pose'][1] * 20)
        pygame.draw.rect(self.screen, (0, 255, 0), (ego_x-15, ego_y-8, 30, 16))
        
        # 绘制自车朝向箭头
        ego_yaw = self.ego_vehicle['pose'][5]
        arrow_length = 25
        arrow_end_x = ego_x + int(arrow_length * math.cos(ego_yaw))
        arrow_end_y = ego_y - int(arrow_length * math.sin(ego_yaw))
        pygame.draw.line(self.screen, (0, 255, 0), (ego_x, ego_y), (arrow_end_x, arrow_end_y), 3)
        
        # 绘制waypoints轨迹
        if self.last_waypoints:
            ego_world_x = self.ego_vehicle['pose'][0]
            waypoint_points = []
            
            for waypoint in self.last_waypoints:
                if len(waypoint) >= 2:
                    relative_x = waypoint[0] - ego_world_x
                    screen_x = ego_x + int(relative_x * 5)  # 缩放因子
                    screen_y = road_y - int(waypoint[1] * 20)
                    
                    if 0 <= screen_x <= self.window_width:
                        waypoint_points.append((screen_x, screen_y))
                        # 绘制waypoint点
                        pygame.draw.circle(self.screen, (0, 255, 255), (screen_x, screen_y), 4)
            
            # 绘制waypoint连线
            if len(waypoint_points) > 1:
                pygame.draw.lines(self.screen, (0, 255, 255), False, waypoint_points, 2)
        
        # 绘制其他车辆
        ego_world_x = self.ego_vehicle['pose'][0]
        active_vehicles = 0
        for vehicle in self.other_vehicles:
            if len(vehicle) >= 6 and any(abs(x) > 1e-6 for x in vehicle[:6]):
                active_vehicles += 1
                relative_x = vehicle[0] - ego_world_x
                screen_x = ego_x + int(relative_x * 5)  # 缩放因子
                screen_y = road_y - int(vehicle[1] * 20)
                
                if -50 <= screen_x <= self.window_width + 50:  # 扩大渲染范围
                    # 根据距离调整颜色
                    distance = abs(relative_x)
                    if distance < 20:
                        color = (255, 0, 0)  # 红色 - 近距离
                    elif distance < 50:
                        color = (255, 165, 0)  # 橙色 - 中距离
                    else:
                        color = (255, 255, 0)  # 黄色 - 远距离
                    
                    pygame.draw.rect(self.screen, color, (screen_x-12, screen_y-6, 24, 12))
                    
                    # 绘制车辆速度向量
                    if len(vehicle) >= 9:
                        vel_x = vehicle[6]
                        vel_scale = 2
                        vel_end_x = screen_x + int(vel_x * vel_scale)
                        pygame.draw.line(self.screen, color, (screen_x, screen_y), (vel_end_x, screen_y), 2)
        
        # 显示信息
        font = pygame.font.Font(None, 36)
        speed_text = font.render(f"Speed: {self.ego_vehicle['velocity'][0]:.1f} m/s", True, (255, 255, 255))
        step_text = font.render(f"Step: {self.current_step}", True, (255, 255, 255))
        reward_text = font.render(f"Reward: {self.episode_reward:.2f}", True, (255, 255, 255))
        vehicles_text = font.render(f"Active Vehicles: {active_vehicles}", True, (255, 255, 255))
        lane_text = font.render(f"Lane: {self._get_ego_lane()}", True, (255, 255, 255))
        
        self.screen.blit(speed_text, (10, 10))
        self.screen.blit(step_text, (10, 50))
        self.screen.blit(reward_text, (10, 90))
        self.screen.blit(vehicles_text, (10, 130))
        self.screen.blit(lane_text, (10, 170))
        
        # 显示图例
        legend_y = self.window_height - 100
        legend_font = pygame.font.Font(None, 24)
        legend_texts = [
            ("Green: Ego Vehicle", (0, 255, 0)),
            ("Cyan: Waypoints", (0, 255, 255)),
            ("Red: Close Vehicles", (255, 0, 0)),
            ("Orange: Medium Distance", (255, 165, 0)),
            ("Yellow: Far Vehicles", (255, 255, 0))
        ]
        
        for i, (text, color) in enumerate(legend_texts):
            legend_surface = legend_font.render(text, True, color)
            self.screen.blit(legend_surface, (10, legend_y + i * 20))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
