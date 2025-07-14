# ad_algorithm/ad_model.py
"""
Author: zzy-office zhuzy_2013@163.com
Date: 2025-07-14 15:00:00
LastEditors: zzy-office zhuzy_2013@163.com
LastEditTime: 2025-07-14 15:30:00
FilePath: /pioneer_sim/ad_algorithm/ad_model.py
Description: AD算法核心逻辑模块

Copyright (c) 2025 by zzy-office, All Rights Reserved.
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Union
import threading

class ADModel:
    """AD算法模型类"""
    
    def __init__(self):
        self.model_loaded = False
        self.model_version = "1.0.0"
        
        # 避障参数
        self.safety_distance = 8.0      # 安全距离(米)
        self.lane_width = 3.5          # 车道宽度(米)
        self.max_lateral_offset = 2.0   # 最大横向偏移(米)
        self.lookahead_distance = 30.0  # 前瞻距离(米)
        self.planning_horizon = 5.0     # 规划时域(秒)
        self.target_speed = 15.0        # 目标速度(m/s)
        
        # 启动模型加载
        self.load_model()
    
    def load_model(self) -> None:
        """加载AD模型"""
        print("🔄 正在加载AD模型...")
        try:
            # 这里加载你的实际模型
            # 例如：UniAD, VAD, PlanT等
            
            # 模拟加载过程
            time.sleep(3)
            
            # 实际加载代码示例：
            # self.model = load_uniad_model(config_path, checkpoint_path)
            # self.model.eval()
            
            self.model_loaded = True
            print("✅ AD模型加载完成")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.model_loaded = False
    
    def is_model_ready(self) -> bool:
        """检查模型是否准备就绪"""
        return self.model_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'version': self.model_version,
            'loaded': self.model_loaded,
            'safety_distance': self.safety_distance,
            'planning_horizon': self.planning_horizon
        }
    
    def calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """计算两点间距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def parse_other_vehicles(self, other_vehicles_data: Union[np.ndarray, List]) -> List[Dict[str, Any]]:
        """解析其他车辆数据，统一转换为字典格式"""
        vehicles = []
        
        if isinstance(other_vehicles_data, np.ndarray):
            # 处理numpy数组格式
            for vehicle_array in other_vehicles_data:
                # 检查是否是有效车辆（不全为0）
                if np.any(np.abs(vehicle_array) > 1e-6):
                    vehicle_dict = {
                        'pose': vehicle_array[:6].tolist(),  # [x, y, z, roll, pitch, yaw]
                        'velocity': [0.0, 0.0, 0.0]  # 默认速度，如果有更多数据可以扩展
                    }
                    
                    # 如果数组长度超过6，可能包含速度信息
                    if len(vehicle_array) >= 9:
                        vehicle_dict['velocity'] = vehicle_array[6:9].tolist()
                    
                    vehicles.append(vehicle_dict)
        
        elif isinstance(other_vehicles_data, list):
            # 处理列表格式
            for vehicle in other_vehicles_data:
                if isinstance(vehicle, dict):
                    # 已经是字典格式
                    vehicles.append(vehicle)
                elif isinstance(vehicle, (list, np.ndarray)):
                    # 数组格式，转换为字典
                    if len(vehicle) >= 6 and np.any(np.abs(vehicle[:6]) > 1e-6):
                        vehicle_dict = {
                            'pose': vehicle[:6] if isinstance(vehicle, list) else vehicle[:6].tolist(),
                            'velocity': vehicle[6:9] if len(vehicle) >= 9 else [0.0, 0.0, 0.0]
                        }
                        if isinstance(vehicle_dict['velocity'], np.ndarray):
                            vehicle_dict['velocity'] = vehicle_dict['velocity'].tolist()
                        vehicles.append(vehicle_dict)
        
        return vehicles
    
    def is_collision_risk(self, ego_pos: List[float], other_pos: List[float], 
                         ego_vel: List[float], other_vel: List[float]) -> bool:
        """判断是否存在碰撞风险"""
        # 当前距离
        current_dist = self.calculate_distance(ego_pos[:2], other_pos[:2])
        
        # 如果当前距离就很近，认为有风险
        if current_dist < self.safety_distance:
            return True
        
        # 预测未来几秒的位置
        for t in np.arange(0.5, 3.0, 0.5):  # 0.5到3秒，每0.5秒检查一次
            ego_future = [
                ego_pos[0] + ego_vel[0] * t,
                ego_pos[1] + ego_vel[1] * t
            ]
            other_future = [
                other_pos[0] + other_vel[0] * t,
                other_pos[1] + other_vel[1] * t
            ]
            
            future_dist = self.calculate_distance(ego_future, other_future)
            if future_dist < self.safety_distance:
                return True
        
        return False
    
    def generate_avoidance_waypoints(self, ego_pos: List[float], ego_vel: List[float],
                                   obstacles: List[Dict[str, Any]]) -> List[List[float]]:
        """生成避障轨迹点"""
        waypoints = []
        
        # 基础前进方向（沿着当前朝向）
        ego_yaw = ego_pos[5] if len(ego_pos) > 5 else 0.0  # yaw是第6个元素（索引5）
        base_direction = [math.cos(ego_yaw), math.sin(ego_yaw)]
        
        # 分析障碍物分布
        left_blocked = False
        right_blocked = False
        front_blocked = False
        
        print(f"🔍 分析 {len(obstacles)} 个障碍物...")
        
        for i, obstacle in enumerate(obstacles):
            obs_pos = obstacle['pose']
            obs_vel = obstacle['velocity']
            
            print(f"  障碍物 {i}: 位置=({obs_pos[0]:.1f}, {obs_pos[1]:.1f}), 速度=({obs_vel[0]:.1f}, {obs_vel[1]:.1f})")
            
            # 检查是否在前方
            relative_pos = [obs_pos[0] - ego_pos[0], obs_pos[1] - ego_pos[1]]
            forward_dist = (relative_pos[0] * base_direction[0] + 
                          relative_pos[1] * base_direction[1])
            lateral_dist = (relative_pos[0] * (-base_direction[1]) + 
                          relative_pos[1] * base_direction[0])
            
            print(f"    前向距离: {forward_dist:.1f}m, 横向距离: {lateral_dist:.1f}m")
            
            if forward_dist > 0 and forward_dist < self.lookahead_distance:
                if self.is_collision_risk(ego_pos, obs_pos, ego_vel, obs_vel):
                    print(f"    ⚠️ 检测到碰撞风险!")
                    if abs(lateral_dist) < self.lane_width:
                        front_blocked = True
                        print(f"    🚧 前方阻塞")
                    elif lateral_dist > 0:
                        left_blocked = True
                        print(f"    🚧 左侧阻塞")
                    else:
                        right_blocked = True
                        print(f"    🚧 右侧阻塞")
        
        # 决定避障策略
        lateral_offset = 0.0
        if front_blocked:
            if not right_blocked:
                lateral_offset = -self.max_lateral_offset  # 向右避让
                print("🚗 策略: 向右避让")
            elif not left_blocked:
                lateral_offset = self.max_lateral_offset   # 向左避让
                print("🚗 策略: 向左避让")
            else:
                # 两边都有障碍，减速并保持直行
                lateral_offset = 0.0
                self.target_speed = max(5.0, self.target_speed * 0.5)
                print("🚗 策略: 减速直行")
        else:
            print("🚗 策略: 正常直行")
        
        # 生成轨迹点
        num_points = 10
        for i in range(num_points):
            t = (i + 1) * 0.5  # 每0.5秒一个点
            
            # 纵向位移
            longitudinal_dist = self.target_speed * t
            
            # 横向位移（使用平滑的正弦函数）
            if lateral_offset != 0:
                # 前半段逐渐偏移，后半段逐渐回正
                if i < num_points // 2:
                    current_lateral = lateral_offset * math.sin(math.pi * i / num_points)
                else:
                    current_lateral = lateral_offset * math.sin(math.pi * (num_points - i) / num_points)
            else:
                current_lateral = 0.0
            
            # 计算世界坐标
            waypoint_x = (ego_pos[0] + 
                         longitudinal_dist * base_direction[0] + 
                         current_lateral * (-base_direction[1]))
            waypoint_y = (ego_pos[1] + 
                         longitudinal_dist * base_direction[1] + 
                         current_lateral * base_direction[0])
            waypoint_yaw = ego_yaw
            
            waypoints.append([waypoint_x, waypoint_y, waypoint_yaw])
        
        return waypoints
    
    def generate_velocity_profile(self, waypoints: List[List[float]]) -> List[float]:
        """生成速度规划"""
        velocity_profile = []
        
        for i, waypoint in enumerate(waypoints):
            # 根据曲率调整速度
            if i > 0:
                prev_waypoint = waypoints[i-1]
                dist = self.calculate_distance(waypoint[:2], prev_waypoint[:2])
                if dist > 0:
                    # 计算曲率（简化）
                    if i > 1:
                        prev_prev = waypoints[i-2]
                        angle_change = abs(math.atan2(waypoint[1] - prev_waypoint[1], 
                                                    waypoint[0] - prev_waypoint[0]) - 
                                         math.atan2(prev_waypoint[1] - prev_prev[1], 
                                                   prev_waypoint[0] - prev_prev[0]))
                        curvature = angle_change / dist if dist > 0 else 0
                        speed_factor = max(0.5, 1.0 - curvature * 2.0)
                        velocity_profile.append(self.target_speed * speed_factor)
                    else:
                        velocity_profile.append(self.target_speed)
                else:
                    velocity_profile.append(self.target_speed)
            else:
                velocity_profile.append(self.target_speed)
        
        return velocity_profile
    
    def predict(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """AD算法预测接口"""
        if not self.model_loaded:
            raise RuntimeError("AD模型尚未加载完成")
        
        # 模拟处理时间
        time.sleep(0.02)
        
        # 提取自车信息
        ego_vehicle = obs.get('ego_vehicle', {})
        ego_pose = ego_vehicle.get('pose', np.array([0, 0, 0, 0, 0, 0]))
        ego_velocity = ego_vehicle.get('velocity', np.array([0, 0, 0]))
        
        # 转换为列表格式
        if isinstance(ego_pose, np.ndarray):
            ego_pos = ego_pose.tolist()
        else:
            ego_pos = ego_pose
            
        if isinstance(ego_velocity, np.ndarray):
            ego_vel = ego_velocity.tolist()
        else:
            ego_vel = ego_velocity
        
        # 提取并解析其他车辆信息
        other_vehicles_raw = obs.get('other_vehicles', [])
        other_vehicles = self.parse_other_vehicles(other_vehicles_raw)
        
        print(f"🎯 自车位置: ({ego_pos[0]:.1f}, {ego_pos[1]:.1f}), "
              f"速度: {math.sqrt(ego_vel[0]**2 + ego_vel[1]**2):.1f} m/s")
        print(f"🚙 检测到 {len(other_vehicles)} 辆有效的其他车辆")
        
        # 生成避障轨迹
        waypoints = self.generate_avoidance_waypoints(ego_pos, ego_vel, other_vehicles)
        
        # 生成速度规划
        velocity_profile = self.generate_velocity_profile(waypoints)
        
        # 构建规划结果
        plan_result = {
            'waypoints': [[wp[0], wp[1], wp[2]] for wp in waypoints],
            'target_speed': self.target_speed,
            'velocity_profile': velocity_profile,
            'timestamp': info.get('timestamp', time.time()),
            'planning_horizon': self.planning_horizon,
            'algorithm': 'GeometricAvoidance',
            'obstacles_detected': len(other_vehicles),
            'avoidance_active': any(abs(wp[1] - ego_pos[1]) > 0.5 for wp in waypoints)
        }
        
        # 计算置信度（基于障碍物距离和数量）
        if other_vehicles:
            distances = []
            for vehicle in other_vehicles:
                dist = self.calculate_distance(ego_pos[:2], vehicle['pose'][:2])
                distances.append(dist)
            
            min_distance = min(distances)
            confidence = max(0.3, min(0.95, min_distance / self.safety_distance))
        else:
            confidence = 0.95
        
        print(f"📋 生成 {len(waypoints)} 个路径点，置信度: {confidence:.3f}")
        
        return plan_result, confidence
