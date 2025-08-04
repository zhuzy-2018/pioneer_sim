# pioneer_ad_env/envs/ad_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
import time
from typing import Dict, Any, List, Tuple, Optional

# 导入现有的SUMO仿真器类
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'sumo', 'sumo_scripts'))
from start_simulation3 import SumoSimulator, MCPServer
import traci

class ADEnv(gym.Env):
    """简化版基于SUMO交通流的自动驾驶仿真环境"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: Optional[str] = None, 
                 sumocfg_filename: str = "cxg_sumo.sumocfg",
                 sim_step_length: float = 0.1,
                 headless: bool = False):
        super().__init__()
        
        # 环境参数
        self.render_mode = render_mode
        self.max_episode_steps = 2000
        
        # 初始化SUMO仿真器
        self.sumo_sim = SumoSimulator(
            sumocfg_filename=sumocfg_filename,
            sim_step_length=sim_step_length,
            sim_delay=500,
            headless=headless
        )
        
        # 初始化MCP服务器
        self.mcp_server = MCPServer()
        self._register_mcp_functions()
        
        # 状态变量
        self.step_count = 0
        self.ego_vehicle_id = "ego_vehicle"
        
        # # 初始化Rerun可视化
        # if render_mode == "human":
        #     self._init_rerun()
        
        # 简化的动作空间 - 只保留目标位置和速度
        self.action_space = spaces.Dict({
            'target_x': spaces.Box(low=1500, high=2000, shape=(1,), dtype=np.float32),
            'target_y': spaces.Box(low=1000, high=1200, shape=(1,), dtype=np.float32),
            'target_speed': spaces.Box(low=0.0, high=30.0, shape=(1,), dtype=np.float32)
        })
        
        # 简化的观测空间 - 只保留基本信息
        self.observation_space = spaces.Dict({
            'ego_position': spaces.Box(low=-2000, high=2000, shape=(2,), dtype=np.float32),
            'ego_speed': spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32),
            'ego_angle': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            'num_vehicles': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            'step_count': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        })
    
    
    def _register_mcp_functions(self):
        """注册MCP服务器函数"""
        self.mcp_server.register_function(
            "start_simulation", 
            self.sumo_sim.start_simulation, 
            "启动SUMO仿真"
        )
        self.mcp_server.register_function(
            "add_ego_vehicle", 
            self.sumo_sim.add_ego_vehicle, 
            "添加ego车辆"
        )
        self.mcp_server.register_function(
            "add_static_obstacle", 
            self.sumo_sim.add_static_obstacle, 
            "添加静态障碍物"
        )
    
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 重置状态
        self.step_count = 0
        
        # 关闭之前的仿真
        if self.sumo_sim.connected:
            self.sumo_sim.close()
        
        # 启动新的仿真
        if not self.sumo_sim.start_simulation():
            raise RuntimeError("无法启动SUMO仿真")
        
        # 确保SUMO完全就绪
        print("⏳ 等待SUMO完全就绪...")
        time.sleep(1.5)
        
        # 验证SUMO状态
        try:
            version = traci.getVersion()
            views = traci.gui.getIDList()
            print(f"✅ SUMO就绪检查: 版本={version}, GUI视图={views}")
        except Exception as e:
            raise RuntimeError(f"SUMO启动后状态异常: {e}")
        
        # 添加车辆和障碍物
        print("🚗 添加仿真实体...")
        self.mcp_server.call_function("add_ego_vehicle", x=1706, y=1102)
        self.mcp_server.call_function("add_static_obstacle", x=1600, y=1042)
        self.mcp_server.call_function("add_static_obstacle", x=1650, y=1050)
        
        # 额外等待确保所有实体添加完成
        time.sleep(0.5)
        
        # 验证车辆添加是否成功
        try:
            vehicle_list = traci.vehicle.getIDList()
            print(f"✅ 车辆添加验证: {vehicle_list}")
            if "ego_vehicle" not in vehicle_list:
                print("⚠️ ego车辆添加可能失败")
        except Exception as e:
            print(f"⚠️ 车辆验证失败: {e}")
        
        # 执行几步仿真确保场景稳定
        print("🔄 执行初始化步骤...")
        for _ in range(3):
            try:
                traci.simulationStep()
                time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ 初始化步骤失败: {e}")
                break
        
        # 获取初始观测
        obs = self._get_observation()
        info = self._get_info()
        
        print("✅ 环境重置完成")
        return obs, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """获取当前观测（简化版）"""
        if not self.sumo_sim.connected or self.ego_vehicle_id not in traci.vehicle.getIDList():
            return self._get_default_observation()
        
        try:
            # 获取ego车辆基本状态
            ego_pos = traci.vehicle.getPosition(self.ego_vehicle_id)
            ego_speed = traci.vehicle.getSpeed(self.ego_vehicle_id)
            ego_angle = math.radians(traci.vehicle.getAngle(self.ego_vehicle_id))
            
            # 获取车辆总数
            num_vehicles = len(traci.vehicle.getIDList())
            
            obs = {
                'ego_position': np.array([ego_pos[0], ego_pos[1]], dtype=np.float32),
                'ego_speed': np.array([ego_speed], dtype=np.float32),
                'ego_angle': np.array([ego_angle], dtype=np.float32),
                'num_vehicles': np.array([num_vehicles], dtype=np.int32),
                'step_count': np.array([self.step_count], dtype=np.int32)
            }
            
            return obs
            
        except Exception as e:
            print(f"⚠️ 获取观测失败: {e}")
            return self._get_default_observation()
    
    def _get_default_observation(self) -> Dict[str, Any]:
        """获取默认观测（当SUMO未连接时）"""
        return {
            'ego_position': np.zeros(2, dtype=np.float32),
            'ego_speed': np.zeros(1, dtype=np.float32),
            'ego_angle': np.zeros(1, dtype=np.float32),
            'num_vehicles': np.array([0], dtype=np.int32),
            'step_count': np.array([self.step_count], dtype=np.int32)
        }
    
    def step(self, action: Dict[str, Any]):
        """执行一步仿真（简化版）"""
        if not self.sumo_sim.connected:
            obs = self._get_default_observation()
            return obs, 0.0, True, False, {"error": "SUMO未连接"}
        
        try:
            # 解析动作（简化版）
            target_x = float(action.get('target_x', [1706])[0])
            target_y = float(action.get('target_y', [1102])[0])
            target_speed = float(action.get('target_speed', [15.0])[0])
            
            # 构建控制命令
            control_command = None
            if self.ego_vehicle_id in traci.vehicle.getIDList():
                current_pos = traci.vehicle.getPosition(self.ego_vehicle_id)
                
                # 计算目标角度
                dx = target_x - current_pos[0]
                dy = target_y - current_pos[1]
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    target_angle = math.degrees(math.atan2(dy, dx))
                else:
                    target_angle = traci.vehicle.getAngle(self.ego_vehicle_id)
                
                control_command = {
                    "x": target_x,
                    "y": target_y,
                    "angle": target_angle,
                    "speed": target_speed
                }
            
            # 执行SUMO仿真步骤
            if not self.sumo_sim.step():
                obs = self._get_observation()
                return obs, 0.0, True, False, {"info": "仿真自然结束"}
            
            self.step_count += 1
            
            # # 可视化
            # if self.render_mode == "human":
            #     try:

            #     except Exception as e:
            #         print(f"⚠️ 第{self.step_count}步可视化错误: {e}")
            
            # 简化的奖励 - 暂时给固定值
            reward = 0.1
            
            # 简化的终止条件
            terminated = False
            truncated = self.step_count >= self.max_episode_steps
            
            # 获取观测和信息
            obs = self._get_observation()
            info = self._get_info()
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"⚠️ 仿真步骤出错: {e}")
            obs = self._get_observation()
            return obs, 0.0, False, False, {"error": str(e)}
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息（简化版）"""
        info = {
            'step_time': self.sumo_sim.sim_step_length,
            'sumo_connected': self.sumo_sim.connected,
            'sumo_running': self.sumo_sim.running
        }
        
        if self.sumo_sim.connected and self.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                info.update({
                    'ego_speed': traci.vehicle.getSpeed(self.ego_vehicle_id),
                    'ego_position': traci.vehicle.getPosition(self.ego_vehicle_id),
                    'ego_angle': traci.vehicle.getAngle(self.ego_vehicle_id),
                    'num_vehicles': len(traci.vehicle.getIDList())
                })
            except:
                pass
        
        return info
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            # Rerun可视化在step()中已经处理
            pass
        elif self.render_mode == "rgb_array":
            if self.sumo_sim.connected:
                image, _, _, _ = self.sumo_sim.capture_scene_data()
                return image
        
        return None
    
    def close(self):
        """关闭环境"""
        print("🧹 关闭AD环境...")
        
        # 关闭SUMO仿真
        if hasattr(self, 'sumo_sim'):
            self.sumo_sim.close()
        
        
        print("✅ AD环境已关闭")