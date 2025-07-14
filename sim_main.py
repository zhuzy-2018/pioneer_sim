# simulation/sim_main.py
import sys
import os
import time

# 添加communications目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'communications'))

from communications.env_client import SimulationClient
import pioneer_ad_env
import gymnasium

def create_gym_env_with_grpc():
    """使用gRPC的仿真环境"""
    env = gymnasium.make('pioneer_ad_env/ADEnv-v0', render_mode='human')
    
    with SimulationClient() as client:
        obs, info = env.reset()
        done = False
        step_count = 0
        
        print('🎮 开始gRPC仿真')
        print(f"初始状态: 位置({obs['ego_vehicle']['pose'][0]:.1f}, {obs['ego_vehicle']['pose'][1]:.1f})")
        
        while not done and step_count < 500:
            # 发送观测数据到AD算法
            plan_result = client.send_observation(obs, info, step_count=step_count)
            
            if plan_result is not None:
                print(f"Step {step_count}: 收到规划 - waypoints数量: {len(plan_result.get('waypoints', []))}, "
                      f"目标速度: {plan_result.get('target_speed', 0):.1f}")
                
                # 执行规划动作
                action = {
                    'waypoints': plan_result.get('waypoints', []),
                    'target_speed': plan_result.get('target_speed', 15.0)
                }
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1
                
                # 渲染
                env.render()
                time.sleep(0.05)  # 控制帧率
                
            else:
                print("❌ AD算法无响应，使用默认动作")
                # 默认直行动作
                action = {
                    'waypoints': [[obs['ego_vehicle']['pose'][0] + 10, obs['ego_vehicle']['pose'][1], 0.0]],
                    'target_speed': 15.0
                }
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1
                
            # 定期检查服务健康状态
            if step_count % 100 == 0:
                if not client.check_health():
                    print("❌ AD服务不健康，结束仿真")
                    break
    
    env.close()
    print(f"✅ 仿真完成，共执行 {step_count} 步")

if __name__ == "__main__":
    create_gym_env_with_grpc()
