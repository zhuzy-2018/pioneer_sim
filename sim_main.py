# simulation/sim_main.py
import sys
import os
import time
import numpy as np

# 添加路径
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '..', 'communications'))
sys.path.append(os.path.join(current_dir, '..', 'sim_envs'))

from communications.env_client import SimulationClient
import gymnasium

# 注册环境
try:
    import pioneer_ad_env
    print("✅ pioneer_ad_env 导入成功")
except ImportError as e:
    print(f"❌ pioneer_ad_env 导入失败: {e}")
    print("尝试手动注册环境...")
    
    # 手动注册环境
    from gymnasium.envs.registration import register
    register(
        id='pioneer_ad_env/ADEnv-v0',
        entry_point='pioneer_ad_env.envs:ADEnv',
    )
    print("✅ 环境手动注册成功")

def create_gym_env_with_grpc(headless: bool = False):
    """使用gRPC的SUMO仿真环境"""
    try:
        # 创建环境
        env = gymnasium.make('pioneer_ad_env/ADEnv-v0', render_mode='human', headless=headless)
        print("✅ SUMO环境创建成功")
        
        with SimulationClient() as client:
            print("✅ gRPC客户端连接成功")
            
            # 重置环境
            print("🔄 重置环境中...")
            obs, info = env.reset()
            print(f"✅ 环境重置完成")
            # 使用简化后的观测格式
            print(f"初始状态: 位置({obs['ego_position'][0]:.1f}, {obs['ego_position'][1]:.1f})")
            print(f"初始速度: {obs['ego_speed'][0]:.1f} m/s")
            print(f"场景车辆数量: {obs['num_vehicles'][0]}")
            
            done = False
            step_count = 0
            total_reward = 0.0
            
            print('🎮 开始SUMO gRPC仿真')
            
            while not done and step_count < 1000:
                try:
                    # 发送简化的观测数据到AD算法
                    plan_result = client.send_observation(obs, info, step_count=step_count)
                    
                    if plan_result is not None:
                        # 解析规划结果并转换为简化动作格式
                        waypoints = plan_result.get('waypoints', [])
                        target_speed = plan_result.get('target_speed', 15.0)
                        
                        # 从waypoints中提取第一个目标点
                        if isinstance(waypoints, list) and len(waypoints) > 0:
                            first_waypoint = waypoints[0]
                            if isinstance(first_waypoint, (list, tuple)) and len(first_waypoint) >= 2:
                                target_x = float(first_waypoint[0])
                                target_y = float(first_waypoint[1])
                            else:
                                # 默认目标：当前位置前方10米
                                target_x = obs['ego_position'][0] + 10.0
                                target_y = obs['ego_position'][1]
                        else:
                            # 默认目标：当前位置前方10米
                            target_x = obs['ego_position'][0] + 10.0
                            target_y = obs['ego_position'][1]
                        
                        # 确保target_speed格式正确
                        if isinstance(target_speed, (list, np.ndarray)):
                            target_speed_val = float(target_speed[0])
                        else:
                            target_speed_val = float(target_speed)
                        
                        # 简化的动作格式
                        action = {
                            'target_x': np.array([target_x], dtype=np.float32),
                            'target_y': np.array([target_y], dtype=np.float32), 
                            'target_speed': np.array([target_speed_val], dtype=np.float32)
                        }
                        
                        if step_count % 50 == 0:
                            print(f"Step {step_count}: 目标点({target_x:.1f}, {target_y:.1f}), "
                                  f"目标速度: {target_speed_val:.1f} m/s")
                        
                    else:
                        print(f"⚠️ Step {step_count}: AD算法无响应，使用默认动作")
                        # 默认直行动作
                        action = {
                            'target_x': np.array([obs['ego_position'][0] + 10.0], dtype=np.float32),
                            'target_y': np.array([obs['ego_position'][1]], dtype=np.float32),
                            'target_speed': np.array([15.0], dtype=np.float32)
                        }
                    
                    # 执行动作
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    step_count += 1
                    
                    # 检查终止条件
                    if terminated:
                        print(f"🛑 仿真终止: {info.get('termination_reason', '未知原因')}")
                    elif truncated:
                        print(f"⏰ 仿真截断: 达到最大步数")
                    
                    # 渲染（Rerun在环境内部处理）
                    if step_count % 10 == 0:
                        env.render()
                    
                    # 控制仿真速度
                    time.sleep(0.02)  # 50 FPS
                    
                    # 定期输出统计信息
                    if step_count % 100 == 0:
                        avg_reward = total_reward / step_count
                        current_pos = obs['ego_position']
                        current_speed = obs['ego_speed'][0]
                        print(f"📊 Step {step_count}: 位置({current_pos[0]:.1f}, {current_pos[1]:.1f}), "
                              f"速度: {current_speed:.1f} m/s, 平均奖励: {avg_reward:.3f}")
                        
                        # 检查服务健康状态
                        if not client.check_health():
                            print("❌ AD服务不健康，结束仿真")
                            break
                    
                    # 检查SUMO连接状态
                    if not info.get('sumo_connected', False):
                        print("❌ SUMO连接断开，结束仿真")
                        break
                    
                except KeyboardInterrupt:
                    print("\n🛑 用户中断仿真")
                    break
                except Exception as e:
                    print(f"⚠️ 仿真步骤出错: {e}")
                    continue
            
            # 仿真结束统计
            final_avg_reward = total_reward / max(step_count, 1)
            print(f"\n📈 仿真统计:")
            print(f"   总步数: {step_count}")
            print(f"   总奖励: {total_reward:.2f}")
            print(f"   平均奖励: {final_avg_reward:.3f}")
            print(f"   终止原因: {'正常结束' if done else '用户中断'}")
            
    except Exception as e:
        print(f"❌ 环境创建或运行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保环境被正确关闭
        try:
            env.close()
            print("✅ 环境已关闭")
        except:
            pass

def create_gym_env_standalone(headless: bool = False):
    """独立运行SUMO环境（不使用gRPC）"""
    try:
        env = gymnasium.make('pioneer_ad_env/ADEnv-v0', render_mode='human', headless=headless)
        print("✅ SUMO环境创建成功（独立模式）")
        
        obs, info = env.reset()
        print(f"初始状态: 位置({obs['ego_position'][0]:.1f}, {obs['ego_position'][1]:.1f})")
        print(f"初始速度: {obs['ego_speed'][0]:.1f} m/s")
        print(f"场景车辆数量: {obs['num_vehicles'][0]}")
        
        done = False
        step_count = 0
        total_reward = 0.0
        
        print('🎮 开始SUMO独立仿真')
        
        while not done and step_count < 500:
            try:
                current_pos = obs['ego_position']
                current_speed = obs['ego_speed'][0]
                current_angle = obs['ego_angle'][0]
                num_vehicles = obs['num_vehicles'][0]
                
                # 简单的规划策略
                # 1. 基础前进距离
                forward_distance = 10.0
                
                # 2. 根据速度调整前进距离
                if current_speed > 0:
                    forward_distance = max(10.0, current_speed * 2.0)  # 2秒的前瞻
                
                # 3. 根据当前角度计算目标点
                target_x = current_pos[0] + forward_distance * np.cos(current_angle)
                target_y = current_pos[1] + forward_distance * np.sin(current_angle)
                
                # 4. 添加一些随机性避免过于单调
                if step_count % 100 == 0:  # 每100步微调一下方向
                    lateral_offset = np.random.uniform(-5, 5)
                    target_x += lateral_offset * np.cos(current_angle + np.pi/2)
                    target_y += lateral_offset * np.sin(current_angle + np.pi/2)
                
                # 5. 简单的速度控制
                target_speed = 15.0
                
                # 如果有其他车辆，简单的避障逻辑
                if num_vehicles > 1:  # 除了ego车还有其他车
                    # 降低速度以防万一
                    target_speed = 12.0
                    
                    # 如果车辆很多，进一步降速
                    if num_vehicles > 5:
                        target_speed = 8.0
                
                # 根据当前速度调整目标速度（平滑加减速）
                if current_speed < target_speed:
                    target_speed = min(target_speed, current_speed + 2.0)  # 最大加速2m/s²
                elif current_speed > target_speed:
                    target_speed = max(target_speed, current_speed - 3.0)  # 最大减速3m/s²
                
                # 确保边界约束
                target_x = np.clip(target_x, 1500, 2000)  # X范围约束
                target_y = np.clip(target_y, 1000, 1200)  # Y范围约束
                target_speed = np.clip(target_speed, 0.0, 30.0)  # 速度约束
                
                # 构建简化动作
                action = {
                    'target_x': np.array([target_x], dtype=np.float32),
                    'target_y': np.array([target_y], dtype=np.float32), 
                    'target_speed': np.array([target_speed], dtype=np.float32)
                }
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step_count += 1
                
                # 定期输出状态
                if step_count % 50 == 0:
                    print(f"Step {step_count}: 位置({current_pos[0]:.1f}, {current_pos[1]:.1f}), "
                          f"速度: {current_speed:.1f} m/s, 目标: ({target_x:.1f}, {target_y:.1f}), "
                          f"奖励: {reward:.3f}, 车辆数: {num_vehicles}")
                
                # 检查异常状态
                if info.get('error'):
                    print(f"⚠️ 检测到错误: {info['error']}")
                    break
                
                time.sleep(0.05)  # 20 FPS
                
            except KeyboardInterrupt:
                print("\n🛑 用户中断仿真")
                break
            except Exception as e:
                print(f"⚠️ 独立仿真步骤出错: {e}")
                # 尝试恢复
                continue
        
        # 统计信息
        final_avg_reward = total_reward / max(step_count, 1)
        print(f"\n📈 独立仿真统计:")
        print(f"   总步数: {step_count}")
        print(f"   总奖励: {total_reward:.2f}")
        print(f"   平均奖励: {final_avg_reward:.3f}")
        print(f"   最终位置: ({obs['ego_position'][0]:.1f}, {obs['ego_position'][1]:.1f})")
        print(f"   最终速度: {obs['ego_speed'][0]:.1f} m/s")
        
    except Exception as e:
        print(f"❌ 独立环境运行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
            print("✅ 环境已关闭")
        except:
            pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SUMO仿真主程序')
    parser.add_argument('--mode', choices=['grpc', 'standalone'], default='standalone',
                        help='运行模式: grpc(使用AD算法服务) 或 standalone(独立运行)')
    parser.add_argument('--headless', default=True, type=bool,
                        help='是否以无头模式运行（模拟GUI）')
    args = parser.parse_args()

    if args.mode == 'grpc':
        print("🚀 启动gRPC模式...")
        create_gym_env_with_grpc(args.headless)
    else:
        print("🚀 启动独立模式...")
        create_gym_env_standalone(args.headless)