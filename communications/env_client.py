# communications/env_client.py
import grpc
import json
import time
import os
import sys
from typing import Dict, Any, Optional

# 添加protos目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'protos'))

import pioneer_sim_pb2
import pioneer_sim_pb2_grpc

class SimulationClient:
    """仿真环境gRPC客户端"""
    
    def __init__(self, host: str = None, port: int = None, timeout: int = 60):
        # 从环境变量获取服务地址
        self.host = host or os.getenv('AD_SERVICE_HOST', 'localhost')
        self.port = port or int(os.getenv('AD_SERVICE_PORT', '50051'))
        self.timeout = timeout
        
        # 创建gRPC连接
        self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')
        self.stub = pioneer_sim_pb2_grpc.SimulationServiceStub(self.channel)
        
        # 等待服务启动
        self.wait_for_service()
        
        print(f"✅ 连接到AD服务: {self.host}:{self.port}")
    
    def wait_for_service(self) -> None:
        """等待AD服务可用"""
        print(f"🔄 等待AD服务启动 ({self.host}:{self.port})...")
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            try:
                # 发送健康检查
                health_request = pioneer_sim_pb2.HealthRequest(
                    service_name="simulation_client"
                )
                response = self.stub.HealthCheck(health_request, timeout=5)
                
                if response.healthy:
                    print(f"✅ AD服务已就绪 - 状态: {response.status}")
                    return
                    
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    time.sleep(1)
                    continue
                else:
                    print(f"⚠️ 健康检查失败: {e}")
                    time.sleep(1)
                    continue
        
        raise ConnectionError(f"❌ AD服务启动超时 ({self.timeout}s)")
    
    def send_observation(self, obs_dict: Dict[str, Any], info_dict: Dict[str, Any], 
                        step_count: int = 0) -> Optional[Dict[str, Any]]:
        """发送观测数据到AD算法"""
        try:
            # 创建请求
            request = pioneer_sim_pb2.ObservationRequest(
                obs_json=json.dumps(obs_dict, default=self._json_serializer),
                info_json=json.dumps(info_dict, default=self._json_serializer),
                timestamp=int(time.time() * 1000),  # 毫秒时间戳
                step_count=step_count
            )
            
            # 发送请求
            response = self.stub.SendObservation(request, timeout=10)
            
            if response.success:
                plan_data = json.loads(response.plan_json)
                plan_data['confidence'] = response.confidence
                return plan_data
            else:
                print(f"❌ AD算法处理失败: {response.error_message}")
                return None
                
        except grpc.RpcError as e:
            print(f"❌ gRPC调用失败: {e.code()} - {e.details()}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            return None
    
    def check_health(self) -> bool:
        """检查AD服务健康状态"""
        try:
            request = pioneer_sim_pb2.HealthRequest(service_name="simulation_client")
            response = self.stub.HealthCheck(request, timeout=5)
            return response.healthy
        except grpc.RpcError:
            return False
    
    def close(self) -> None:
        """关闭连接"""
        if self.channel:
            self.channel.close()
            print("🔌 gRPC连接已关闭")
    
    @staticmethod
    def _json_serializer(obj):
        """JSON序列化辅助函数，处理numpy数组等"""
        if hasattr(obj, 'tolist'):  # numpy数组
            return obj.tolist()
        elif hasattr(obj, '__dict__'):  # 对象
            return obj.__dict__
        else:
            return str(obj)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# 使用示例
if __name__ == "__main__":
    # 测试客户端
    with SimulationClient() as client:
        # 模拟观测数据
        obs_data = {
            'camera_front': 'base64_image_data',
            'lidar_points': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            'speed': 30.5,
            'steering_angle': 0.1
        }
        
        info_data = {
            'ego_pos': [100.0, 200.0, 0.0],
            'timestamp': time.time(),
            'collision': False,
            'traffic_light': 'green'
        }
        
        # 发送数据
        result = client.send_observation(obs_data, info_data, step_count=1)
        print("📊 收到规划结果:", result)
