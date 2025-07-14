# communications/ad_server.py
"""
Author: zzy-office zhuzy_2013@163.com
Date: 2025-07-14 15:00:00
LastEditors: zzy-office zhuzy_2013@163.com
LastEditTime: 2025-07-14 15:00:00
FilePath: /pioneer_sim/communications/ad_server.py
Description: AD算法gRPC服务端

Copyright (c) 2025 by zzy-office, All Rights Reserved.
"""

import grpc
from concurrent import futures
import json
import time
import signal
import sys
import os
import threading
from typing import Dict, Any

# 添加protos目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'protos'))

# 添加ad_algorithm目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ad_algorithm'))

import pioneer_sim_pb2
import pioneer_sim_pb2_grpc
from ad_model import ADModel

class ADAlgorithmService(pioneer_sim_pb2_grpc.SimulationServiceServicer):
    """AD算法gRPC服务"""
    
    def __init__(self):
        self.ad_model = ADModel()
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0
        }
    
    def HealthCheck(self, request, context):
        """健康检查"""
        model_info = self.ad_model.get_model_info()
        return pioneer_sim_pb2.HealthResponse(
            healthy=model_info['loaded'],
            status="ready" if model_info['loaded'] else "loading",
            version=model_info['version']
        )
    
    def SendObservation(self, request, context):
        """处理观测数据并返回规划结果"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        # 检查模型是否加载
        if not self.ad_model.is_model_ready():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details('AD模型尚未加载完成')
            self.stats['failed_requests'] += 1
            return pioneer_sim_pb2.PlanResponse(
                success=False,
                error_message="AD模型尚未加载完成"
            )
        
        try:
            # 解析输入数据
            obs_dict = json.loads(request.obs_json)
            info_dict = json.loads(request.info_json)
            
            print(f"📥 收到观测数据 - Step: {request.step_count}, "
                  f"时间戳: {request.timestamp}")
            
            # 调用AD模型进行预测
            plan_result, confidence = self.ad_model.predict(obs_dict, info_dict)
            
            # 更新统计信息
            processing_time = time.time() - start_time
            self.stats['successful_requests'] += 1
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['successful_requests'] - 1) + 
                 processing_time) / self.stats['successful_requests']
            )
            
            print(f"📤 规划完成 - 耗时: {processing_time:.3f}s, "
                  f"置信度: {confidence:.3f}")
            
            # 返回结果
            return pioneer_sim_pb2.PlanResponse(
                success=True,
                plan_json=json.dumps(plan_result),
                confidence=confidence,
                error_message=""
            )
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON解析错误: {e}"
            print(f"❌ {error_msg}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(error_msg)
            self.stats['failed_requests'] += 1
            return pioneer_sim_pb2.PlanResponse(
                success=False,
                error_message=error_msg
            )
            
        except Exception as e:
            error_msg = f"AD算法处理错误: {e}"
            print(f"❌ {error_msg}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            self.stats['failed_requests'] += 1
            return pioneer_sim_pb2.PlanResponse(
                success=False,
                error_message=error_msg
            )
    
    def print_stats(self):
        """打印统计信息"""
        print(f"\n📊 AD服务统计:")
        print(f"  总请求数: {self.stats['total_requests']}")
        print(f"  成功请求: {self.stats['successful_requests']}")
        print(f"  失败请求: {self.stats['failed_requests']}")
        print(f"  平均处理时间: {self.stats['avg_processing_time']:.3f}s")

def serve(port: int = 50051):
    """启动gRPC服务"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ad_service = ADAlgorithmService()
    
    pioneer_sim_pb2_grpc.add_SimulationServiceServicer_to_server(
        ad_service, server
    )
    
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    print(f"🚀 AD算法gRPC服务启动在端口 {port}")
    model_info = ad_service.ad_model.get_model_info()
    print(f"🎯 模型参数: 安全距离={model_info['safety_distance']}m, "
          f"规划时域={model_info['planning_horizon']}s")
    
    # 优雅关闭处理
    def signal_handler(sig, frame):
        print("\n🛑 正在关闭AD服务...")
        ad_service.print_stats()
        server.stop(0)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        signal_handler(None, None)
