"""
Author: zzy-office zhuzy_2013@163.com
Date: 2025-07-14 14:49:10
LastEditors: zzy-office zhuzy_2013@163.com
LastEditTime: 2025-07-14 15:00:00
FilePath: /pioneer_sim/ad_algorithm/ad_main.py
Description: AD算法主程序入口

Copyright (c) 2025 by zzy-office, All Rights Reserved.
"""

import sys
import os

# 添加communications目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'communications'))

from communications.ad_server import serve

def main():
    """AD算法服务主程序"""
    print("🚀 启动AD算法服务...")
    
    # 从环境变量获取端口，默认50051
    port = int(os.getenv('GRPC_PORT', '50051'))
    
    try:
        serve(port)
    except KeyboardInterrupt:
        print("\n👋 AD算法服务已停止")
    except Exception as e:
        print(f"❌ 服务启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
