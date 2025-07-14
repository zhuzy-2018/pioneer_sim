# communications/generate_proto.py
import subprocess
import os

def generate_protobuf():
    """生成protobuf文件"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    proto_file = os.path.join(current_dir, "pioneer_sim.proto")
    output_dir = os.path.join(current_dir, "protos")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成Python代码
    cmd = [
        "python", "-m", "grpc_tools.protoc",
        f"--proto_path={current_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        proto_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Protobuf文件生成成功")
        
        # 创建__init__.py
        init_file = os.path.join(output_dir, "__init__.py")
        with open(init_file, "w") as f:
            f.write("# Generated protobuf files\n")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 生成protobuf文件失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    generate_protobuf()
    