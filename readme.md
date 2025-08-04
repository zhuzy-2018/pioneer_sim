

# Install 
1. install gymnasium
    require python 3.8+ 
    **我在python3.11环境测试。**


```bash
# 更新submodule
git submodule update --init --recursive
```

```bash
pip install -r requirements.txt
```
1. 构建环境
   参考[gymnasium文档](https://gymnasium.org.cn/tutorials/gymnasium_basics/environment_creation/)
```bash
sim_envs/
├── pyproject.toml #打包整个环境并注册
└── zzy_example_env
    ├── envs
    │   ├── __init__.py #注册自定义类
    │   └── test.py #环境代码，继承gymnasium.Env，定义step，reset等方法
    └── __init__.py #注册环境
```
```bash
# 构建gRPC通信文件
python communications/generate_proto.py
```

```bash
# 在sim_envs目录下执行，构建仿真环境
pip install -e sim_envs
```

## docker配置
如果在docker下运行，需要注意几个问题
1. **暴露必要的端口**
> 如果在docker内启动需要暴露以下端口
> 50051 (for gRPC)
> 9090, 9876 (for rerun)
2. **启动虚拟屏幕**
```bash
# 安装必要的包
apt-get update
apt-get install -y xvfb libgl1-mesa-glx libglu1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

# 启动虚拟显示并运行你的程序
xvfb-run -a -s "-screen 0 1024x768x24" python your_script.py
#使用VSCODE运行程序可以使得端口自动转发， 从而在终端机上自动打开网页可视化

```
## 运行
```bash
python ad_main.py
# 另一个终端
python sim_main.py
```
