

# Install 
1. install gymnasium
    require python 3.8+ 
    **我在python3.11环境测试。**
```bash
pip install gymnasium pygame
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
pip install -e .
```
## 运行
```bash
python sim_main.py
```
