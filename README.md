```python
import gymnasium as gym #世界库
from stable_baselines3 import PPO #引入ppo算法
import os
```

# 1. 创建环境
# CartPole-v1 是经典的倒立摆任务
```python
env = gym.make("CartPole-v1")
```
#基于公式的仿真，输出环境ID为："CartPole-v1"的物理环境
#包括：[小车位置, 小车速度, 杆子角度, 杆顶速度]

# 2. 定义模型
# 我们使用 PPO 算法，MlpPolicy 表示使用全连接神经网络
# device="cpu"：对于这种极小的模型，M4 Max 的 CPU 处理速度极快，无需调用 GPU
```python
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
```
#多层感知机，与环境交互（环境输出：小车位置、小车速度、杆子角度、杆顶速度），输出日志，使用cpu

```python
print("---------- 🚀 开始训练 ----------")
```

# 3. 开始训练
# total_timesteps=10000：让 AI 尝试玩 10,000 步
# 在 M4 Max 上，这应该瞬间完成
model.learn(total_timesteps=20000)

```python
print("---------- ✅ 训练结束 ----------")
```

# 4. 保存模型
# 保存为 ppo_cartpole.zip
```python
model.save("ppo_cartpole")
print("模型已保存！")
```
