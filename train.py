import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from tqdm import tqdm

env_config = {
    "observation": {
        "type": "TimeToCollision",
        #"vehicles_count": 7,  # 确保观测空间中的车辆数量一致
        #"features": ["presence", "x", "y", "vx", "vy"],
        #"normalize": True
    },
    # 其他配置参数
}
env = gym.make('merge-v0')
obs, info = env.reset()

#obs, info = env.reset()
n_cpu = 6
batch_size =64
# 创建PPO模型
model = PPO(
    'MlpPolicy',  # 使用多层感知机（MLP）策略
    env,
    policy_kwargs=dict(net_arch=[256, 256]),  # 定义策略网络结构
    learning_rate=5e-4,  # 学习率
    n_steps=batch_size * 12 // n_cpu,
    batch_size=batch_size,
    gamma=0.8,  # 折扣因子
    verbose=1,  # 输出训练信息
    tensorboard_log="./highway_PPO/1"  # TensorBoard日志路径
)

# 定义总的训练时间步数
total_timesteps = int(100000)
# 定义每次更新的时间步数
update_timesteps = model.n_steps

# 使用tqdm创建进度条
for _ in tqdm(range(0, total_timesteps, update_timesteps), desc="Training Progress", unit="timesteps"):
    model.learn(total_timesteps=update_timesteps, reset_num_timesteps=False)

# 保存模型
model.save("./highway_PPO/model")



