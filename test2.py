import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import time

# 创建Highway环境
env = gym.make("merge-v0", render_mode="rgb_array")

# 加载训练好的模型
model = PPO.load("./highway_PPO/model")

# 初始化计数器
collision_count = 0
num_episodes = 100

for episode in range(num_episodes):
    done = truncated = False
    obs, info = env.reset()
    episode_collision = False  # 用于标记本回合是否发生碰撞
    
    while not (done or truncated):
        # 打印或记录观测数据
        #print(f"Observation: {obs}")
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        env.render()
        time.sleep(0.2)  # 每帧渲染后延迟0.2秒

        # 检查碰撞信息
        if info.get("crashed", False):
            episode_collision = True

    if episode_collision:
        collision_count += 1

    print(f"Episode {episode + 1}/{num_episodes} - Total collisions so far: {collision_count}")

# 计算碰撞百分比
collision_percentage = (collision_count / num_episodes) * 100
print(f"Total collisions after {num_episodes} episodes: {collision_count}")
print(f"Collision percentage: {collision_percentage:.2f}%")
