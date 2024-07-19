import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from tqdm import tqdm

# 创建Highway环境
env = gym.make("merge-v0")
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)