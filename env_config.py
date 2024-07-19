import gymnasium as gym
from matplotlib import pyplot as plt

# 初始化环境
env = gym.make('highway-v0', render_mode='rgb_array')
env.reset()

# 创建一个列表来存储每一步的渲染结果
frames = []

# 运行环境并存储多个画面
for _ in range(100):  # 这里可以设置想要运行的步数
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    frames.append(env.render())

# 使用matplotlib显示多个画面
fig, axes = plt.subplots(1, len(frames), figsize=(20, 5))

for i, frame in enumerate(frames):
    axes[i].imshow(frame)
    axes[i].axis('off')

plt.show()
