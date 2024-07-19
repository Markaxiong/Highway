import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 定义 TensorBoard 日志文件路径
log_dir = './highway_PPO/'

# 初始化 EventAccumulator
event_acc = event_accumulator.EventAccumulator(log_dir)
event_acc.Reload()

# 获取所有可用的标签
print("Available tags:", event_acc.Tags()['scalars'])

# 读取指定标签的数据
try:
    rew_values = event_acc.Scalars('rollout/ep_rew_mean')
except KeyError as e:
    print(f"KeyError: {e}")
    print("Please check the available tags and make sure the specified tag exists in the logs.")
    exit()

# 提取步数（横坐标）和奖励值（纵坐标）
steps = [x.step for x in rew_values]
values = [x.value for x in rew_values]

# 绘制图表
plt.plot(steps, values)
plt.xlabel('Steps')
plt.ylabel('Episode Reward Mean')
plt.title('Training Progress')
plt.show()
