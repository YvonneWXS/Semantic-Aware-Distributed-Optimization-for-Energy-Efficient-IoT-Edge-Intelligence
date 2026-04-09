import matplotlib.pyplot as plt
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('/root/autodl-tmp/MADDPG_adjusted_0310/test.csv')

# 创建一个图形来绘制 Energy 和 Reward 曲线
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 绘制 Reward 曲线
ax1.plot(df['Episode'], df['MADDPG_Reward'], label='MADDPG Reward', color='b', marker='o')
ax1.plot(df['Episode'], df['DQN_Reward'], label='DQN Reward', color='g', marker='x')
ax1.plot(df['Episode'], df['D3QN_Reward'], label='D3QN Reward', color='r', marker='s')
ax1.set_title('Reward vs Episode')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.legend()

# 绘制 Energy 曲线
ax2.plot(df['Episode'], df['MADDPG_Energy'], label='MADDPG Energy', color='b', marker='o')
ax2.plot(df['Episode'], df['DQN_Energy'], label='DQN Energy', color='g', marker='x')
ax2.plot(df['Episode'], df['D3QN_Energy'], label='D3QN Energy', color='r', marker='s')
ax2.set_title('Energy vs Episode')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Energy')
ax2.legend()

# 显示图形
plt.tight_layout()
plt.show()
