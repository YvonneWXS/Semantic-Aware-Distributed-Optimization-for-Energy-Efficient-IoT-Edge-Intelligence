import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set_theme(style="whitegrid")

# 读取两个CSV文件
file1 = "results/MyEnv/MyEnv/mappo/check/3agents/IPPO/episode_rewards_and_energy_mappo.csv"
file2 = "results/MyEnv/MyEnv/mappo/check/3agents/MAPPO/episode_rewards_and_energy_mappo.csv"


df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# # 计算滑动平均（平滑曲线）
# def smooth(y, box_pts=1):
#     box = np.ones(box_pts)/box_pts


#     return np.convolve(y, box, mode='same')

# 创建两个子图
fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 12))

# 画奖励曲线
ax1.plot(df1["Episode"],df1["Reward"], label="IPPO", color="purple")
ax1.plot(df2["Episode"], df2["Reward"], label="MAPPO", color="blue")

# # 画奖励曲线的标准差阴影
# ax1.fill_between(df1["Episode"], 
#                  smooth(df1["Reward"] - df1["Reward"].std()), 

#                  smooth(df1["Reward"] + df1["Reward"].std()), 
#                  color="orange", alpha=0.3)

# ax1.fill_between(df2["Episode"], 
#                  smooth(df2["Reward"] - df2["Reward"].std()), 
#                  smooth(df2["Reward"] + df2["Reward"].std()), 
#                  color="blue", alpha=0.3)

# 设置奖励曲线的标签和图例
ax1.set_xlabel("Episode", fontsize=14)
ax1.set_ylabel("Rewards", fontsize=14)
ax1.legend(loc="best", fontsize=12)

# 画能耗曲线
ax2.plot(df1["Episode"], df1["Energy"], label="IPPO", color="purple")
ax2.plot(df2["Episode"], df2["Energy"], label="MAPPO", color="blue")

# # 画能耗曲线的标准差阴影
# ax2.fill_between(df1["Episode"], 
#                  smooth(df1["Energy"] - df1["Energy"].std()), 
#                  smooth(df1["Energy"] + df1["Energy"].std()), 
#                  color="orange", alpha=0.3)

# ax2.fill_between(df2["Episode"], 
#                  smooth(df2["Energy"] - df2["Energy"].std()), 
#                  smooth(df2["Energy"] + df2["Energy"].std()), 
#                  color="blue", alpha=0.3)

# 设置能耗曲线的标签和图例
ax2.set_xlabel("Episode", fontsize=14)
ax2.set_ylabel("Episode Everage Energy", fontsize=14)
ax2.legend(loc="best", fontsize=12)


#绘制违反资源约束的比例曲线
ax3.plot(df1["Episode"], df1["Violation_Rate"], label="IPPO", color="purple")
ax3.plot(df2["Episode"], df2["Violation_Rate"], label="MAPPO", color="blue")

ax3.set_xlabel("Episode", fontsize=14)
ax3.set_ylabel("Violation_Rate", fontsize=14)
ax3.legend(loc="best", fontsize=12)

# 调整子图之间的间距
plt.tight_layout()

save_path = "results\MyEnv\MyEnv\mappo\check\\3agents\compare.jpg"
plt.savefig(save_path)

# 显示图形
plt.show()

