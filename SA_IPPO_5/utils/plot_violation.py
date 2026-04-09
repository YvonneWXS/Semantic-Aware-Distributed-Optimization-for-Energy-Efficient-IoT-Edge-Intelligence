import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_violation_curve(csv_path, output_name="violation_curve.png"):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 设置风格
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制 "User Violation Probability" (每用户超时概率)
    # 也可以绘制 "System Violation Rate" (系统违约率)
    
    # 1. 绘制原始曲线 (透明度高)
    ax.plot(df['Episode'], df['User_Violation_Prob'], color='red', alpha=0.3, label='Raw User Violation Prob')
    
    # 2. 绘制平滑曲线 (移动平均)
    window_size = 50  # 平滑窗口大小
    df['Smooth_Prob'] = df['User_Violation_Prob'].rolling(window=window_size).mean()
    ax.plot(df['Episode'], df['Smooth_Prob'], color='darkred', linewidth=2, label=f'Smoothed (MA={window_size})')
    
    # # 3. 绘制累计违约次数曲线
    # df['Cumulative_Violations'] = df['User_Violation_Prob'].cumsum()
    # ax.plot(df['Episode'], df['Cumulative_Violations'], color='blue', label='Cumulative Violations')

    ax.set_xlabel('Training Episodes', fontsize=12)
    ax.set_ylabel('User Timeout Probability', fontsize=12)
    ax.set_title('Constraint Violation Evolution during Training', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"Plot saved to {output_name}")

if __name__ == "__main__":
    # 替换路径
    csv_path = "../results/MyEnv/MyEnv/mappo/check/run2/eval_episode_info.csv" 
    plot_violation_curve(csv_path)