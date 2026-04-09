import numpy as np
import pandas as pd

class Env:
    def __init__(self):
        np.random.seed(47)
        self.agent_num = 5
        self.obs_dim = 3

        # 基础单位
        self.Hz = 1
        self.kHz = 1000 * self.Hz
        self.mHz = 1000 * self.kHz
        self.GHz = 1000 * self.mHz

        # 数据大小
        self.bit = 1
        self.B = 8 * self.bit
        self.KB = 1024 * self.B
        self.MB = 1024 * self.KB

        # UE 的固定参数
        self.local_comp = np.zeros(self.agent_num)
        self.distance = np.zeros(self.agent_num)
        self.channel_gain = np.zeros(self.agent_num)

        # ---- 生成 UE 的基础参数（与环境一致）----
        for i in range(self.agent_num):
            self.local_comp[i] = np.random.randint(1.5 * self.GHz, 2 * self.GHz)
            self.distance[i] = np.random.uniform(10, 100)
            self.channel_gain[i] = 1e-3 * (1.0 / self.distance[i]) ** 2.5

        # 任务相关变量
        self.task_size = np.zeros(self.agent_num)
        self.computing_density = np.zeros(self.agent_num)
        self.local_delay = np.zeros(self.agent_num)
        self.max_delay = np.zeros(self.agent_num)

    # -------- 用于生成“表格”的 reset_for_step ----------
    def reset_for_step(self, step):
        np.random.seed(1 + step)
        rows = []

        for ue_id in range(self.agent_num):
            # 随机生成
            task_size = np.random.randint(1.5 * self.MB, 2 * self.MB)
            computing_density = np.random.uniform(300, 500)
            local_delay = task_size * computing_density / self.local_comp[ue_id]
            max_delay = np.random.uniform(local_delay, 2 * local_delay)

            rows.append({
                "step": step,
                "ue_id": ue_id,
                "task_size": float(task_size),
                "computing_density": float(computing_density),
                "max_delay": float(max_delay)
            })

        return rows


# ---------------------- 生成表格 ----------------------

env = Env()

def generate_table(num_steps=50):
    all_rows = []
    for step in range(num_steps):
        all_rows.extend(env.reset_for_step(step))
    return pd.DataFrame(all_rows)

df = generate_table()
print(df)
df.to_csv("new_obs_{}.csv".format(env.agent_num), index=False)