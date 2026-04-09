import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from env import ENV

# [修改] 增加 power 参数
def calculate_delay(action, observation, local_comp, channel_gain, env, upload_rate):
    # 注意：这里的 action 结构预期为 [offload, semantic, resource, power]
    offload = action[0]
    semantic = action[1]
    resource = action[2]
    # power = action[3] # 延迟计算主要依赖 upload_rate，而 upload_rate 已经由 power 算出传入
    
    task_size, computing_density, max_delay, local_energy = observation
    
    if offload == 0:
         delay = task_size * computing_density / local_comp
    else:
        # 语义提取时延 + 传输时延 + MEC计算时延
        # 注意：这里假设 upload_rate 已经是根据当前 power 算出来的
        se_delay = env.alpha * (task_size**env.r) * ((semantic**(-1)) - 1) / local_comp
        tx_delay = semantic * task_size / upload_rate
        mec_delay = semantic * task_size * computing_density / (env.MEC_f * resource)
        
        delay = se_delay + tx_delay + mec_delay
    return delay
        
# [修改] 增加 power 参数，不再使用 env.transmission_power
def calculate_energy(offload, semantic, power, task_size, computing_density, local_comp, channel_gain, env, upload_rate):
    if offload == 0:
        energy = env.κ * task_size * computing_density * local_comp**2
    else:
        # 语义提取能耗
        SE_energy = env.κ * env.alpha * (task_size**env.r) * (semantic**(-env.beta)-1) * local_comp**2
        # 传输能耗：使用当前搜索到的 power
        upload_energy = power * (task_size * semantic) / upload_rate
        energy = SE_energy + upload_energy
    return energy

def greedy_algorithm(env, observation):
    """
    贪婪算法实现（支持功率优化）：
    1. 按优先级顺序处理每个UE
    2. 为当前UE遍历 [语义因子, 发射功率] 组合
    3. 计算满足时延约束所需的最小 MEC 资源
    4. 选择满足所有约束且能耗最低的动作
    """
    UEs = env.UEs
    selected_actions = []
    remaining_resource = 1.0  # 初始可用MEC资源比例 (0.0 - 1.0)
    energy_array = []
    
    # [新增] 定义离散功率等级 (与 env_discrete.py 保持一致)
    power_levels = [0.1, 0.2, 0.3, 0.4, 0.5] 

    # 本地能耗高的优先卸载 (优先级排序)
    ue_priority = sorted(range(UEs), key=lambda i: -observation[i][3])  
    
    for i in ue_priority:
        task_size, computing_density, max_delay, local_energy = observation[i]
        local_comp = env.UE_params[i]['local_comp']
        channel_gain = env.UE_params[i]['channel_gain']
        
        # 默认策略：本地执行 (作为保底)
        best_energy = local_energy
        best_action = [0, 1.0, 0.0, 0.0] # [offload, semantic, resource, power]
        
        # ------------------------------------------------------------------
        # 遍历所有可能的卸载策略 (语义因子 x 发射功率)
        # ------------------------------------------------------------------
        
        # 获取环境定义的语义因子列表 (从 env.actions2 中提取或直接生成)
        # 假设 env.actions2 包含 [offload, semantic]，我们需要提取唯一的 semantic 值
        # 或者直接使用 env 定义的 semantic_factor 范围
        # 这里为了稳健，我们重新生成一下语义范围，或者从 env 获取
        discrete_step = 1.0 / env.k
        semantic_factors = np.arange(discrete_step, 1.0 + discrete_step, discrete_step)
        semantic_threshold = 0.3
        
        for semantic in semantic_factors:
            if semantic <= semantic_threshold: continue # 过滤无效语义因子
            
            for power in power_levels:
                # 1. 计算当前功率下的上行速率
                # 注意：Greedy 通常假设带宽均分。但在逐个决策时，不知道最终几个人卸载。
                # 策略：假设当前用户卸载，W = Total_W / (当前已决策卸载人数 + 1 + 估计剩余人数?)
                # 简化处理：沿用 env.W (假设静态平均划分) 或者 假设 W = env.transmission_bandwidth / UEs
                upload_rate = env.W * math.log2(1 + power * channel_gain / (env.W * env.noise_power))
                
                # 2. 计算固定时延部分 (语义提取 + 传输)
                se_delay = env.alpha * (task_size**env.r) * ((semantic**(-1)) - 1) / local_comp
                tx_delay = semantic * task_size / upload_rate
                
                fixed_delay = se_delay + tx_delay
                
                # 3. 检查是否超时 (即使 MEC 处理时间为 0)
                if fixed_delay >= max_delay:
                    continue # 当前 [语义, 功率] 组合不可行，换下一个
                
                # 4. 计算满足时延约束所需的最小 MEC 资源
                remaining_time_for_mec = max_delay - fixed_delay
                # MEC 计算量 = semantic * task_size * computing_density
                # Required_f = Workload / Time
                # Resource_ratio = Required_f / MEC_Total_F
                required_resource_ratio = (semantic * task_size * computing_density) / (env.MEC_f * remaining_time_for_mec)
                
                # 5. 检查资源是否足够
                if required_resource_ratio > remaining_resource:
                    continue # 资源不足，该组合不可行
                
                # 6. 计算能耗
                current_energy = calculate_energy(1, semantic, power, task_size, computing_density, local_comp, channel_gain, env, upload_rate)
                
                # 7. 更新最优解
                if current_energy < best_energy:
                    best_energy = current_energy
                    # 记录动作: [Offload=1, Semantic, Resource, Power]
                    best_action = [1, semantic, required_resource_ratio, power]

        # ------------------------------------------------------------------
        # 决策结束，更新状态
        # ------------------------------------------------------------------
        
        # 如果选择了卸载，扣除资源
        if best_action[0] == 1:
            remaining_resource -= best_action[2]
            
        energy_array.append(best_energy)
        selected_actions.append(best_action)
        
    total_energy = sum(energy_array)
    return selected_actions, total_energy


if __name__ == '__main__':
    # 初始化环境
    energy_records = []
    
    UEs_range = range(5, 31, 5)

    for UEs in UEs_range:
        # 注意：确保 ENV 类能够正确初始化
        env = ENV(UEs=UEs, MECs=1, k=100) 
        total_energy_list = []
        step_total_energy = []
        
        # 运行 50 个 step 取平均
        for step in range(0, 50):
            observation = env.reset(step)
            # 贪婪算法调用
            greedy_actions, total_energy = greedy_algorithm(env, observation)
            step_total_energy.append(total_energy)

        step_average_energy = sum(step_total_energy) / len(step_total_energy)
        total_energy_list.append(step_average_energy)
        
        # 记录数据
        print(f"UEs: {UEs}, Avg Energy: {step_average_energy:.4f}")
        energy_records.append({'Number of UEs': UEs, 'Average Energy Consumption': step_average_energy})
    
    # 将数据转换成 DataFrame
    df = pd.DataFrame(energy_records)

    # 保存为 CSV 文件
    df.to_csv("greedy_energy_with_power_opt.csv", index=False)
    print("Results saved to greedy_energy_with_power_opt.csv")
    print(df)