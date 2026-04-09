import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from env import ENV
import config

# 带宽权重到动作的映射
def get_available_bw_weights():
    """从配置中获取可用的带宽权重"""
    return config.bw_weights

def get_available_power_levels():
    """从配置中获取可用的功率级别"""
    return config.power_list

def dynamic_bandwidth_allocation(bw_weights, total_bandwidth):
    """
    动态带宽分配机制 - 基于离散权重的归一化分配
    参数: bw_weights - 每个UE的带宽权重数组
          total_bandwidth - 总带宽
    返回: 每个需要卸载的UE分配到的实际带宽
    """
    # 计算非零权重的总和，用于归一化
    total_weights = sum(w for w in bw_weights if w > 0)

    # 如果总权重为0，说明没有UE需要卸载，返回0带宽
    if total_weights == 0:
        return [0.0] * len(bw_weights)

    # 归一化分配带宽
    allocated_bandwidths = []
    for i in range(len(bw_weights)):
        if bw_weights[i] > 0:
            # 计算该UE应得的带宽比例
            proportion = bw_weights[i] / total_weights
            # 计算理想带宽
            ideal_bw = proportion * total_bandwidth
            # 进行向下取整量化，模拟RB资源块
            # 这里假设最小资源单位为10kHz
            min_rb_unit = 10 * 10**3  # 10kHz
            quantized_bw = math.floor(ideal_bw / min_rb_unit) * min_rb_unit
            allocated_bandwidths.append(max(0, quantized_bw))
        else:
            # 如果权重为0，则分配0带宽
            allocated_bandwidths.append(0.0)

    return allocated_bandwidths

# 修改计算延迟函数，包含动态带宽
def calculate_delay(action, observation, local_comp, channel_gain, env, assigned_bandwidth):
    # 注意：这里的 action 结构预期为 [offload, semantic, resource, power, bw_weight]
    offload = action[0]
    semantic = action[1]
    resource = action[2]
    power = action[3]

    task_size, computing_density, max_delay, local_energy = observation

    if offload == 0:
         delay = task_size * computing_density / local_comp
    else:
        # 使用分配的带宽计算上行速率
        if assigned_bandwidth <= 0:
            # 如果分配的带宽为0，使用一个小的默认值以避免除零错误
            assigned_bandwidth = 10 * 10**3  # 10kHz

        uplink_rate = assigned_bandwidth * math.log2(1 + power * channel_gain / (assigned_bandwidth * config.noise_power))

        # 语义提取时延 + 传输时延 + MEC计算时延
        se_delay = env.alpha * (task_size**env.r) * ((semantic**(-1)) - 1) / local_comp
        tx_delay = semantic * task_size / uplink_rate
        mec_delay = semantic * task_size * computing_density / (env.MEC_f * resource)

        delay = se_delay + tx_delay + mec_delay
    return delay

# 修改计算能耗函数，包含动态带宽
def calculate_energy(offload, semantic, power, task_size, computing_density, local_comp, channel_gain, env, assigned_bandwidth):
    if offload == 0:
        energy = env.κ * task_size * computing_density * local_comp**2
    else:
        # 语义提取能耗
        SE_energy = env.κ * env.alpha * (task_size**env.r) * (semantic**(-env.beta)-1) * local_comp**2

        # 使用分配的带宽计算上行速率
        if assigned_bandwidth <= 0:
            # 如果分配的带宽为0，使用一个小的默认值以避免除零错误
            assigned_bandwidth = 10 * 10**3  # 10kHz

        uplink_rate = assigned_bandwidth * math.log2(1 + power * channel_gain / (assigned_bandwidth * config.noise_power))
        # 传输能耗：使用当前功率和分配的带宽
        upload_energy = power * (semantic * task_size) / uplink_rate
        energy = SE_energy + upload_energy
    return energy

def greedy_algorithm(env, observation):
    """
    贪婪算法实现（支持功率优化和动态带宽分配）：
    1. 按优先级顺序处理每个UE
    2. 为当前UE遍历 [语义因子, 发射功率, 带宽权重] 组合
    3. 计算满足时延约束所需的最小 MEC 资源
    4. 选择满足所有约束且能耗最低的动作
    """
    UEs = env.UEs
    selected_actions = []
    remaining_resource = 1.0  # 初始可用MEC资源比例 (0.0 - 1.0)
    energy_array = []

    # 从配置获取参数
    power_levels = get_available_power_levels()
    bw_weights = get_available_bw_weights()

    # 本地能耗高的优先卸载 (优先级排序)
    ue_priority = sorted(range(UEs), key=lambda i: -observation[i][3])

    # 首先确定所有UE的决策，以便计算全局带宽分配
    # 这里使用迭代方式处理，但为了简化，我们采用近似贪婪方法
    # 每个UE选择其局部最优决策，然后在环境中评估整体性能

    for i in ue_priority:
        task_size, computing_density, max_delay, local_energy = observation[i]
        local_comp = env.UE_params[i]['local_comp']
        channel_gain = env.UE_params[i]['channel_gain']

        # 默认策略：本地执行 (作为保底)
        best_energy = local_energy
        best_action = [0, 1.0, 0.0, 0.0, 0]  # [offload, semantic, resource, power, bw_weight]

        # ------------------------------------------------------------------
        # 遍历所有可能的卸载策略 (语义因子 x 发射功率 x 带宽权重)
        # ------------------------------------------------------------------

        # 获取环境定义的语义因子列表
        discrete_step = 1.0 / env.k
        semantic_factors = np.arange(discrete_step, 1.0 + discrete_step, discrete_step)
        semantic_threshold = config.min_semantic_factor_list[0]

        for semantic in semantic_factors:
            if semantic <= semantic_threshold:
                continue  # 过滤无效语义因子

            for power in power_levels:
                for bw_weight in bw_weights:
                    # 跳过bw_weight为0的情况，当我们想要卸载时
                    if bw_weight == 0:
                        continue

                    # 需要估算当前决策下的带宽分配
                    # 为了简化，我们使用一个临时的带宽分配策略
                    # 假设目前只有这个UE决定卸载（或至少有这个意图）

                    # 1. 计算当前功率和带宽权重下的上行速率
                    # 由于我们需要预测带宽，暂时假定该UE将获得部分带宽
                    # 使用一个假设的带宽分配（如果只有这个UE传输，则理论上可以获得全部带宽）
                    # 但我们应该考虑实际情况：其他UE也可能在传输
                    # 简化策略：计算在所有可能的bw_weight组合下的潜在能耗，并选择最小的

                    # 由于带宽分配是全局的，我们需要一种方式来估计
                    # 在贪婪算法中，我们可以尝试每个UE的局部最优选择
                    # 并在最终评估时使用全局带宽分配
                    estimated_bandwidth = env.total_bandwidth / UEs  # 一个初始估计值

                    # 如果带宽权重为0，分配一个很小的带宽避免除零
                    if bw_weight == 0:
                        temp_bandwidth = 10 * 10**3  # 10kHz
                    else:
                        # 为了评估目的，假设当前UE单独使用带宽
                        temp_bandwidth = env.total_bandwidth / UEs * (bw_weight / max(bw_weights))

                    uplink_rate = temp_bandwidth * math.log2(1 + power * channel_gain / (temp_bandwidth * config.noise_power))

                    # 2. 计算固定时延部分 (语义提取 + 传输)
                    se_delay = env.alpha * (task_size**env.r) * ((semantic**(-1)) - 1) / local_comp
                    tx_delay = semantic * task_size / uplink_rate

                    fixed_delay = se_delay + tx_delay

                    # 3. 检查是否超时 (即使 MEC 处理时间为 0)
                    if fixed_delay >= max_delay:
                        continue  # 当前 [语义, 功率, bw_weight] 组合不可行，换下一个

                    # 4. 计算满足时延约束所需的最小 MEC 资源
                    remaining_time_for_mec = max_delay - fixed_delay
                    # MEC 计算量 = semantic * task_size * computing_density
                    # Required_f = Workload / Time
                    # Resource_ratio = Required_f / MEC_Total_F
                    if remaining_time_for_mec <= 0:
                        continue  # 时间不够，跳过

                    required_resource_ratio = (semantic * task_size * computing_density) / (env.MEC_f * remaining_time_for_mec)

                    # 5. 检查资源是否足够
                    if required_resource_ratio > remaining_resource:
                        continue  # 资源不足，该组合不可行

                    # 6. 计算能耗
                    current_energy = calculate_energy(1, semantic, power, task_size, computing_density,
                                                   local_comp, channel_gain, env, temp_bandwidth)

                    # 7. 更新最优解
                    if current_energy < best_energy:
                        best_energy = current_energy
                        # 记录动作: [Offload=1, Semantic, Resource, Power, BW_Weight]
                        best_action = [1, semantic, required_resource_ratio, power, bw_weight]

        # ------------------------------------------------------------------
        # 决策结束，更新状态
        # ------------------------------------------------------------------

        # 如果选择了卸载，扣除资源
        if best_action[0] == 1:
            remaining_resource -= best_action[2]

        energy_array.append(best_energy)
        selected_actions.append(best_action)

    # 现在计算实际的全局能耗和延迟，使用真实的动态带宽分配
    # 提取所有UE的决策
    offload_decisions = [action[0] for action in selected_actions]
    semantic_factors = [action[1] if len(action) > 1 else 1.0 for action in selected_actions]
    resource_allocations = [action[2] if len(action) > 2 else 0.0 for action in selected_actions]
    power_allocations = [action[3] if len(action) > 3 else 0.0 for action in selected_actions]
    bw_weight_allocations = [action[4] if len(action) > 4 else 0 for action in selected_actions]

    # 使用环境的动态带宽分配功能
    allocated_bandwidths = env.dynamic_bandwidth_allocation(bw_weight_allocations)

    # 重新计算总能耗，使用真实的带宽分配
    total_energy = 0
    for i in range(UEs):
        task_size, computing_density, max_delay, local_energy = observation[i]
        local_comp = env.UE_params[i]['local_comp']
        channel_gain = env.UE_params[i]['channel_gain']

        if offload_decisions[i] == 0:  # 本地处理
            total_energy += local_energy
        else:  # 卸载处理
            assigned_bandwidth = allocated_bandwidths[i]
            if assigned_bandwidth <= 0:
                assigned_bandwidth = 10 * 10**3  # 10kHz

            uplink_rate = assigned_bandwidth * math.log2(1 + power_allocations[i] * channel_gain /
                                                       (assigned_bandwidth * config.noise_power))
            upload_energy = power_allocations[i] * (semantic_factors[i] * task_size) / uplink_rate
            SEtask_energy = env.κ * env.alpha * (task_size ** env.r) * (semantic_factors[i]**(-env.beta)-1) * local_comp**2
            total_energy += SEtask_energy + upload_energy

    return selected_actions, total_energy


if __name__ == '__main__':
    # 初始化环境
    energy_records = []

    # 从配置获取UE数量范围
    UEs_range = range(5, config.num_UEs + 1, 5)

    for UEs in UEs_range:
        # 注意：确保 ENV 类能够正确初始化
        env = ENV(UEs=UEs, MECs=1, k=config.k_discretization)
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
    df.to_csv("greedy_energy_with_dynamic_bandwidth.csv", index=False)
    print("Results saved to greedy_energy_with_dynamic_bandwidth.csv")
    print(df)