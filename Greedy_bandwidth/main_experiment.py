"""
语义感知MEC系统批量对比实验主程序
自动遍历所有参数组合并执行实验
"""
import math
import pandas as pd
import numpy as np
from env import ENV
from SA_greedy_energy1st import greedy_algorithm
import config

def run_single_experiment(num_UEs, total_bandwidth, mec_capacity, min_semantic_factor, data_size_kb):
    """
    运行单次实验
    """
    # 转换数据单位
    data_size_bytes = data_size_kb * 1024  # KB to bytes
    total_bandwidth_hz = total_bandwidth * 1000  # kHz to Hz
    mec_capacity_hz = mec_capacity * 10**9  # GHz to Hz

    # 创建环境
    env = ENV(UEs=num_UEs, MECs=1, k=config.k_discretization)
    # 更新环境参数
    env.total_bandwidth = total_bandwidth_hz
    env.MEC_f = mec_capacity_hz
    env.UE_params = []  # 重新生成UE参数
    for i in range(num_UEs):
        local_comp = np.random.randint(1.5 * env.GHz, 2 * env.GHz)    # UE的本地计算能力
        distance = np.random.uniform(10, 100)  # 随机生成用户设备和基站之间的距离
        channel_gain = 1e-3 * (1.0 / distance) ** 2.5  # 计算信道增益
        # 将每个UE的参数存储在字典中
        ue_params = {
            'local_comp': local_comp,
            'channel_gain': channel_gain,
        }
        env.UE_params.append(ue_params)

    # 生成观测值
    obs = []
    for i in range(num_UEs):
        # 根据指定的数据大小，而不是随机生成
        task_size = data_size_bytes
        computing_density = np.random.uniform(300, 500)  # 处理任务每比特数据的成本
        local_comp = env.UE_params[i]['local_comp']
        local_delay = task_size * computing_density / local_comp  # 本地处理任务时间
        local_energy = env.κ * task_size * computing_density * local_comp**2
        # 根据配置生成最大时延
        max_delay = np.random.uniform(local_delay, config.max_delay_factor * local_delay)  # 任务最大容忍时间随机取
        observation = np.array([task_size, computing_density, max_delay, local_energy])
        obs.append(observation)

    # 运行贪婪算法
    actions, total_energy = greedy_algorithm(env, obs)

    # 计算其他性能指标
    total_delay = 0
    total_tx_delay = 0
    total_comp_delay = 0
    total_bandwidth_utilization = 0
    avg_semantic_factor = 0
    num_offloaded = 0

    # 计算详细性能指标
    for i in range(num_UEs):
        task_size, computing_density, max_delay, local_energy = obs[i]
        local_comp = env.UE_params[i]['local_comp']
        channel_gain = env.UE_params[i]['channel_gain']
        action = actions[i]

        offload = action[0]
        semantic = action[1]
        resource = action[2]
        power = action[3]
        bw_weight = action[4]

        # 累计语义因子
        avg_semantic_factor += semantic
        if offload == 1:
            num_offloaded += 1

        if offload == 0:  # 本地处理
            delay = task_size * computing_density / local_comp
            tx_delay = 0
            comp_delay = delay
        else:  # 卸载处理
            # 使用分配的带宽计算时延
            allocated_bandwidths = env.dynamic_bandwidth_allocation([act[4] for act in actions])
            assigned_bandwidth = allocated_bandwidths[i]
            if assigned_bandwidth <= 0:
                assigned_bandwidth = 10 * 10**3  # 10kHz

            uplink_rate = assigned_bandwidth * math.log2(1 + power * channel_gain / (assigned_bandwidth * config.noise_power))

            # 语义提取时延
            se_delay = env.alpha * (task_size**env.r) * ((semantic**(-1)) - 1) / local_comp
            # 传输时延
            tx_delay = semantic * task_size / uplink_rate
            # MEC计算时延
            mec_delay = semantic * task_size * computing_density / (env.MEC_f * resource) if resource > 0 else 0
            # 总时延
            delay = se_delay + tx_delay + mec_delay

            total_tx_delay += tx_delay
            total_comp_delay += mec_delay

        total_delay += delay

    # 计算平均值
    avg_delay = total_delay / num_UEs if num_UEs > 0 else 0
    avg_tx_delay = total_tx_delay / num_offloaded if num_offloaded > 0 else 0
    avg_comp_delay = total_comp_delay / num_offloaded if num_offloaded > 0 else 0
    avg_semantic_factor /= num_UEs if num_UEs > 0 else 0

    # 计算带宽利用率
    allocated_bandwidths = env.dynamic_bandwidth_allocation([act[4] for act in actions])
    total_allocated_bw = sum(allocated_bandwidths)
    bandwidth_utilization = total_allocated_bw / total_bandwidth_hz if total_bandwidth_hz > 0 else 0

    return {
        'num_UEs': num_UEs,
        'data_size_KB': data_size_kb,
        'total_bandwidth_kHz': total_bandwidth,
        'mec_capacity_GHz': mec_capacity,
        'min_semantic_factor': min_semantic_factor,
        'total_energy': total_energy,
        'avg_delay': avg_delay,
        'avg_tx_delay': avg_tx_delay,
        'avg_comp_delay': avg_comp_delay,
        'bandwidth_utilization': bandwidth_utilization,
        'avg_semantic_factor': avg_semantic_factor,
        'num_offloaded': num_offloaded,
        'offload_ratio': num_offloaded / num_UEs if num_UEs > 0 else 0
    }


def run_batch_experiments():
    """
    运行批量实验
    """
    print("开始批量对比实验...")

    results = []

    # 根据selected_parameter选择要测试的参数
    if config.selected_parameter == 'num_UEs':
        param_values = config.num_UEs_list
        fixed_data_size = config.BASE_VALUES['DataSize']  # 使用基础参数
        fixed_bandwidth = config.BASE_VALUES['bandwidth']
        fixed_mec_capacity = config.BASE_VALUES['mec_capacity']
        fixed_min_semantic = config.BASE_VALUES['min_semantic_factor']

        for param_value in param_values:
            print(f"测试 UE 数量: {param_value}")
            run_results = []

            for run_idx in range(config.num_runs_per_config):
                result = run_single_experiment(
                    num_UEs=param_value,
                    total_bandwidth=fixed_bandwidth,
                    mec_capacity=fixed_mec_capacity,
                    min_semantic_factor=fixed_min_semantic,
                    data_size_kb=fixed_data_size
                )
                run_results.append(result)

            # 计算平均值
            avg_result = {}
            for key in run_results[0].keys():
                if isinstance(run_results[0][key], (int, float)):
                    avg_result[key] = sum(r[key] for r in run_results) / len(run_results)
                else:
                    avg_result[key] = run_results[0][key]  # 保持不变的值

            results.append(avg_result)

    elif config.selected_parameter == 'DataSize':
        param_values = config.DataSize_KB
        fixed_UEs = config.BASE_VALUES['num_UEs']
        fixed_bandwidth = config.BASE_VALUES['bandwidth']
        fixed_mec_capacity = config.BASE_VALUES['mec_capacity']
        fixed_min_semantic = config.BASE_VALUES['min_semantic_factor']

        for param_value in param_values:
            print(f"测试数据大小: {param_value} KB")
            run_results = []

            for run_idx in range(config.num_runs_per_config):
                result = run_single_experiment(
                    num_UEs=fixed_UEs,
                    total_bandwidth=fixed_bandwidth,
                    mec_capacity=fixed_mec_capacity,
                    min_semantic_factor=fixed_min_semantic,
                    data_size_kb=param_value
                )
                run_results.append(result)

            # 计算平均值
            avg_result = {}
            for key in run_results[0].keys():
                if isinstance(run_results[0][key], (int, float)):
                    avg_result[key] = sum(r[key] for r in run_results) / len(run_results)
                else:
                    avg_result[key] = run_results[0][key]

            results.append(avg_result)

    elif config.selected_parameter == 'bandwidth':
        param_values = config.total_bandwidth_kHz
        fixed_UEs = config.BASE_VALUES['num_UEs']
        fixed_data_size = config.BASE_VALUES['DataSize']
        fixed_mec_capacity = config.BASE_VALUES['mec_capacity']
        fixed_min_semantic = config.BASE_VALUES['min_semantic_factor']

        for param_value in param_values:
            print(f"测试带宽: {param_value} kHz")
            run_results = []

            for run_idx in range(config.num_runs_per_config):
                result = run_single_experiment(
                    num_UEs=fixed_UEs,
                    total_bandwidth=param_value,
                    mec_capacity=fixed_mec_capacity,
                    min_semantic_factor=fixed_min_semantic,
                    data_size_kb=fixed_data_size
                )
                run_results.append(result)

            # 计算平均值
            avg_result = {}
            for key in run_results[0].keys():
                if isinstance(run_results[0][key], (int, float)):
                    avg_result[key] = sum(r[key] for r in run_results) / len(run_results)
                else:
                    avg_result[key] = run_results[0][key]

            results.append(avg_result)

    elif config.selected_parameter == 'mec_capacity':
        param_values = config.mec_capacity_GHz
        fixed_UEs = config.BASE_VALUES['num_UEs']
        fixed_data_size = config.BASE_VALUES['DataSize']
        fixed_bandwidth = config.BASE_VALUES['bandwidth']
        fixed_min_semantic = config.BASE_VALUES['min_semantic_factor']

        for param_value in param_values:
            print(f"测试MEC容量: {param_value} GHz")
            run_results = []

            for run_idx in range(config.num_runs_per_config):
                result = run_single_experiment(
                    num_UEs=fixed_UEs,
                    total_bandwidth=fixed_bandwidth,
                    mec_capacity=param_value,
                    min_semantic_factor=fixed_min_semantic,
                    data_size_kb=fixed_data_size
                )
                run_results.append(result)

            # 计算平均值
            avg_result = {}
            for key in run_results[0].keys():
                if isinstance(run_results[0][key], (int, float)):
                    avg_result[key] = sum(r[key] for r in run_results) / len(run_results)
                else:
                    avg_result[key] = run_results[0][key]

            results.append(avg_result)

    elif config.selected_parameter == 'min_semantic_factor':
        param_values = config.min_semantic_factor_list
        fixed_UEs = config.BASE_VALUES['num_UEs']
        fixed_data_size = config.BASE_VALUES['DataSize']
        fixed_bandwidth = config.BASE_VALUES['bandwidth']
        fixed_mec_capacity = config.BASE_VALUES['mec_capacity']

        for param_value in param_values:
            print(f"测试最小语义因子: {param_value}")
            run_results = []

            for run_idx in range(config.num_runs_per_config):
                result = run_single_experiment(
                    num_UEs=fixed_UEs,
                    total_bandwidth=fixed_bandwidth,
                    mec_capacity=fixed_mec_capacity,
                    min_semantic_factor=param_value,
                    data_size_kb=fixed_data_size
                )
                run_results.append(result)

            # 计算平均值
            avg_result = {}
            for key in run_results[0].keys():
                if isinstance(run_results[0][key], (int, float)):
                    avg_result[key] = sum(r[key] for r in run_results) / len(run_results)
                else:
                    avg_result[key] = run_results[0][key]

            results.append(avg_result)

    # 保存结果到CSV
    df = pd.DataFrame(results)
    df.to_csv("experiment_results.csv", index=False)
    print(f"实验完成！共 {len(results)} 个结果已保存到 experiment_results.csv")
    print("\n结果摘要：")
    print(df)

    return df


def run_full_combination_experiments():
    """
    运行所有参数组合的完整实验（可选）
    """
    print("开始全参数组合实验...")

    results = []

    # 遍历所有参数组合
    for num_UEs in config.num_UEs_list[:2]:  # 为节省时间，只测试前2个值
        for data_size in config.DataSize_KB[:2]:  # 只测试前2个值
            for bandwidth in config.total_bandwidth_kHz[:2]:  # 只测试前2个值
                for mec_cap in config.mec_capacity_GHz[:2]:  # 只测试前2个值
                    print(f"测试组合: UEs={num_UEs}, Data={data_size}KB, BW={bandwidth}kHz, MEC={mec_cap}GHz")

                    run_results = []
                    for run_idx in range(2):  # 为节省时间，只运行2次
                        result = run_single_experiment(
                            num_UEs=num_UEs,
                            total_bandwidth=bandwidth,
                            mec_capacity=mec_cap,
                            min_semantic_factor=config.min_semantic_factor_list[0],  # 固定语义因子
                            data_size_kb=data_size
                        )
                        run_results.append(result)

                    # 计算平均值
                    avg_result = {}
                    for key in run_results[0].keys():
                        if isinstance(run_results[0][key], (int, float)):
                            avg_result[key] = sum(r[key] for r in run_results) / len(run_results)
                        else:
                            avg_result[key] = run_results[0][key]

                    results.append(avg_result)

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("full_combination_results.csv", index=False)
    print(f"全组合实验完成！共 {len(results)} 个结果已保存到 full_combination_results.csv")

    return df


if __name__ == '__main__':
    # 只运行批量实验（单参数变化）
    df_results = run_batch_experiments()

    print("\n" + "="*60)
    print("批量对比实验已完成！")
    print(f"当前测试参数: {config.selected_parameter}")
    print(f"结果已保存到 experiment_results.csv")
    print("="*60)