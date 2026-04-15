# SA-GA 批量实验配置文件
# 支持批量遍历的参数列表

# 批量实验参数列表（支持遍历）
DataSize_list = [128, 256, 512, 1024]  # 任务数据大小 (KB)
num_UEs_list = [5, 10, 15, 20, 25, 30]  # 用户设备数量
bandwidth_list = [750, 1000, 1500, 2000]  # 总带宽 (kHz)
mec_capacity_list = [10.0, 12.5, 15.0, 17.5, 20.0]  # MEC服务器计算能力 (Giga Cycles/s)
min_semantic_factor_list = [0.2, 0.3, 0.4, 0.5]  # 最小语义提取因子

# 固定参数（可在批量实验中手动修改）
class FixedConfig:
    # 遗传算法参数
    pop_size = 100  # 种群大小
    generations = 3000  # 迭代次数
    early_stop_patience = 500  # 早停耐心值
    early_stop_threshold = 0.01  # 早停阈值
    crossover_rate = 0.89  # 交叉率
    mutation_rate = 0.9  # 变异率
    max_attempts = 100  # 最大尝试次数（用于初始化）

    # 动作空间参数
    semantic_factor_steps = 8  # 语义提取因子离散化步数 (0.3-1.0)
    resource_allocation_steps = 10  # 资源分配离散化步数 (0.1-1.0)
    transmission_power_steps = 5  # 传输功率离散化步数 (0.1-0.5)
    bw_weight_set = [0, 1, 2, 3]  # 带宽权重集合

    # 环境参数
    UE_local_comp_range = (1.5, 2.0)  # UE本地计算能力范围 (GHz)
    UE_distance_range = (10, 100)  # UE到基站距离范围 (m)
    task_size_range = (1.5, 2.0)  # 任务大小范围 (MB)
    computing_density_range = (300, 500)  # 计算密度范围 (cycles/bit)
    max_delay_factor = 2.0  # 最大延迟因子（相对于本地延迟）

    # 物理常数
    kappa = 1e-27  # 芯片结构因子
    alpha = 1  # 语义提取CPU周期参数1
    beta = 2  # 语义提取CPU周期参数2
    r = 1  # 语义提取CPU周期参数3
    noise_power = 1e-20  # 噪声功率 (W)
    delta_b = 1  # 资源块大小 (kHz)

    # 实验设置
    num_trials = 50  # 每个配置的试验次数
    result_dir = "./results"  # 结果保存目录

# 批量实验配置生成器
def generate_experiment_configs():
    """生成所有参数组合的配置列表"""
    configs = []
    for data_size in DataSize_list:
        for num_UEs in num_UEs_list:
            for bandwidth in bandwidth_list:
                for mec_capacity in mec_capacity_list:
                    for min_semantic_factor in min_semantic_factor_list:
                        config = {
                            'data_size': data_size,
                            'num_UEs': num_UEs,
                            'bandwidth': bandwidth,
                            'mec_capacity': mec_capacity,
                            'min_semantic_factor': min_semantic_factor,
                            'fixed': FixedConfig()
                        }
                        configs.append(config)
    return configs

def print_experiment_summary():
    """打印实验配置摘要"""
    configs = generate_experiment_configs()
    print(f"总实验配置数: {len(configs)}")
    print(f"数据大小组合: {DataSize_list}")
    print(f"用户设备数量组合: {num_UEs_list}")
    print(f"带宽组合: {bandwidth_list}")
    print(f"MEC能力组合: {mec_capacity_list}")
    print(f"最小语义因子组合: {min_semantic_factor_list}")
    print(f"每个配置试验次数: {FixedConfig.num_trials}")
    print(f"预计总试验次数: {len(configs) * FixedConfig.num_trials}")

if __name__ == "__main__":
    print_experiment_summary()