"""
批量实验配置文件
包含不同参数组合，用于遍历实验
"""

# 数据大小列表 (KB)
DataSize_list = [128, 256, 512, 1024]  # KB

# 用户设备数量列表
num_UEs_list = [5, 10, 15, 20, 25, 30]

# 总带宽列表 (kHz)
bandwidth_list = [750, 1000, 1500, 2000]  # kHz

# MEC计算能力列表 (Gcps)
mec_capacity_list = [10.0, 12.5, 15.0, 17.5, 20.0]  # Gcps

# 最小语义因子列表
min_semantic_factor_list = [0.2, 0.3, 0.4, 0.5]

# 其他固定参数（可在批量实验中手动修改）
other_params = {
    'population_size': 100,          # 种群大小（如果使用遗传算法）
    'iterations': 1000,              # 迭代次数
    'mutation_rate': 0.1,            # 变异率（如果使用遗传算法）
    'power_min': 0.02,               # 传输功率最小值 (W)
    'power_max': 0.1,                # 传输功率最大值 (W)
    'episode_length': 50,            # 每个episode最大步长
    'num_env_steps': 200000,         # 总环境步数
    'lr': 5e-4,                      # 学习率
    'hidden_size': 64,               # 隐藏层大小
    'layer_N': 1,                    # 网络层数
    'use_popart': False,             # 是否使用PopArt
    'use_valuenorm': True,           # 是否使用ValueNorm
    'gamma': 0.99,                   # 折扣因子
    'clip_param': 0.2,               # PPO裁剪参数
}

# 实验命名模板
exp_name_template = "SA_IPPO_DWDNA_UE{num_UEs}_BW{bandwidth}_MEC{mec_capacity}_SE{min_semantic}"

def generate_exp_name(num_UEs, bandwidth, mec_capacity, min_semantic):
    """生成实验名称"""
    return exp_name_template.format(
        num_UEs=num_UEs,
        bandwidth=bandwidth,
        mec_capacity=mec_capacity,
        min_semantic=min_semantic
    )