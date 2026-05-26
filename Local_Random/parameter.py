"""
ALE/ROE 实验参数配置
与论文原文（Table I + 系统模型）严格一致
"""

import numpy as np

# ==================== 常用默认配置（在此手动修改） ====================
DEFAULT_DATA_SIZE_KB = 250
DEFAULT_NUM_UES = 5
DEFAULT_BANDWIDTH_KHZ = 1000
DEFAULT_MEC_CAPACITY_GHZ = 20.0

# ==================== 批量实验参数（列表形式） ====================
data_size_kb = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
num_of_ues = [5, 10, 15, 20, 25, 30]
bandwidth_khz = [750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
mec_capacity = [10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0]

# ==================== 系统固定参数（与论文原文严格一致） ====================
SYSTEM_PARAMS = {
    # 芯片与语义提取参数（论文 Eq. (2)）
    "κ": 1e-27,                          # 芯片电容系数
    "r": 1.0,                            # 语义提取公式指数 η
    "alpha": 1.0,                        # 语义提取公式系数 α
    "beta": 2.0,                         # 语义提取公式指数 k

    # 通信参数（论文 Eq. (4)）
    "transmission_bandwidth": 1_000_000, # 总带宽 B = 1 MHz
    "noise_power": 1e-20,                # 噪声功率 σ₀² = -170 dBm（线性值）
    "transmission_power": 0.1,           # 默认传输功率 P_i = 0.1 W

    # 计算参数（论文 Eq. (7)）
    "MEC_f": 20_000_000_000,             # MEC 总计算能力 F = 20 Gcycles/s
    "T": 1.0,                            # 时隙长度 = 1 s

    # 动作离散化参数
    "k": 100,                            # semantic_factor 与 resource_allocation 离散等级
    "semantic_threshold": 0.3,           # β_min = 0.3

    # DW-DNA 专用参数（Proposed 方案）
    "delta_b": 10_000,                   # 最小资源块 Δb = 10 kHz

    # 单位转换
    "Hz": 1,
    "kHz": 1_000,
    "mHz": 1_000_000,
    "GHz": 1_000_000_000,
    "bit": 1,
    "B": 8,
    "KB": 8_192,      # 1024 * 8
    "MB": 8_388_608,  # 1024 * 1024 * 8
}

# ==================== UE 每 episode 随机参数（与论文一致） ====================
ue_params_per_episode = {
    "task_size": "np.random.uniform(1.5*MB, 2.0*MB)",      # D_i^o ∈ [1.5, 2.0] MB
    "computing_density": "np.random.uniform(300, 500)",    # C_i ∈ [300, 500] cycles/bit
    "local_comp": "np.random.uniform(1.5*GHz, 2.0*GHz)",   # f_l ∈ [1.5, 2.0] GHz
    "channel_gain": "1e-3 * (1.0 / np.random.uniform(10, 100))**2.5",  # 距离 [10,100] m
    "max_delay": "np.random.uniform(local_delay, 2*local_delay)",
}

# ==================== 离散动作集合 ====================
discrete_sets = {
    "semantic_factor_set": "np.round(np.linspace(0.3, 1.0, SYSTEM_PARAMS['k']), 2)",
    "resource_allocation_set": "np.round(np.linspace(0.01, 1.0, SYSTEM_PARAMS['k']), 2)",
    "power_levels": [0.1, 0.2, 0.3, 0.4, 0.5],          # 论文中 P_i ∈ [0.1, 0.5] W
    "w_bw_levels": [0, 1, 2, 3, 4],                      # DW-DNA 离散权重
}

# ==================== 实验运行参数 ====================
EXPERIMENT_PARAMS = {
    "num_runs": 10,
    "output_dir": "results",
}

# ==================== 单位转换辅助函数 ====================
def convert_units(value, from_unit, to_unit):
    """单位转换辅助函数"""
    units = SYSTEM_PARAMS
    if from_unit in units and to_unit in units:
        return value * units[from_unit] / units[to_unit]
    else:
        raise ValueError(f"Unsupported units: {from_unit} -> {to_unit}")

# 导出常用单位常量
Hz = SYSTEM_PARAMS["Hz"]
kHz = SYSTEM_PARAMS["kHz"]
mHz = SYSTEM_PARAMS["mHz"]
GHz = SYSTEM_PARAMS["GHz"]
bit = SYSTEM_PARAMS["bit"]
B = SYSTEM_PARAMS["B"]
KB = SYSTEM_PARAMS["KB"]
MB = SYSTEM_PARAMS["MB"]
