"""
批量实验参数配置文件

支持多种实验配置:
- 数据大小
- 用户数量
- 总带宽
- MEC计算能力
- 最小语义提取因子

使用方式:
1. 修改下方参数列表，选择要遍历的参数
2. 设置其他参数为固定值
3. 运行 main_experiment.py 自动执行所有实验
"""

import numpy as np
from typing import List, Dict, Any, Optional


# ==================== 实验参数列表 ====================

# 数据大小列表 (KB)
DataSize_list: List[float] = [128, 256, 512, 1024]

# 用户数量列表
num_UEs_list: List[int] = [5, 10, 15, 20, 25, 30]

# 总带宽列表 (kHz)
bandwidth_list: List[int] = [750, 1000, 1500, 2000]

# MEC计算能力列表 (Giga Cycles/s)
mec_capacity_list: List[float] = [10.0, 12.5, 15.0, 17.5, 20.0]

# 最小语义提取因子列表
min_semantic_factor_list: List[float] = [0.2, 0.3, 0.4, 0.5]


# ==================== 固定参数 (非对比参数) ====================

# 默认固定参数
DEFAULT_PARAMS: Dict[str, Any] = {
    'data_size': 256,           # 数据大小 (KB)
    'num_UEs': 5,             # 用户数量
    'bandwidth': 1000,         # 总带宽 (kHz)
    'mec_capacity': 20.0,       # MEC计算能力 (Giga Cycles/s)
    'min_semantic_factor': 0.3,  # 最小语义提取因子
}


# ==================== GA算法参数 ====================

# 遗传算法超参数
GA_PARAMS: Dict[str, Any] = {
    'pop_size': 50,               # 种群大小
    'generations': 2000,           # 最大代数
    'crossover_rate': 0.8,         # 交叉概率
    'mutation_rate': 0.3,         # 变异概率
    'early_stop_threshold': 0.001,   # 早停阈值
    'patience': 500,               # 早停耐心值
}


# ==================== 实验配置函数 ====================

def get_single_var_config(var_name: str) -> List[Dict[str, Any]]:
    """
    获取单个变量的实验配置列表

    参数:
        var_name: 变量名 ('data_size', 'num_UEs', 'bandwidth', 'mec_capacity', 'min_semantic_factor')

    返回:
        实验配置列表
    """
    configs = []

    if var_name == 'data_size':
        for data_size in DataSize_list:
            config = DEFAULT_PARAMS.copy()
            config['data_size'] = data_size
            configs.append(config)
    elif var_name == 'num_UEs':
        for num_UEs in num_UEs_list:
            config = DEFAULT_PARAMS.copy()
            config['num_UEs'] = num_UEs
            configs.append(config)
    elif var_name == 'bandwidth':
        for bandwidth in bandwidth_list:
            config = DEFAULT_PARAMS.copy()
            config['bandwidth'] = bandwidth
            configs.append(config)
    elif var_name == 'mec_capacity':
        for mec_capacity in mec_capacity_list:
            config = DEFAULT_PARAMS.copy()
            config['mec_capacity'] = mec_capacity
            configs.append(config)
    elif var_name == 'min_semantic_factor':
        for min_semantic_factor in min_semantic_factor_list:
            config = DEFAULT_PARAMS.copy()
            config['min_semantic_factor'] = min_semantic_factor
            configs.append(config)
    else:
        raise ValueError(f"Unknown variable name: {var_name}")

    return configs


def get_full_factorial_config() -> List[Dict[str, Any]]:
    """
    获取全因子实验配置列表

    返回:
        所有参数组合的配置列表
    """
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
                        }
                        configs.append(config)

    return configs


def get_custom_config(var_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    获取自定义实验配置

    参数:
        var_name: 要遍历的变量名，None表示全因子

    返回:
        配置列表
    """
    if var_name is None:
        return get_full_factorial_config()
    else:
        return get_single_var_config(var_name)


# ==================== 结果保存配置 ====================

# 结果保存目录
RESULT_DIR: str = "experiment_results"

# 结果文件名模板
RESULT_FILENAME_TEMPLATE: str = "GA_results_{var_name}.csv"

# 是否保存每代能耗历史
SAVE_ENERGY_HISTORY: bool = False

# 是否保存最优解详情
SAVE_BEST_SOLUTION: bool = True


# ==================== 便捷函数 ====================

def print_config_info():
    """打印配置信息"""
    print("=" * 60)
    print("GA批量实验配置信息")
    print("=" * 60)

    print("\n[实验参数列表]")
    print(f"  DataSize_list: {DataSize_list} KB")
    print(f"  num_UEs_list: {num_UEs_list}")
    print(f"  bandwidth_list: {bandwidth_list} kHz")
    print(f"  mec_capacity_list: {mec_capacity_list} Gcps")
    print(f"  min_semantic_factor_list: {min_semantic_factor_list}")

    print("\n[固定参数]")
    for key, value in DEFAULT_PARAMS.items():
        print(f"  {key}: {value}")

    print("\n[GA算法参数]")
    for key, value in GA_PARAMS.items():
        print(f"  {key}: {value}")

    print("\n[结果保存]")
    print(f"  保存目录: {RESULT_DIR}")
    print(f"  保存能耗历史: {SAVE_ENERGY_HISTORY}")
    print(f"  保存最优解: {SAVE_BEST_SOLUTION}")

    print("=" * 60)


if __name__ == '__main__':
    print_config_info()

    # 测试获取配置
    print("\n测试: 获取 num_UEs 的实验配置")
    configs = get_single_var_config('num_UEs')
    print(f"  共 {len(configs)} 个配置")
    for i, c in enumerate(configs[:3]):
        print(f"    {i+1}. {c}")
    if len(configs) > 3:
        print(f"    ...")