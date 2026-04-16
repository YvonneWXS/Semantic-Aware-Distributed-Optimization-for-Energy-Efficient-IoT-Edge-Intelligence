"""
无线通信与边缘计算批量实验配置文件

本文件提供两种配置模式：
1. 传统的IPPO算法参数配置（通过get_config()函数）
2. 批量实验参数配置（通过BatchExperimentConfig类）

作者：无线通信与边缘计算算法工程师
日期：2026/04/15
"""

import argparse
import itertools
import math


def get_config():
    """
    IPPO算法参数配置解析器

    该函数提供传统的IPPO算法训练参数配置，包括：
    - 训练参数（学习率、批次大小、迭代次数等）
    - 网络参数（隐藏层大小、层数等）
    - PPO参数（clip参数、GAE参数等）
    - 环境参数（环境名称、episode长度等）

    返回：
        argparse.ArgumentParser: 参数解析器
    """
    parser = argparse.ArgumentParser(
        description="IPPO算法参数配置",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 准备参数
    parser.add_argument("--algorithm_name", type=str, default="mappo", choices=["rmappo", "mappo"])
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="实验名称，用于区分不同实验",
    )
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="是否使用GPU训练，默认为True",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="确保随机种子有效，默认为True",
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=2,
        help="训练线程数",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=1,
        help="训练rollout的并行环境数",
    )
    parser.add_argument(
        "--n_eval_rollout_threads",
        type=int,
        default=1,
        help="评估rollout的并行环境数",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=200000,
        help="训练的环境步数",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="marl",
        help="WandB用户名",
    )

    # 环境参数
    parser.add_argument("--env_name", type=str, default="MyEnv", help="环境名称")
    parser.add_argument(
        "--use_obs_instead_of_state",
        action="store_true",
        default=False,
        help="是否使用全局状态或拼接的观测",
    )

    # 回放缓冲区参数
    parser.add_argument("--episode_length", type=int, default=50, help="episode最大长度")

    # 网络参数
    parser.add_argument(
        "--share_policy",
        action="store_false",
        default=False,
        help="是否共享策略",
    )
    parser.add_argument(
        "--use_centralized_V",
        action="store_false",
        default=False,
        help="是否使用集中式V函数",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="actor/critic网络的隐藏层维度",
    )
    parser.add_argument(
        "--layer_N",
        type=int,
        default=1,
        help="actor/critic网络的层数",
    )
    parser.add_argument("--use_ReLU", action="store_false", default=True, help="是否使用ReLU激活函数")
    parser.add_argument(
        "--use_valuenorm",
        action="store_false",
        default=True,
        help="是否使用running mean和std标准化奖励",
    )
    parser.add_argument(
        "--use_feature_normalization",
        action="store_false",
        default=True,
        help="是否对输入应用layernorm",
    )
    parser.add_argument(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help="是否使用正交初始化",
    )
    parser.add_argument("--gain", type=float, default=0.01, help="最后一层的增益")

    # 循环参数
    parser.add_argument(
        "--use_recurrent_policy",
        action="store_false",
        default=False,
        help="是否使用循环策略",
    )
    parser.add_argument("--recurrent_N", type=int, default=1, help="循环层数")
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=10,
        help="训练循环策略的块长度",
    )

    # 优化器参数
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=5e-4,
        help="critic学习率",
    )
    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop优化器的epsilon",
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    # PPO参数
    parser.add_argument("--ppo_epoch", type=int, default=15, help="PPO epoch数")
    parser.add_argument(
        "--use_clipped_value_loss",
        action="store_false",
        default=True,
        help="是否裁剪值损失",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="PPO裁剪参数",
    )
    parser.add_argument(
        "--num_mini_batch",
        type=int,
        default=1,
        help="PPO的mini-batch数",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.01,
        help="熵系数",
    )
    parser.add_argument(
        "--value_loss_coef",
        type=float,
        default=1,
        help="值损失系数",
    )
    parser.add_argument(
        "--use_max_grad_norm",
        action="store_false",
        default=True,
        help="是否使用梯度最大范数",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10.0,
        help="梯度最大范数",
    )
    parser.add_argument(
        "--use_gae",
        action="store_false",
        default=True,
        help="是否使用广义优势估计",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="奖励折扣因子",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="GAE lambda参数",
    )
    parser.add_argument(
        "--use_huber_loss",
        action="store_false",
        default=True,
        help="是否使用Huber损失",
    )
    parser.add_argument("--huber_delta", type=float, default=10.0, help="Huber损失的delta参数")

    # 运行参数
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="是否使用线性学习率衰减",
    )

    # 保存参数
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="模型保存间隔",
    )

    # 日志参数
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="日志打印间隔",
    )

    # 评估参数
    parser.add_argument(
        "--use_eval",
        action="store_true",
        default=True,
        help="是否进行评估",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25,
        help="评估间隔",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=32,
        help="单次评估的episode数",
    )

    # 渲染参数
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        default=False,
        help="是否保存渲染视频",
    )
    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
        help="是否在训练期间渲染环境",
    )

    # 预训练参数
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="预训练模型路径",
    )

    return parser


class BatchExperimentConfig:
    """
    无线通信与边缘计算批量实验配置类

    该类提供批量实验的参数配置，支持：
    1. 批量遍历参数（数据大小、UE数量、带宽、MEC容量）
    2. 固定参数（种群大小、迭代次数、变异率等）
    3. 参数组合生成和验证
    4. 单位转换和参数标准化

    使用示例：
        config = BatchExperimentConfig()
        combinations = config.generate_param_combinations()
        for params in combinations:
            # 运行实验
            run_experiment(params)
    """

    # ==================== 批量遍历参数列表 ====================
    # 这些参数将在批量实验中遍历所有组合

    # 数据大小列表 (KB)
    DataSize_list = [128, 256, 512, 1024]  # KB

    # UE数量列表
    num_UEs_list = [5, 10, 15, 20, 25, 30]

    # 带宽列表 (kHz)
    bandwidth_list = [750, 1000, 1500, 2000]  # kHz

    # MEC计算容量列表 (Gcps)
    mec_capacity_list = [10.0, 12.5, 15.0, 17.5, 20.0]  # Gcps

    # ==================== 固定参数配置 ====================
    # 这些参数在批量实验中保持固定，作为非对比条件

    FIXED_PARAMS = {
        # 遗传算法参数
        'population_size': 100,           # 种群大小
        'iterations': 1000,               # 迭代次数
        'mutation_rate': 0.1,             # 变异率
        'crossover_rate': 0.8,            # 交叉率
        'selection_method': 'tournament', # 选择方法：tournament或roulette

        # 功率参数 (W)
        'power_min': 0.1,                 # 最小发射功率 (W)
        'power_max': 0.5,                 # 最大发射功率 (W)
        'power_step': 0.01,               # 功率调整步长 (W)

        # 语义通信参数
        'semantic_factor_min': 0.1,       # 最小语义因子
        'semantic_factor_max': 1.0,       # 最大语义因子
        'semantic_weight': 0.5,           # 语义重要性权重

        # 资源分配参数
        'resource_allocation_min': 0.1,   # 最小资源分配比例
        'resource_allocation_max': 1.0,   # 最大资源分配比例

        # 带宽权重参数
        'bw_weight_min': 0,               # 最小带宽权重
        'bw_weight_max': 3,               # 最大带宽权重

        # 能量效率参数
        'energy_efficiency_weight': 0.7,  # 能量效率权重
        'latency_weight': 0.3,            # 延迟权重

        # 环境参数
        'noise_spectral_density': 1e-17,  # 噪声谱密度 (W/Hz)
        'path_loss_exponent': 3.5,        # 路径损耗指数
        'reference_distance': 1.0,        # 参考距离 (m)
        'carrier_frequency': 2.4e9,       # 载波频率 (Hz)

        # 计算复杂度参数
        'computational_intensity': 1000,  # 计算强度 (cycles/bit)
        'cpu_frequency': 2.5e9,           # CPU频率 (Hz)

        # 实验参数
        'num_runs': 10,                   # 每个参数组合的运行次数
        'convergence_threshold': 1e-4,    # 收敛阈值
        'max_runtime': 3600,              # 最大运行时间 (秒)
    }

    # ==================== 单位转换常量 ====================
    KB_TO_BITS = 8192                     # 1 KB = 8192 bits (1024 * 8)
    KHZ_TO_HZ = 1000                      # 1 kHz = 1000 Hz
    GCPS_TO_CPS = 1e9                     # 1 Gcps = 1e9 cycles per second

    @classmethod
    def generate_param_combinations(cls):
        """
        生成所有参数组合

        返回：
            list: 包含所有参数组合的列表，每个组合是一个字典
        """
        # 定义参数网格
        param_grid = {
            'data_size': cls.DataSize_list,          # KB
            'num_UEs': cls.num_UEs_list,             # 数量
            'bandwidth': cls.bandwidth_list,         # kHz
            'mec_capacity': cls.mec_capacity_list,   # Gcps
        }

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))

            # 添加固定参数
            param_dict.update(cls.FIXED_PARAMS)

            # 添加转换后的参数（标准化单位）
            param_dict.update(cls._convert_units(param_dict))

            # 添加实验ID
            param_dict['experiment_id'] = len(combinations) + 1

            combinations.append(param_dict)

        return combinations

    @classmethod
    def _convert_units(cls, params):
        """
        将参数转换为标准单位

        参数：
            params (dict): 原始参数字典

        返回：
            dict: 包含转换后参数的字典
        """
        converted = {}

        # 数据大小：KB → bits
        if 'data_size' in params:
            converted['data_size_bits'] = params['data_size'] * cls.KB_TO_BITS

        # 带宽：kHz → Hz
        if 'bandwidth' in params:
            converted['bandwidth_hz'] = params['bandwidth'] * cls.KHZ_TO_HZ

        # MEC容量：Gcps → cps
        if 'mec_capacity' in params:
            converted['mec_capacity_cps'] = params['mec_capacity'] * cls.GCPS_TO_CPS

        # 计算总比特率 (bits/s)
        if 'data_size_bits' in converted and 'bandwidth_hz' in converted:
            # 假设使用BPSK调制，频谱效率为1 bit/s/Hz
            converted['max_bitrate'] = converted['bandwidth_hz']  # bits/s

        return converted

    @classmethod
    def get_total_experiments(cls):
        """
        获取总实验数量

        返回：
            int: 总实验数量
        """
        return (len(cls.DataSize_list) * len(cls.num_UEs_list) *
                len(cls.bandwidth_list) * len(cls.mec_capacity_list))

    @classmethod
    def get_experiment_summary(cls):
        """
        获取实验配置摘要

        返回：
            dict: 包含实验配置摘要的字典
        """
        return {
            'total_experiments': cls.get_total_experiments(),
            'data_size_range': f"{min(cls.DataSize_list)}-{max(cls.DataSize_list)} KB",
            'num_UEs_range': f"{min(cls.num_UEs_list)}-{max(cls.num_UEs_list)}",
            'bandwidth_range': f"{min(cls.bandwidth_list)}-{max(cls.bandwidth_list)} kHz",
            'mec_capacity_range': f"{min(cls.mec_capacity_list)}-{max(cls.mec_capacity_list)} Gcps",
            'fixed_params': cls.FIXED_PARAMS.copy(),
        }

    @classmethod
    def validate_params(cls, params):
        """
        验证参数的有效性

        参数：
            params (dict): 要验证的参数字典

        返回：
            tuple: (is_valid, error_message)
        """
        # 检查必需参数
        required_params = ['data_size', 'num_UEs', 'bandwidth', 'mec_capacity']
        for param in required_params:
            if param not in params:
                return False, f"缺少必需参数: {param}"

        # 检查参数范围
        if params['data_size'] not in cls.DataSize_list:
            return False, f"数据大小 {params['data_size']} KB 不在允许范围内"

        if params['num_UEs'] not in cls.num_UEs_list:
            return False, f"UE数量 {params['num_UEs']} 不在允许范围内"

        if params['bandwidth'] not in cls.bandwidth_list:
            return False, f"带宽 {params['bandwidth']} kHz 不在允许范围内"

        if params['mec_capacity'] not in cls.mec_capacity_list:
            return False, f"MEC容量 {params['mec_capacity']} Gcps 不在允许范围内"

        # 检查固定参数范围
        if params['mutation_rate'] < 0 or params['mutation_rate'] > 1:
            return False, f"变异率 {params['mutation_rate']} 必须在0-1之间"

        if params['crossover_rate'] < 0 or params['crossover_rate'] > 1:
            return False, f"交叉率 {params['crossover_rate']} 必须在0-1之间"

        if params['power_min'] >= params['power_max']:
            return False, "最小功率必须小于最大功率"

        return True, "参数验证通过"

    @classmethod
    def get_param_combination_by_id(cls, experiment_id):
        """
        根据实验ID获取参数组合

        参数：
            experiment_id (int): 实验ID（从1开始）

        返回：
            dict: 参数组合字典，如果ID无效则返回None
        """
        combinations = cls.generate_param_combinations()
        if 1 <= experiment_id <= len(combinations):
            return combinations[experiment_id - 1]
        return None


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例1: 获取传统IPPO配置
    parser = get_config()
    args = parser.parse_args([])  # 使用空列表获取默认参数
    print("传统IPPO配置示例:")
    print(f"  算法名称: {args.algorithm_name}")
    print(f"  隐藏层大小: {args.hidden_size}")
    print(f"  学习率: {args.lr}")
    print()

    # 示例2: 批量实验配置
    config = BatchExperimentConfig()

    # 获取实验摘要
    summary = config.get_experiment_summary()
    print("批量实验配置摘要:")
    print(f"  总实验数量: {summary['total_experiments']}")
    print(f"  数据大小范围: {summary['data_size_range']}")
    print(f"  UE数量范围: {summary['num_UEs_range']}")
    print(f"  带宽范围: {summary['bandwidth_range']}")
    print(f"  MEC容量范围: {summary['mec_capacity_range']}")
    print()

    # 生成前3个参数组合作为示例
    combinations = config.generate_param_combinations()
    print("前3个参数组合示例:")
    for i, params in enumerate(combinations[:3], 1):
        print(f"\n组合 {i}:")
        print(f"  实验ID: {params['experiment_id']}")
        print(f"  数据大小: {params['data_size']} KB ({params['data_size_bits']} bits)")
        print(f"  UE数量: {params['num_UEs']}")
        print(f"  带宽: {params['bandwidth']} kHz ({params['bandwidth_hz']} Hz)")
        print(f"  MEC容量: {params['mec_capacity']} Gcps ({params['mec_capacity_cps']} cps)")
        print(f"  种群大小: {params['population_size']}")
        print(f"  迭代次数: {params['iterations']}")

    # 验证参数
    print("\n参数验证示例:")
    is_valid, message = config.validate_params(combinations[0])
    print(f"  验证结果: {is_valid}")
    print(f"  验证消息: {message}")