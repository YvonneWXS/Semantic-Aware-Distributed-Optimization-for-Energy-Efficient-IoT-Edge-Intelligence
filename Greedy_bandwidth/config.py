"""
语义感知MEC系统参数配置文件
用于对比实验参数调节
"""

# 系统基本参数 - 用于批量实验的参数列表
DataSize_KB = [100, 200, 300, 400]                        # 单UE数据大小范围 (KB)
num_UEs_list = [5, 10, 15, 20]                            # 用户设备数量列表
total_bandwidth_kHz = [750, 1000, 1250]                   # 系统总带宽列表 (kHz)
mec_capacity_GHz = [10.0, 12.5, 15.0]                    # MEC服务器总计算容量列表 (GHz)

# 语义提取参数
min_semantic_factor_list = [0.2, 0.3, 0.4, 0.5]          # 最小语义提取因子列表
k_discretization = 100                                     # 语义因子离散化参数

# 约束参数
max_delay_factor = 2                                       # 最大时延相对于本地处理时延的倍数

# 动作空间参数
power_list = [0.1, 0.2, 0.3, 0.4, 0.5]                  # 发射功率可选列表 (W)
bw_weights = [0, 1, 2, 3]                                 # 带宽权重可选列表

# 物理参数（这些是固定的，除非您想改变基础环境设置）
kappa = 10**(-27)                                          # 芯片结构对cpu处理的影响因子
alpha_se = 1                                               # 语义提取CPU周期参数1
beta_se = 2                                                # 语义提取CPU周期参数2
r_se = 1                                                   # 语义提取CPU周期参数3
noise_power = 10**(-20)                                    # 噪声功率 -170dBm
MEC_freq_base = 20 * 10**9                                 # MEC计算频率基准 20GHz

# 仿真参数
slot_duration = 1                                          # 时隙长度 (s)

# 实验控制参数
num_runs_per_config = 10                                   # 每个配置的运行次数（用于平均）
selected_parameter = 'DataSize'                             # 选择要测试的参数: 'num_UEs', 'DataSize', 'bandwidth', 'mec_capacity', 'min_semantic_factor'

# 基础参数设置 - 当selected_parameter不在列表中时使用的固定值
BASE_VALUES = {
    'DataSize': 150,                # 当测试其他参数时，数据大小的默认值 (KB)
    'num_UEs': 20,                  # 当测试其他参数时，UE数量的默认值
    'bandwidth': 1000,              # 当测试其他参数时，带宽的默认值 (kHz)
    'mec_capacity': 12.5,           # 当测试其他参数时，MEC容量的默认值 (GHz)
    'min_semantic_factor': 0.3      # 当测试其他参数时，最小语义因子的默认值
}