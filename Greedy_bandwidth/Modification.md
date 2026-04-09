# 语义感知MEC系统改进Greedy算法综合文档

## 1. 项目概述

本项目实现了基于离散权重的动态归一化带宽分配机制（DW-DNA）的改进Greedy算法，用于语义感知MEC系统中联合优化卸载决策、语义提取因子、发射功率、MEC计算资源和动态带宽分配，最小化UE总能耗并满足时延约束。

### 核心特性：
- 支持5维动作空间联合优化
- 基于离散权重的动态带宽分配机制
- 自动化批量对比实验功能
- 符合5G物理约束的RB资源块模拟
- **新增**：参数化的基础环境配置

## 2. 文件结构

```
├── env.py                     # 环境定义文件（支持动态带宽分配）
├── SA_greedy_energy1st.py     # 改进的Greedy算法文件
├── config.py                 # 参数配置文件（支持参数列表和基础参数）
├── main_experiment.py        # 批量实验主程序
├── experiment_results.csv    # 批量实验结果文件
├── 改进Greedy算法详细说明文档.md  # 本综合文档
└── README.md                 # 快速入门指南
```

## 3. 核心改进

### 3.1 动作空间扩展
- **原动作空间**：[offload, semantic, mec_resource]（3维）
- **新动作空间**：[offload, semantic, mec_resource, tx_power, bw_weight]（5维）
- **带宽权重定义**：bw_weight ∈ {0, 1, 2, 3}
  - `bw_weight = 0`：不卸载或不需要带宽
  - `bw_weight ∈ {1, 2, 3}`：卸载时的带宽需求强度

### 3.2 动态带宽分配机制（DW-DNA）
```
1. 意向阶段：每个UE输出离散权重 w_i ∈ {0, 1, 2, 3}
2. 归一化阶段：β_i = w_i / Σ(w_j) * Total_Bandwidth (ε防止分母为0)
3. 量子化阶段：B_i = floor((β_i * W) / Δb) * Δb
```
- 最小资源单位：10kHz（模拟5G RB资源块）
- 向下取整：模拟物理资源块不可分割特性
- 余量处理：剩余带宽视为系统损耗/保护间隔

### 3.3 能耗与延迟计算
- **传输速率**：`uplink_rate_i = B_i * log2(1 + P_i * h_i / (B_i * N₀))`
- **传输能耗**：`upload_energy_i = P_i * (SEF_i * D_i) / uplink_rate_i`
- **总能耗**：`total_energy_i = semantic_extraction_energy_i + upload_energy_i (if offload)`
- **总时延**：`total_delay_i = semantic_extraction_delay_i + transmission_delay_i + mec_computation_delay_i`

## 4. 批量实验功能

### 4.1 参数配置（config.py）
```python
# 系统基本参数 - 用于批量实验的参数列表
DataSize_KB = [100, 200, 300, 400]        # 单UE数据大小范围 (KB)
num_UEs_list = [5, 10, 15, 20]            # 用户设备数量列表
total_bandwidth_kHz = [750, 1000, 1250]     # 系统总带宽列表 (kHz)
mec_capacity_GHz = [10.0, 12.5, 15.0]     # MEC容量列表 (GHz)

# 基础参数设置 - 当selected_parameter不在列表中时使用的固定值
BASE_VALUES = {
    'DataSize': 150,                # 当测试其他参数时，数据大小的默认值 (KB)
    'num_UEs': 10,                  # 当测试其他参数时，UE数量的默认值
    'bandwidth': 1000,              # 当测试其他参数时，带宽的默认值 (kHz)
    'mec_capacity': 12.5,           # 当测试其他参数时，MEC容量的默认值 (GHz)
    'min_semantic_factor': 0.3      # 当测试其他参数时，最小语义因子的默认值
}
```

### 4.2 实验控制参数
```python
num_runs_per_config = 10                    # 每配置运行次数
selected_parameter = 'num_UEs'              # 指定要测试的参数
```

### 4.3 支持的测试参数
- `'num_UEs'`：用户设备数量
- `'DataSize'`：数据大小
- `'bandwidth'`：带宽
- `'mec_capacity'`：MEC容量
- `'min_semantic_factor'`：最小语义因子

### 4.4 性能指标输出
批量实验自动输出以下指标：
- `num_UEs`: 用户设备数量
- `data_size_KB`: 数据大小 (KB)
- `total_bandwidth_kHz`: 总带宽 (kHz)
- `mec_capacity_GHz`: MEC容量 (GHz)
- `total_energy`: 系统总能耗
- `avg_delay`: 平均总时延
- `avg_tx_delay`: 平均传输时延
- `avg_comp_delay`: 平均计算时延
- `bandwidth_utilization`: 平均带宽利用率
- `avg_semantic_factor`: 平均语义提取因子
- `offload_ratio`: 卸载比例

## 5. 运行方法

### 5.1 批量实验运行
1. 配置参数列表和 `BASE_VALUES` 中的基础参数
2. 设置 `selected_parameter` 来选择要测试的参数
3. 运行 `python main_experiment.py`
4. 结果保存到 `experiment_results.csv`

### 5.2 参数配置示例
```python
# 基础参数设置
BASE_VALUES = {
    'DataSize': 200,              # 200KB数据大小（当不测试数据大小时）
    'num_UEs': 15,               # 15个用户（当不测试用户数时）
    'bandwidth': 1250,           # 1250kHz带宽（当不测试带宽时）
    'mec_capacity': 15.0,        # 15.0GHz MEC容量（当不测试MEC容量时）
    'min_semantic_factor': 0.4   # 最小语义因子0.4（当不测试语义因子时）
}

# 批量实验参数
num_UEs_list = [5, 10, 15, 20, 25]        # 测试不同用户数
selected_parameter = 'num_UEs'              # 选择测试用户数
```

## 6. 算法特性

### 6.1 优化策略
- 贪心策略：逐个UE优化，考虑全局资源约束
- 预校验机制：先检查时延约束再计算MEC资源
- 多维联合优化：同时优化5个决策变量

### 6.2 鲁棒性设计
- 数值稳定性：防除零错误、边界检查
- 资源约束：MEC资源和带宽资源严格守恒
- 算法收敛：平滑梯度提供稳定优化方向

### 6.3 物理真实性
- RB资源块模拟：向下取整符合5G标准
- 信道增益模型：路径损耗和阴影衰落
- 功率约束：符合实际发射功率范围

## 7. 应用场景

### 7.1 性能评估
- 不同网络负载下的能耗性能
- 不同资源配置下的时延表现
- 语义压缩对系统性能的影响

### 7.2 参数敏感性分析
- 用户密度对系统性能的影响
- 带宽资源对卸载决策的影响
- MEC容量对计算延迟的影响

### 7.3 算法对比
- 与其他卸载算法的性能对比
- 不同带宽分配策略的比较
- 语义感知vs传统算法的差异

## 8. 使用建议

### 8.1 实验设计
1. 确保参数范围符合实际网络条件
2. 验证时延约束设置的合理性
3. 考虑功率和带宽资源的实际限制
4. 选择合适的 `num_runs_per_config` 以平衡精度和效率

### 8.2 参数设置
- 在 `BASE_VALUES` 中设置非变量的基础环境参数
- 在相应列表中设置要变化的参数
- 调整 `selected_parameter` 来指定当前测试的参数

### 8.3 结果分析
1. 关注关键性能指标的趋势变化
2. 识别系统瓶颈和优化机会
3. 验证算法在极端条件下的表现

## 9. 扩展方向

### 9.1 算法改进
- 引入机器学习算法提高优化精度
- 考虑用户移动性对资源分配的影响
- 支持多MEC服务器的协作计算

### 9.2 功能扩展
- 支持更多类型的服务质量需求
- 增加安全性考虑（加密计算开销）
- 考虑网络拓扑对性能的影响

---

本项目为语义感知MEC系统提供了一套完整的仿真平台，支持参数化的批量对比实验，通过统一的配置管理实现了灵活性与易用性的完美结合。