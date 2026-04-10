# 语义感知遗传算法 (SA-GA) 项目

## 项目概述

本项目实现了基于离散权重的动态归一化带宽分配（DW-DNA）的语义感知遗传算法（SA-GA），用于优化物联网边缘计算场景中的能效与延迟约束问题。

### 核心特性

1. **语义感知优化**：结合语义提取模型，在保证任务语义准确性的前提下优化能效
2. **离散权重带宽分配**：采用DW-DNA机制动态分配带宽，模拟5G/6G网络中的RB资源块分配
3. **多变量联合优化**：同时优化卸载决策、语义提取因子、计算资源分配、传输功率和带宽权重
4. **批量实验支持**：支持多参数组合的批量实验，便于系统性能分析

## 文件结构

```
SA_GA/
├── env.py                    # 环境模型，包含带宽分配和能耗延迟计算
├── genetic_algorithm.py      # SA-GA遗传算法实现
├── config.py                 # 批量实验参数配置
├── main.py                   # 批量实验入口文件
├── README.md                 # 本项目说明文档
└── Modification.md           # 修改内容和影响说明
```

## 运行环境要求

### 软件依赖
- Python 3.7+
- NumPy 1.19+
- pandas 1.3+

### 安装依赖
```bash
pip install numpy pandas
```

## 快速开始

### 1. 运行单次实验
```bash
cd SA_GA
python genetic_algorithm.py
```

### 2. 运行批量实验
```bash
cd SA_GA
python main.py
```

### 3. 自定义实验配置
编辑 `config.py` 文件中的参数列表：
```python
# 批量实验参数列表
DataSize_list = [128, 256, 512, 1024]  # 任务数据大小 (KB)
num_UEs_list = [5, 10, 15, 20, 25, 30]  # 用户设备数量
bandwidth_list = [750, 1000, 1500, 2000]  # 总带宽 (kHz)
mec_capacity_list = [10.0, 12.5, 15.0, 17.5, 20.0]  # MEC服务器计算能力 (Giga Cycles/s)
min_semantic_factor_list = [0.2, 0.3, 0.4, 0.5]  # 最小语义提取因子
```

## 算法参数说明

### 遗传算法参数（config.py中的FixedConfig类）
- `pop_size`: 种群大小（默认：100）
- `generations`: 最大迭代次数（默认：3000）
- `crossover_rate`: 交叉概率（默认：0.89）
- `mutation_rate`: 变异概率（默认：0.9）
- `early_stop_patience`: 早停耐心值（默认：500）
- `early_stop_threshold`: 早停阈值（默认：0.01）
- `num_trials`: 每个配置的试验次数（默认：50）

### 动作空间参数
- `semantic_factor_set`: 语义提取因子集合 [0.3, 0.4, ..., 1.0]
- `resource_allocation_set`: 资源分配比例集合 [0.1, 0.2, ..., 1.0]
- `transmission_power_set`: 传输功率集合 [0.1, 0.2, ..., 0.5]
- `bw_weight_set`: 带宽权重集合 [0, 1, 2, 3]

### 环境参数
- `transmission_bandwidth`: 总传输带宽（可配置）
- `MEC_f`: MEC服务器计算能力（可配置）
- `delta_b`: 资源块大小（默认：1 kHz）
- `kappa`: 芯片结构因子（默认：1e-27）
- `noise_power`: 噪声功率（默认：1e-20 W）

## 核心算法修改

### 1. 动作空间扩展
- 新增离散带宽权重：`bw_weight ∈ {0,1,2,3}`
- 个体编码从4变量扩展为5变量：`(offload, semantic, resource, power, bw_weight)`
- 一致性约束：`offload=0 → bw_weight=0`，`offload=1 → bw_weight∈{1,2,3}`

### 2. 带宽分配机制（DW-DNA）
```python
def allocate_bandwidth(self, bw_weights):
    sum_weights = np.sum(bw_weights)
    if sum_weights < epsilon:
        return np.zeros_like(bw_weights)
    normalized_weights = bw_weights / sum_weights
    ideal_bandwidth = normalized_weights * total_bandwidth
    allocated_bandwidth = np.floor(ideal_bandwidth / delta_b) * delta_b
    return allocated_bandwidth
```

### 3. 遗传操作适配
- **初始化**：随机生成满足一致性约束的`bw_weight`
- **交叉**：所有5个变量一起进行单点交叉
- **变异**：`bw_weight`独立变异，保持与`offload`的一致性
- **适应度**：优先满足时延约束，再最小化总能耗

## 结果分析

### 输出文件
批量实验将生成以下文件：
1. `results_YYYYMMDD_HHMMSS/experiment_results.csv` - 汇总结果
2. `results_YYYYMMDD_HHMMSS/experiment_details.csv` - 详细结果（可选）
3. `results_YYYYMMDD_HHMMSS/experiment_summary.txt` - 实验总结报告

### 结果字段说明
- `experiment_id`: 实验唯一标识
- `data_size_kb`: 任务数据大小（KB）
- `num_UEs`: 用户设备数量
- `bandwidth_kHz`: 总带宽（kHz）
- `mec_capacity_ghz`: MEC计算能力（Giga Cycles/s）
- `min_semantic_factor`: 最小语义提取因子
- `avg_energy`: 平均能耗
- `std_energy`: 能耗标准差
- `min_energy`: 最小能耗
- `max_energy`: 最大能耗
- `num_successful_trials`: 成功试验次数
- `total_trials`: 总试验次数

## 性能评估

### 收敛性
算法包含早停机制，当连续500代最佳能耗改进小于1%时停止迭代。

### 约束处理
适应度函数优先满足时延约束，违反约束的个体被赋予极低适应度（-1e10）。

### 带宽分配效果
DW-DNA机制能够：
1. 避免带宽浪费：空闲用户的带宽权重为0，带宽分配给有需求的用户
2. 防止死锁：高竞争时自动退化为公平分配
3. 模拟物理约束：通过向下取整模拟RB资源块的离散特性

## 扩展与定制

### 1. 修改环境参数
编辑`env.py`中的`__init__`方法，调整：
- 用户设备参数范围
- 信道模型参数
- 任务生成分布

### 2. 自定义遗传操作
修改`genetic_algorithm.py`中的：
- `selection()`: 选择策略
- `crossover()`: 交叉策略
- `mutation()`: 变异策略

### 3. 添加新优化目标
修改`fitness()`函数，加入新的优化目标（如成本、可靠性等）。

## 故障排除

### 常见问题
1. **导入错误**：确保所有文件在同一目录下
2. **内存不足**：减少`pop_size`或`generations`参数
3. **收敛缓慢**：调整`mutation_rate`和`crossover_rate`
4. **所有试验失败**：检查约束条件是否过于严格

### 调试建议
1. 在`genetic_algorithm.py`中启用详细日志
2. 使用小规模配置测试算法基本功能
3. 检查`bw_weight`与`offload`的一致性约束

## 参考文献

1. 语义提取模型（SEF）参考论文：paper.pdf
2. 带宽分配机制：DW-DNA（基于离散权重的动态归一化分配）
3. 遗传算法基础：标准遗传算法与约束处理技术

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请参考项目文档或联系开发者。