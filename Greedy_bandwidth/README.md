# 语义感知MEC系统改进Greedy算法 - 批量实验平台

## 项目概述

本项目实现了基于离散权重的动态归一化带宽分配机制（DW-DNA）的改进Greedy算法，用于语义感知MEC系统中联合优化卸载决策、语义提取因子、发射功率、MEC计算资源和动态带宽分配。

### 核心功能
- **5维联合优化**：支持卸载决策、语义因子、MEC资源、发射功率、带宽权重的同时优化
- **动态带宽分配**：基于离散权重的归一化分配机制，模拟5G RB资源块
- **批量实验**：自动遍历参数列表进行对比实验
- **参数化基础环境**：可在config.py中设置基础环境参数
- **完整性能评估**：输出能耗、时延、资源利用率等多项指标

## 文件结构

```
├── env.py                     # 环境定义文件（支持动态带宽分配）
├── SA_greedy_energy1st.py     # 改进的Greedy算法文件
├── config.py                 # 参数配置文件（支持参数列表和基础参数）
├── main_experiment.py        # 批量实验主程序
├── experiment_results.csv    # 批量实验结果文件
├── 改进Greedy算法详细说明文档.md  # 综合技术文档
└── README.md                 # 本快速入门指南
```

## 核心特性

### 1. 扩展动作空间
```
[offload, semantic, mec_resource, tx_power, bw_weight]  # 5维动作空间
```

### 2. DW-DNA带宽分配机制
- 意向阶段：每个UE表达带宽需求强度
- 归一化阶段：全局资源仲裁分配
- 量子化阶段：符合5G RB物理约束

### 3. 参数化批量实验
- **参数列表**：指定要变化的参数范围
- **基础参数**：指定非变化参数的固定值
- **自动遍历**：系统自动遍历参数列表进行对比实验

## 参数配置（config.py）

### 参数列表设置
```python
# 系统基本参数列表
DataSize_KB = [100, 200, 300, 400]          # 数据大小列表
num_UEs_list = [5, 10, 15, 20]              # UE数量列表
total_bandwidth_kHz = [750, 1000, 1250]      # 带宽列表
mec_capacity_GHz = [10.0, 12.5, 15.0]       # MEC容量列表
min_semantic_factor_list = [0.2, 0.3, 0.4, 0.5]  # 语义因子列表
```

### 基础参数设置
```python
# 基础参数 - 当selected_parameter不在列表中时使用的固定值
BASE_VALUES = {
    'DataSize': 150,                # 当测试其他参数时，数据大小的默认值 (KB)
    'num_UEs': 10,                  # 当测试其他参数时，UE数量的默认值
    'bandwidth': 1000,              # 当测试其他参数时，带宽的默认值 (kHz)
    'mec_capacity': 12.5,           # 当测试其他参数时，MEC容量的默认值 (GHz)
    'min_semantic_factor': 0.3      # 当测试其他参数时，最小语义因子的默认值
}
```

### 实验控制
```python
selected_parameter = 'num_UEs'      # 要测试的参数
num_runs_per_config = 10            # 每配置运行次数
```

## 支持的测试参数

- `num_UEs`：用户设备数量
- `DataSize`：数据大小
- `bandwidth`：系统带宽
- `mec_capacity`：MEC计算容量
- `min_semantic_factor`：最小语义提取因子

## 输出性能指标

实验自动输出以下指标：
- **能耗指标**：系统总能耗、平均能耗
- **时延指标**：平均总时延、传输时延、计算时延
- **资源指标**：带宽利用率、语义提取率、卸载比例

## 运行方法

### 1. 批量实验运行
- 配置参数列表和 `BASE_VALUES` 中的基础参数
- 设置 `selected_parameter` 来选择要测试的参数
- 运行 `python main_experiment.py`
- 结果保存到 `experiment_results.csv`

### 2. 自定义配置
在 `config.py` 中修改相应的参数即可。

## 结果文件

- `experiment_results.csv`：批量实验结果
- 直接用于数据分析和论文绘图

## 应用场景

- MEC系统性能评估
- 资源分配算法对比
- 参数敏感性分析
- 语义通信效果验证

---

**说明**：通过配置 `config.py` 中的 `selected_parameter` 来指定要测试的参数，其他参数使用 `BASE_VALUES` 中的默认值。