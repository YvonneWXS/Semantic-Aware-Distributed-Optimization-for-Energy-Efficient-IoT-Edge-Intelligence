# GA 遗传算法 - 语义感知分布式边缘计算优化

## 项目简介

本项目实现了基于遗传算法（Genetic Algorithm, GA）的语义感知分布式优化方案，用于解决边缘计算系统中的能耗最小化问题。

### 核心功能

- **5维决策变量优化**: 卸载决策、语义提取因子、MEC资源分配、传输功率、带宽权重
- **DW-DNA带宽分配**: 基于离散权重的动态归一化分配机制
- **时延约束**: 满足任务最大容忍时延的前提下最小化总能耗
- **批量实验**: 支持多种参数组合的自动实验

## 文件结构

```
GA/
├── env.py                 # 环境模型 (已修改)
├── genetic_algorithm.py  # 遗传算法实现 (已修改)
├── config.py            # 批量实验配置
├── main_experiment.py   # 实验运行入口
├── README.md          # 项目说明
└── Modification.md   # 修改清单
```

## 安装依赖

```bash
pip install numpy
```

## 快速开始

### 1. 单次实验

```python
from genetic_algorithm import run_single_experiment

# 运行单次实验
result, solution, history = run_single_experiment(
    data_size=256,           # 数据大小 (KB)
    num_UEs=5,             # 用户数量
    bandwidth=1000,         # 总带宽 (kHz)
    mech_capacity=20.0,      # MEC计算能力 (Giga Cycles/s)
    min_semantic_factor=0.3,  # 最小语义提取因子
    a=1,                   # 实验编号
    pop_size=50,            # 种群大小
    generations=2000         # 进化代数
)

print(f"总能耗: {result['total_energy']:.6f}")
print(f"卸载数量: {result['offload_count']}")
```

### 2. 批量实验

```bash
# 运行单变量对比实验
python main_experiment.py --var num_UEs

# 运行全因子实验
python main_experiment.py --full
```

### 3. 自定义配置

在 `config.py` 中修改参数:

```python
# 要对比的变量
VAR_TO_COMPARE = 'num_UEs'  # 可选: data_size, num_UEs, bandwidth, mec_capacity, min_semantic_factor

# 固定参数
DEFAULT_PARAMS = {
    'data_size': 256,
    'num_UEs': 5,
    'bandwidth': 1000,
    'mec_capacity': 20.0,
    'min_semantic_factor': 0.3,
}

# GA超参数
GA_PARAMS = {
    'pop_size': 50,
    'generations': 2000,
    'crossover_rate': 0.8,
    'mutation_rate': 0.3,
}
```

## 参数说明

### 环境参数

| 参数 | 说明 | 单位 | 默认值 |
|------|------|------|--------|
| `UEs` | 用户设备数量 | 个 | 5 |
| `total_bandwidth` | 总带宽 | kHz | 1000 |
| `mec_capacity` | MEC计算能力 | Giga Cycles/s | 20.0 |
| `min_semantic_factor` | 最小语义提取因子 | - | 0.3 |

### GA参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `pop_size` | 种群大小 | 50 |
| `generations` | 最大进化代数 | 2000 |
| `crossover_rate` | 交叉概率 | 0.8 |
| `mutation_rate` | 变异概率 | 0.3 |
| `patience` | 早停耐心值 | 500 |

### 动作空间

| 维度 | 变量 | 取值范围 |
|------|------|---------|
| 1 | 卸载决策 | {0, 1} |
| 2 | 语义提取因子 | [0.3, 1.0] (离散8值) |
| 3 | MEC资源分配 | [0.1, 1.0] (离散10值) |
| 4 | 传输功率 | [0.1, 0.5] (离散5值) |
| 5 | 带宽权重 | {0, 1, 2, 3} |

## 实验结果

结果保存在 `experiment_results/` 目录:

- `GA_results_*.csv`: 能耗结果
- 每行包含: data_size, num_UEs, bandwidth, mec_capacity, min_semantic_factor, total_energy, delay_penalty, offload_count, local_count

## 算法流程

```
1. 初始化种群
   -> 随机生成5维决策变量
   -> 验证时延约束

2. 选择 (轮盘赌)
   -> 基于适应度值选择父个体

3. 交叉 (单点交叉)
   -> 交换父个体部分基因

4. 变异 (保持约束)
   -> 随机变异各维度
   -> 验证约束合法性

5. 迭代进化
   -> 重复步骤2-4
   -> 早停检查

6. 输出最优解
```

## 参考文献

- 带宽分配方案: 基于离散权重的动态归一化分配 (DW-DNA)
- 系统模型: 论文中的语义提取、MEC执行、时延能耗公式

## 联系方式

如有问题，请联系项目维护者。