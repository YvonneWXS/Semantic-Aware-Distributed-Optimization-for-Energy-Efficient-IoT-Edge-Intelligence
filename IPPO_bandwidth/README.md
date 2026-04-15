# SA‑IPPO with DW‑DNA Bandwidth Allocation

Semantic‑Aware Independent Proximal Policy Optimization (SA‑IPPO) with Discrete Weight‑based Dynamic Normalized Allocation (DW‑DNA) for energy‑efficient IoT edge intelligence.

## 项目简介

本项目在原有 IPPO 算法基础上，完成了以下核心修改：

1. **动作空间扩展**：从原来的三维 `[卸载, 语义因子, MEC资源]` 扩展为四维 `[卸载, 语义因子, MEC资源, 带宽权重]`，删除了原有的传输功率维度。
2. **带宽分配机制**：采用 DW‑DNA (Discrete Weight‑based Dynamic Normalized Allocation) 机制，替代原有的静态均分带宽。智能体输出离散的带宽权重 `bw_weight ∈ {0,1,2,3}`，环境层通过全局归一化与向下取整量化，实现按需、离散的带宽分配。
3. **环境与奖励适配**：时延、能耗公式全部使用动态分配的带宽重新计算，保留了论文中的全局自适应惩罚奖励函数、语义提取能耗模型以及 MEC 资源归一化约束。
4. **批量实验支持**：提供完整的参数列表与实验入口，可系统性地考察不同数据量、用户数、带宽、MEC 能力、语义因子下限对算法性能的影响。

## 文件结构

```
SA_IPPO_DWDNA/
├── algorithms/           # IPPO 算法核心（未改动）
├── envs/
│   ├── env_core.py      # 环境核心，已实现 DW‑DNA 带宽分配
│   ├── env_discrete.py  # 离散动作环境封装，动作空间已扩展为四维
│   └── env_wrappers.py  # 环境包装器（未改动）
├── runner/              # 训练运行器（未改动）
├── train/
│   └── train.py         # 训练脚本（未改动，可直接使用）
├── utils/               # 工具函数（未改动）
├── config.py            # 原始配置解析器（未改动）
├── config_batch.py      # 批量实验参数列表 ★新增
├── main.py              # 批量实验入口 ★新增
├── README.md            # 本文件
└── Modification.md      # 详细修改记录 ★新增
```

## 运行环境

- Python 3.7+
- PyTorch 1.8+
- gym 0.21.0
- numpy

依赖安装（建议使用 conda 虚拟环境）：
```bash
pip install torch gym numpy
```

## 快速开始

### 单次训练
```bash
cd SA_IPPO_DWDNA
python train/train.py --env_name MyEnv --algorithm_name rmappo --experiment_name test --num_agents 5 --seed 1
```

### 批量实验（示例）
```bash
python main.py
```
注意：`main.py` 目前为参数遍历框架，实际运行前需根据需求取消注释 `run_experiment` 调用，并确保环境参数（如带宽、MEC 能力等）已正确传递。

## 核心参数说明

### 环境参数（env_core.py 中可调）
| 参数 | 含义 | 默认值 |
|------|------|--------|
| `agent_num` | 用户设备数量 | 5 |
| `transmission_bandwidth` | 系统总带宽 | 1 MHz |
| `MEC_f` | MEC 服务器计算能力 | 20 GHz |
| `transmission_power` | 固定发射功率 | 0.1 W |
| `RB_bandwidth` | 5G NR 资源块带宽 | 180 kHz |
| `κ` | 芯片结构影响因子 | 1e‑27 |
| `alpha`, `beta`, `r` | 语义提取任务参数 | 1, 2, 1 |

### 动作空间
| 维度 | 含义 | 离散值/范围 |
|------|------|-------------|
| 卸载决策 | 0=本地处理，1=卸载 | {0,1} |
| 语义因子 | 语义压缩率 | 0.3 ~ 1.0（8 个离散值） |
| MEC资源 | 分配给该任务的计算资源比例 | 0.1 ~ 1.0（10 个离散值） |
| 带宽权重 | 表达带宽需求强度 | {0,1,2,3} |

### 批量实验参数（config_batch.py）
| 参数列表 | 可选值 |
|----------|--------|
| `DataSize_list` | [128, 256, 512, 1024] KB |
| `num_UEs_list` | [5, 10, 15, 20, 25, 30] |
| `bandwidth_list` | [750, 1000, 1500, 2000] kHz |
| `mec_capacity_list` | [10.0, 12.5, 15.0, 17.5, 20.0] Gcps |
| `min_semantic_factor_list` | [0.2, 0.3, 0.4, 0.5] |

## DW‑DNA 带宽分配流程

1. **意向阶段**：每个智能体输出带宽权重 `w_i ∈ {0,1,2,3}`。
2. **归一化阶段**：计算理想分配比例 `β_i = w_i / (∑ w_j + ε)`。
3. **量子化阶段**：将理想带宽 `β_i · W_total` 向下取整到最近的 RB 整数倍，得到实际分配带宽 `B_i`。
4. **物理计算**：使用 `B_i` 计算上行速率，进而得到时延与能耗。

该机制模拟了 5G/6G 中 RB 资源的离散性，同时通过全局归一化自动避免了带宽超限，使智能体只需学习“需求强度”而非具体的数值分配。

## 奖励函数

奖励函数沿用原文的全局自适应惩罚形式：

```
reward = w1/w * (agent_num / E_total) + (w2/w) * total_time_penalty
```

其中：
- `E_total` 为当前 episode 所有 agent 的总能耗
- `total_time_penalty` 为所有 agent 的时间违反惩罚之和
- `w1, w2` 为权重系数（默认均为 1）
- `w = w1 + w2`

## 修改记录

详细修改内容、原因与影响见 [Modification.md](Modification.md)。

## 注意事项

1. 当前版本将传输功率固定为 0.1 W，若需改为可调参数，请在 `env_core.py` 中修改 `self.transmission_power`。
2. 批量实验脚本 `main.py` 目前仅展示参数遍历框架，实际运行前请根据实验设计调整环境参数的传递方式。
3. 如需改变语义因子、MEC 资源的离散化粒度，请同步修改 `env_discrete.py` 中的 `action_space_params` 与 `env_core.py` 中的 one‑hot 解析逻辑。
4. 训练过程中产生的日志与模型保存在 `results/` 目录下，可按实验名称区分。

## 引用

若使用本代码，请引用原始论文：
> [论文信息]

## 联系方式

如有问题或建议，请联系项目维护者。

---
**版权说明** 本项目基于原有 SA‑IPPO 代码修改，仅供学术研究使用。