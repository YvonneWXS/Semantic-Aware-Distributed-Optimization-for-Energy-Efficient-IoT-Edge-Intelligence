# SA-MAPPO: Semantic-Aware Multi-Agent Proximal Policy Optimization for IoT Edge Computing

## 概述

SA-MAPPO（语义感知多智能体近端策略优化）是一个用于物联网边缘计算卸载的强化学习框架。本项目在原始MAPPO算法基础上进行了扩展，引入了语义感知能力和动态带宽分配机制，以优化边缘计算环境中的能耗、延迟和公平性。

## 关键特性

1. **DW-DNA带宽分配**：基于深度加权网络分配（Deep Weighted Network Allocation）的动态带宽分配机制，模拟5G资源块（RB）分配
2. **4D动作空间**：扩展的动作空间 `[offload_decision, semantic_factor, resource_allocation, bandwidth_weight]`
3. **语义提取模型**：基于任务特征的语义压缩，减少传输数据量
4. **批量实验支持**：支持参数网格搜索和批量实验运行
5. **公平性优化**：考虑用户间公平性的多目标优化

## 安装

### 依赖项

```bash
# 基础依赖
pip install torch numpy gym matplotlib tensorboard

# 可选：用于实验跟踪
pip install wandb
```

### 环境设置

```bash
# 克隆项目
git clone <repository-url>
cd SA_MAPPO_5

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 使用方法

### 单次实验

运行单次训练实验：

```bash
python train/train.py --algorithm_name mappo --env_name MyEnv --experiment_name test_run
```

### 批量实验

使用批量实验运行器进行参数网格搜索：

```bash
python main.py --mode batch --grid_size small
```

可用模式：
- `single`: 单次实验（默认）
- `batch`: 批量实验
- `grid`: 参数网格搜索

### 关键参数

#### 环境参数
- `--agent_num`: 智能体数量（默认：5）
- `--episode_length`: 每回合步数（默认：50）
- `--use_obs_instead_of_state`: 使用观测而非状态（默认：False）

#### 算法参数
- `--algorithm_name`: 算法名称（mappo, rmappo等）
- `--lr`: 学习率（默认：5e-5）
- `--ppo_epoch`: PPO更新轮数（默认：15）
- `--num_env_steps`: 总环境步数（默认：800000）

#### 网络参数
- `--hidden_size`: 隐藏层大小（默认：64）
- `--layer_N`: 网络层数（默认：1）
- `--use_recurrent_policy`: 使用循环策略（默认：False）

## 文件结构

```
SA_MAPPO_5/
├── algorithms/          # MAPPO算法实现
│   ├── algorithm/      # 算法核心
│   │   ├── rMAPPOPolicy.py
│   │   ├── r_actor_critic.py
│   │   └── r_mappo.py
│   └── utils/          # 工具函数
├── envs/               # 环境定义
│   ├── env_core.py     # 核心环境（包含DW-DNA带宽分配）
│   ├── env_discrete.py # 离散动作空间环境
│   ├── env_continuous.py # 连续动作空间环境
│   └── env_wrappers.py # 环境包装器
├── train/              # 训练脚本
│   └── train.py        # 主训练脚本
├── runner/             # 训练运行器
│   ├── separated/      # 分离策略运行器
│   └── shared/         # 共享策略运行器
├── utils/              # 工具函数
│   ├── separated_buffer.py
│   ├── shared_buffer.py
│   ├── util.py
│   └── valuenorm.py
├── scripts/            # 辅助脚本
│   └── render/         # 渲染脚本
├── config.py           # 配置解析器
├── main.py             # 批量实验入口
├── plot.py             # 结果可视化
├── README.md           # 本文档
└── Modification.md     # 修改记录文档
```

## 核心算法

### 动作空间设计

智能体的动作空间为4维：
1. **卸载决策** (offload_decision): 0=本地处理, 1=边缘卸载
2. **语义因子** (semantic_factor): 语义压缩比例 (0.1-0.9)
3. **资源分配** (resource_allocation): 计算资源分配比例 (0.1-1.0)
4. **带宽权重** (bandwidth_weight): 带宽分配权重 (0-3)

### DW-DNA带宽分配

带宽分配算法基于深度加权网络分配：

```python
def _allocate_bandwidth_dwdna(self, bandwidth_weights, offload_decisions):
    total_bandwidth = self.transmission_bandwidth
    rb_size = 180 * self.kHz  # 5G资源块大小
    
    # 仅对卸载用户计算权重
    offloading_weights = [bw_weight * offload for bw_weight, offload in zip(bandwidth_weights, offload_decisions)]
    total_weight = sum(offloading_weights)
    
    if total_weight == 0:
        return [0] * self.agent_num
    
    # 基于权重的带宽分配（向下取整到RB倍数）
    bandwidths = []
    for weight, offload in zip(bandwidth_weights, offload_decisions):
        if offload == 1 and weight > 0:
            normalized_share = weight / total_weight
            ideal_bandwidth = normalized_share * total_bandwidth
            num_rbs = max(0, int(ideal_bandwidth // rb_size))
            allocated_bandwidth = num_rbs * rb_size
        else:
            allocated_bandwidth = 0
        bandwidths.append(allocated_bandwidth)
    
    return bandwidths
```

### 奖励函数

多目标奖励函数考虑：
1. **能耗**: 本地计算能耗 + 传输能耗
2. **延迟**: 处理延迟 + 传输延迟
3. **公平性**: 用户间资源分配公平性
4. **语义效率**: 语义压缩带来的效率增益

## 实验结果

### 性能指标

1. **平均奖励**: 每回合平均累积奖励
2. **能耗效率**: 每比特数据的能耗
3. **延迟满足率**: 满足最大延迟约束的任务比例
4. **公平性指数**: 用户间资源分配的公平性

### 可视化

使用plot.py脚本可视化训练结果：

```bash
python plot.py --log_dir ./results --save_dir ./plots
```

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{sa_mappo_2025,
  title = {SA-MAPPO: Semantic-Aware Multi-Agent PPO for IoT Edge Computing},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo/sa-mappo}
}
```

## 许可证

本项目基于MIT许可证开源。详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request。请确保：
1. 代码符合PEP8规范
2. 添加适当的测试
3. 更新相关文档

## 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱: your.email@example.com
- GitHub Issues: [项目Issues页面](https://github.com/your-repo/sa-mappo/issues)

## 致谢

本项目基于以下开源项目构建：
- [MAPPO](https://github.com/marlbenchmark/on-policy): 多智能体PPO实现
- [PyTorch](https://pytorch.org/): 深度学习框架
- [Gym](https://gym.openai.com/): 强化学习环境接口