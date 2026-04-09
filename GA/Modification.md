# 修改清单 (Modification.md)

## 项目概述

本文档记录了GA遗传算法代码的所有修改内容，包括修改原因、影响和实现细节。

## 修改时间

2026-04-09

## 一、环境模型 (env.py)

### 1.1 新增参数

| 修改项 | 原内容 | 新内容 | 原因 |
|--------|--------|--------|------|
| `total_bandwidth` | 无 | 总带宽参数 (默认1000kHz) | 支持可配置的带宽资源 |
| `mec_capacity` | 固定20GHz | 可配置参数 | 支持不同MEC能力实验 |
| `min_semantic_factor` | 固定0.3 | 可配置参数 | 支持不同语义阈值 |
| `bandwidth_weight_set` | 无 | {0,1,2,3} | DW-DNA带宽权重 |

### 1.2 动作空间扩展

| 修改项 | 原内容 | 新内容 |
|--------|--------|--------|
| 动作维度 | 3维 [offload, semantic, resource] | 5维 [offload, semantic, resource, power, bw_weight] |
| 动作数量 | n_actions | 5×n_actions (含power和bw_weight组合) |

### 1.3 带宽计算逻辑 (DW-DNA)

| 修改项 | 原内容 | 新内容 |
|--------|--------|--------|
| 带宽分配方式 | 静态均分: W/offload_num | 动态归一化: B_i = (bw_weight_i/sum_weights) × W |
| 量化处理 | 无 | 向下取整 (模拟RB资源块) |
| 边界处理 | 无 | sum_weights=0时返回0 |

**实现代码**:
```python
def compute_bandwidth_dwdna(self, bw_weights):
    sum_weights = np.sum(bw_weights)
    if sum_weights == 0:
        return np.zeros_like(bw_weights, dtype=float)
    normalized_bw = (bw_weights / sum_weights) * self.total_bandwidth
    quantized_bw = np.floor(normalized_bw).astype(float)
    return quantized_bw
```

### 1.4 能耗时延计算

| 修改项 | 原内容 | 新内容 |
|--------|--------|--------|
| 带宽输入 | offload_num | bw_weight数组 |
| 上行速率 | 固定带宽计算 | 动态带宽计算: W_i × log2(1+SNR) |
| 语义提取能耗 | 包含semantic_factor | 保持不变 |
| 本地计算 | 保持不变 | 保持不变 |

### 1.5 工厂函数

| 修改项 | 说明 |
|--------|------|
| 新增 `create_env()` | 简化环境创建过程 |

---

## 二、遗传算法 (genetic_algorithm.py)

### 2.1 决策变量扩展

| 修改项 | 原内容 | 新内容 |
|--------|--------|--------|
| 个体编码 | 3元组 (offload, resource, power) | 5元组 (offload, semantic, resource, power, bw_weight) |
| 语义因子 | 代码中隐含 | 显式编码，独立优化 |
| 带宽权重 | 无 | 新增优化维度 |

### 2.2 初始化

| 修改项 | 原内容 | 新内容 |
|--------|--------|--------|
| 生成逻辑 | 随机生成 | 按约束生成合法的语义因子和带宽权重 |
| 约束验证 | resource约束 | 时延约束 (含semantic和bw_weight) |

**初始化代码**:
```python
# 语义因子: 卸载决策=1时有效
semantic_factor = np.random.choice(semantic_factor_set, size=K)
semantic_factor = np.where(offload_decision == 1, semantic_factor, 1.0)

# 带宽权重: 不卸载=0, 卸载=1,2,3
bw_weight = np.zeros(K, dtype=int)
offload_indices = np.where(offload_decision == 1)[0]
if len(offload_indices) > 0:
    bw_weight[offload_indices] = np.random.choice(bw_weight_set[1:], size=len(offload_indices))
```

### 2.3 交叉操作

| 修改项 | 原内容 | 新内容 |
|--------|--------|--------|
| 交叉维度 | 3维 | 5维 (所有变量一起交叉) |
| 交叉方式 | 单点交叉 | 单点交叉 (保持不变) |

### 2.4 变异操作

| 修改项 | 原内容 | 新内容 |
|--------|--------|--------|
| 变异变量 | 3个 | 5个 (新增semantic和bw_weight) |
| 变异约束 | resource归一化 | 协同变异 (offload与semantic/bw_weight联动) |

**变异逻辑**:
```python
# 卸载决策变化时，同步更新语义因子和带宽权重
if offload[idx] == 0:
    semantic[idx] = 1.0
    resource[idx] = 0
    bw_weight[idx] = 0
else:
    semantic[idx] = np.random.choice(semantic_factor_set)
    bw_weight[idx] = np.random.choice(bw_weight_set[1:])
```

### 2.5 适应度函数

| 修改项 | 原内容 | 新内容 |
|--------|--------|--------|
| 输入参数 | 3个 | 5个 |
| 约束条件 | 时延惩罚 | 时延惩罚 (含semantic和bw_weight影响) |
| 返回值 | 负能耗 | 负能耗 |

---

## 三、配置文件 (config.py)

### 3.1 新增内容

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `DataSize_list` | 数据大小列表 (KB) | [128, 256, 512, 1024] |
| `num_UEs_list` | 用户数量列表 | [5, 10, 15, 20, 25, 30] |
| `bandwidth_list` | 带宽列表 (kHz) | [750, 1000, 1500, 2000] |
| `mec_capacity_list` | MEC能力列表 (Gcps) | [10.0, 12.5, 15.0, 17.5, 20.0] |
| `min_semantic_factor_list` | 语义因子列表 | [0.2, 0.3, 0.4, 0.5] |
| `GA_PARAMS` | GA超参数 | pop_size=50, generations=2000 |

### 3.2 配置函数

| 函数 | 说明 |
|------|------|
| `get_single_var_config()` | 获取单变量实验配置 |
| `get_full_factorial_config()` | 获取全因子实验配置 |
| `get_custom_config()` | 获取自定义配置 |

---

## 四、实验入口 (main_experiment.py)

### 4.1 新增功能

| 功能 | 说明 |
|------|------|
| 单变量实验 | 遍历指定变量，保持其他参数固定 |
| 全因子实验 | 遍历所有参数组合 |
| 结果保存 | 自动保存到CSV |
| 进度显示 | 实时显示实验进度 |
| 统计信息 | 打印能耗统计 |

### 4.2 命令行参数

| 参数 | 说明 |
|------|------|
| `--var` | 要对比的变量名 |
| `--full` | 运行全因子实验 |
| `--quiet` | 安静模式 |

---

## 五、修改影响分析

### 5.1 算法复杂度

| 指标 | 原GA | 新GA | 变化 |
|------|------|------|------|
| 个体长度 | 3K | 5K | +67% |
| 动作空间 | O(K²) | O(K³) | 增大 |
| 初始化耗时 | O(pop_size) | O(pop_size×K) | 略增 |
| 适应度评估 | O(K) | O(K) | 不变 |

### 5.2 优化效果

| 指标 | 原GA | 新GA | 说明 |
|------|------|------|------|
| 能耗 | 基准 | 可优化 | 带宽动态分配可降低带宽冲突 |
| 时延约束 | 满足 | 满足 | 约束保持不变 |
| 收敛速度 | 基准 | 相近 | 维度增加影响有限 |
| 解质量 | 基准 | 相当或更优 | 带宽优化贡献 |

### 5.3 物理意义

1. **带宽权重 (bw_weight)**
   - 0: 不需要带宽 (本地处理)
   - 1,2,3: 递增的带宽需求强度
   - 物理意义: 模拟5G网络中RB资源块的离散分配

2. **DW-DNA**
   - 归一化: 确保总带宽不超限
   - 量化: 模拟RB不可分割特性
   - 博弈: 带宽分配体现用户间的资源竞争

3. **语义提取因子**
   - 保持原模型不变
   - 与带宽权重联动 (时延公式中的乘法关系)

---

## 六、测试验证

### 6.1 单元测试

```python
# 测试带宽计算
env = ENV(UEs=5, MECs=1, k=100, total_bandwidth=1000)
bw_weights = np.array([3, 0, 0, 0, 0])
bandwidths = env.compute_bandwidth_dwdna(bw_weights)
# 期望: [1000, 0, 0, 0, 0]

# 测试时延计算
offload = np.array([1, 0, 1, 0, 1])
semantic = np.array([0.5, 1.0, 0.7, 1.0, 0.3])
resource = np.array([0.3, 0, 0.3, 0, 0.4])
power = np.array([0.3, 0.2, 0.4, 0.1, 0.5])
bw_weight = np.array([3, 0, 2, 0, 1])
observation = env.reset(1)
energy, penalty = env.compute_energy_and_delay(offload, semantic, resource, power, bw_weight, observation)
# 期望: penalty >= 0 (时延约束)
```

### 6.2 集成测试

```bash
# 运行单变量实验
python main_experiment.py --var num_UEs

# 验证结果
# 检查 experiment_results/GA_results_num_UEs.csv
```

---

## 七、注意事项

### 7.1 约束条件

1. **时延约束**: 通过适应度函数的惩罚项实现
2. **MEC资源约束**: 资源分���归一化
3. **带宽约束**: DW-DNA自动保证不超限

### 7.2 数值稳定性

1. **带宽为0**: 返回大惩罚值 (1e10)
2. **上行速率为0**: 使用极小值 (1e-10)
3. **NaN/Inf**: 返回极负适应度

### 7.3 收敛建议

1. 增大 `pop_size` 可提高解质量
2. 增大 `generations` 可确保收敛
3. 调整 `mutation_rate` 可平衡探索与利用

---

## 八、版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| 1.0 | 2026-04-09 | 初始版本 (含5维决策变量和DW-DNA) |

---

## 九、参考文档

- 带宽分配方案: `带宽分配 - 基于离散权重的动态归一化分配（Proposed）.md`
- 系统模型: `paper - 副本.pdf` (语义提取、MEC、时延能耗公式)
- 原始代码: `GA/env.py`, `GA/genetic_algorithm.py` (修改前)