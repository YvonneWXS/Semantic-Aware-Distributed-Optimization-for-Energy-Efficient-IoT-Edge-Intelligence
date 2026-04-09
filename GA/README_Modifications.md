# 代码修改说明

## 任务完成情况

### 1. 分析并修改发射功率引入
**分析结果：**
- 在 `genetic_algorithm.py` 中，发射功率已经在动作空间中引入：
  - `transmission_power_set` 定义了发射功率的离散值集合
  - `initialize_population` 函数初始化了每个用户的发射功率
  - `mutation` 函数变异发射功率值
  - `crossover` 函数交叉发射功率基因

**修改：**
- 根据用户要求，将发射功率固定为 0.1W：
  - 在 `genetic_algorithm.py` 中将 `transmission_power_set` 改为 `np.array([0.1])`
  - 在 `env.py` 中将 `transmission_power` 固定为 0.1W

### 2. 能耗指标加入奖励计算公式
**当前实现：**
- `fitness` 函数已经考虑了总能量消耗（包括传输能耗）
- `compute_energy_and_delay` 函数计算的总能量包含了：
  - 本地处理能量
  - 语义提取任务能量
  - 上传能量（与发射功率直接相关）

### 3. 根据用户提供的固定条件修改参数
**env.py 修改：**
1. 语义提取因子最低阈值：0.3 → 0.5
2. 传输带宽：1MHz → 2MHz (2000kHz)
3. 传输功率：随机0.1-0.5W → 固定0.1W
4. 最大容忍时延：随机取 → 固定100ms (0.1s)
5. 重置函数：接受特定DataSize值（KB单位）

**genetic_algorithm.py 修改：**
1. 语义因子集合：0.3-1.0 → 0.5-1.0
2. 传输功率集合：0.1-0.5W → 固定0.1W
3. 遗传算法函数添加`output_dir`参数

### 4. 创建实验运行脚本
**新增文件：** `experiment_runner.py`

**功能：**
1. 遍历DataSize列表：[250,500,750,1000,1250,1500,1750,2000] KB
2. 为每个实验创建独立的结果文件夹：`GA/result/DataSize_XXXKB/`
3. 保存详细的实验结果：
   - 实验参数设置
   - 最优解详细信息
   - 能量收敛历史
   - 汇总结果报告

**结果文件夹结构：**
```
GA/result/
├── DataSize_250KB/
│   ├── experiment_parameters.txt
│   ├── best_solution.csv
│   ├── energy_history.csv
│   └── summary_results.txt
├── DataSize_500KB/
├── ...
└── all_experiments_summary.csv
```

## 使用方法

### 运行单个实验（测试）
```bash
cd GA
python genetic_algorithm.py
```

### 运行完整实验集
```bash
cd GA
python experiment_runner.py
```

### 实验参数
所有实验使用以下固定参数：
- 用户数量 (N): 20
- 总带宽 (B): 2000 kHz (2 MHz)
- MEC服务器计算能力 (Fmec): 20.0 GHz
- 最小语义提取因子 (beta_min): 0.5
- 最大发射功率限制 (Pmax): 0.1 W
- 最大容忍时延 (Tmax): 100 ms
- 任务数据量 (Dn): 实验变量 [250,500,750,1000,1250,1500,1750,2000] KB

## 遗传算法参数
- 种群大小: 100
- 最大代数: 10000
- 早停阈值: 0.01
- 早停耐心值: 2000代

## 验证与约束
算法确保以下约束：
1. 延迟约束：总延迟 ≤ 100ms
2. 资源约束：计算资源分配总和 ≤ 1
3. 功率约束：发射功率 ≤ 0.1W
4. 语义因子约束：语义提取因子 ≥ 0.5（卸载时）