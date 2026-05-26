# 代码修改计划

> 最新修改日期：26/4/10 16/52

## 1. 基本条件

+ 论文原文.pdf：论文原文
+ 修改思路.docx：审稿人意见与导师的意见
+ 带宽分配 - 基于离散权重的动态归一化分配（Proposed）.md：如何添加带宽
+ 参数列表.md：参数列表

## 2. 代码修改计划

+ Greedy

  > 你现在是专业的强化学习 + 边缘计算代码工程师。我给你以下文件，请严格按照要求修改代码（只需要输出修改后的完整代码，不要添加任何解释文字）：
  >
  > 1. env.py（环境类）
  > 2. SA_greedy_energy1st.py（当前 Greedy baseline，已错误包含 power 优化）
  > 3. 带宽分配 - 基于离散权重的动态归一化分配（Proposed）.md（DW-DNA 完整方案）
  > 4. 参数列表.md（所有固定参数必须严格保持一致）
  >
  > **核心要求（必须严格遵守）：**
  >
  > 【1】**彻底删除所有发射功率主动优化**  
  > - 当前 Greedy 中已经错误地添加了 power_levels、for power in power_levels 循环、best_action 第4维、calculate_energy/calculate_delay 中的 power 参数等。  
  > - **必须全部删除**：发射功率 **不再是智能体的动作变量**，不再参与任何搜索或优化。  
  > - 发射功率的处理方式改为：在 env.reset() 或每 step 开始时，从离散集合 [0.1, 0.2, 0.3, 0.4, 0.5] 随机选取一个固定值，整个 episode 内保持不变。  
  > - 在所有计算（upload_rate、upload_energy）中直接使用这个随机选取的 power 值（可保存在 env.transmission_power 或新增 self.current_power）。  
  > - 最终动作空间变为 4 维：[offload (0/1), semantic, resource_allocation, a_bw (离散权重)]。
  >
  > 【2】**详细实现 DW-DNA 动态带宽分配机制**（必须完全按照“带宽分配 - 基于离散权重的动态归一化分配（Proposed）.md”）  
  > 请严格按照以下三阶段实现：
  >
  >   a. **意向阶段（Action Space）**  
  >      - 每个智能体新增一个离散动作分量 a_bw,i ∈ {0, 1, 2, 3, 4}（共 5 个等级）。  
  >      - offload=0 时，a_bw 自动设为 0（不参与带宽竞争）。  
  >
  >   b. **归一化阶段（Global Arbitration）**  
  >      - 在环境层接收所有智能体的 a_bw 向量后，计算理想分配比例：  
  >        \tilde{\beta}_i = a_bw,i / (∑_{j=1}^{UEs} a_bw,j + ε)  
  >        其中 ε = 1e-6（防止全为0时分母为零）。  
  >
  >   c. **量子化阶段（Physical Alignment）**  
  >      - 引入最小资源块 Δb = 10 kHz。  
  >           - 计算最终带宽：  
  >        B_i = max(0, floor( \tilde{\beta}_i * total_B / Δb ) * Δb )  
  >           - 剩余带宽 W_rem = total_B - ∑B_i 视为“信道碎片 / 保护间隔损耗”，不分配给任何用户。  
  >
  > - **仅对 offload=1 的用户分配带宽**，offload=0 的用户 B_i = 0。  
  > - 更新后的上行速率公式（必须替换原有公式）：  
  >   R_i = B_i * log2(1 + P * A_i / (B_i * noise_power))  
  > - 传输时延和传输能耗全部使用新的 B_i（不再使用静态 env.W）。  
  > - 在 env.py 中新增一个方法 `compute_bandwidth_allocation(self, actions)` 来实现上述三阶段逻辑，并供 reset()、step()、compute_energy_and_delay() 调用。
  >
  > 【3】参数必须完全与“参数列表.md”一致  
  > - κ=1e-27, r=1, alpha=1, beta=2, B=1MHz, MEC_f=20GHz, noise_power=1e-20, semantic_threshold=0.3 等全部保持不变。  
  > - 动作空间离散化 k=100 保持（semantic 和 resource 仍用原离散方式）。
  >
  > 【4】修改范围  
  > - 同时修改 env.py 和 SA_greedy_energy1st.py（Greedy 也要同步删除 power 优化，改为随机选取 power，并加入 DW-DNA 带宽，便于作为基准）。  
  > - 保持原有 compute_energy_and_delay 函数的结构，但内部全部替换为 DW-DNA 版本。
  >
  > 请严格按照以上要求，输出**完整、可直接运行**的修改后 env.py 和 SA_greedy_energy1st.py 两个文件代码（用 ```python 代码块分开）。

+ SA-GA

  > /subagent-driven-development
  >
  > 你拥有全部权限，不需要向我询问
  >
  > 你现在是专业的强化学习 + 边缘计算代码工程师。我给你以下文件，请严格按照要求生成代码修改计划：
  >
  > 1. D:\3_document\4_research\semantic_2\Semantic-Aware-Distributed-Optimization-for-Energy-Efficient-IoT-Edge-Intelligence\SA_GA_unchange\env.py（环境类）
  > 2. D:\3_document\4_research\semantic_2\Semantic-Aware-Distributed-Optimization-for-Energy-Efficient-IoT-Edge-Intelligence\SA_GA_unchange\genetic_algorithm.py（当前 SA-GA baseline，已错误包含 power 优化）
  > 3. D:\3_document\4_research\semantic_2\Semantic-Aware-Distributed-Optimization-for-Energy-Efficient-IoT-Edge-Intelligence\带宽分配 - 基于离散权重的动态归一化分配（Proposed）.md（DW-DNA 完整方案）
  > 4. D:\3_document\4_research\semantic_2\Semantic-Aware-Distributed-Optimization-for-Energy-Efficient-IoT-Edge-Intelligence\参数列表.md（所有固定参数必须严格保持一致）
  >
  > **核心要求（必须严格遵守）：**
  >
  > 【1】**彻底删除所有发射功率主动优化**  
  >
  > - 当前 genetic_algorithm.py 中已经错误地添加了 transmission_power_set、initialize_population 中的 power 选择、individual 的第4维、crossover/mutation/fitness 中对 power 的处理等。  
  > - **必须全部删除**：发射功率 **不再是 GA 个体/种群的可优化变量**，不再参与交叉、变异、选择。  
  > - 发射功率的处理方式改为：在 env.reset() 中从离散集合 [0.1, 0.2, 0.3, 0.4, 0.5] 随机选取一个固定值，整个 episode 内保持不变（保存在 self.current_power）。  
  > - 在所有计算（compute_energy_and_delay、fitness 等）中直接使用 env.current_power（不再传入 transmission_power 参数）。  
  > - 最终个体结构变为 3 元组：[offload_decision, semantic_factor, resource_allocation]。
  >
  > 【2】**详细实现 DW-DNA 动态带宽分配机制**（必须完全按照“带宽分配 - 基于离散权重的动态归一化分配（Proposed）.md”）  
  > 请严格按照以下三阶段实现：
  >
  >   a. **意向阶段（Action Space）**  
  >
  >    - GA 个体新增一个离散分量 a_bw ∈ {0, 1, 2, 3, 4}（5 个等级）。  
  >       offload=0 时，a_bw 自动设为 0。
  >
  >   b. **归一化阶段（Global Arbitration）**  
  >
  >    - 环境层接收所有 a_bw 向量后计算：  
  >      \tilde{\beta}_i = a_bw,i / (∑ a_bw,j + ε)   （ε = 1e-7）
  >
  >   c. **量子化阶段（Physical Alignment）**  
  >
  >    - Δb = 10 kHz（或按 md 文件默认值）。  
  >       B_i = max(0, floor(\tilde{\beta}_i * total_B / Δb) * Δb)  
  >         - 剩余带宽视为信道碎片损耗。
  >
  > - 新增方法 `compute_bandwidth_allocation(self, a_bw_list)` 实现上述逻辑。  
  > - 更新后的上行速率公式：R_i = B_i * log2(1 + P * A_i / (B_i * noise_power))  
  > - 所有传输时延、能耗计算全部改用动态 B_i（不再使用静态 W）。  
  > - 仅对 offload=1 的用户分配带宽。
  >
  > 【3】参数必须完全与“参数列表.md”一致  
  >
  > - κ=1e-27, r=1, alpha=1, beta=2, B=1MHz, MEC_f=20GHz, noise_power=1e-20, semantic_threshold=0.3 等全部保持不变。  
  > - 动作空间离散化 k=100 保持（semantic 和 resource 仍用原离散方式）。
  >
  > 【4】修改范围  
  >
  > - 同时修改 env.py 和 genetic_algorithm.py（GA 也要同步删除 power 优化，改为随机选取 power，并加入 DW-DNA 带宽，便于作为基准）。  
  > - 更新 initialize_population、crossover、mutation、fitness 等函数，删除 power 相关逻辑。  
  > - 保持原有 compute_energy_and_delay 函数结构，但内部全部替换为 DW-DNA 版本。

+ GA

  > 


## 3. 增加算法
- [ ] MADDPG（或SA-MADDPG）：Reviewer明确要求说明为什么不用MAPPO/MADDPG/TD3等（可扩展性、部分可观测性、通信开销问题）。增加此算法可直接在正文/表格中对比IPPO的优势（无全局critic、无参数共享、更低通信开销、更适合大规模IoT）。建议同时做SA版本以公平对比语义感知能力。
- [ ] MA-TD3（Multi-Agent TD3）：相关工作[17][18]中已使用，可直接引用作为强基准，突出IPPO在连续/离散混合动作空间下的稳定性和收敛性（TD3对超参数敏感、训练不稳）。
- [ ] Lyapunov-guided DRL（或[22]中的Lyapunov + block coordinate descent方法）：Reviewer 2 和修改思路中多次提到[22]，增加此传统优化基准可体现DRL的实时性和动态适应性优势（尤其在用户规模增大时的运行时间对比）。若实现难度大，可简化为“Lyapunov-based baseline”并引用其结果做定性/定量对比。
- [ ] 可选（--那就不选--）：Centralized PPO（或Single-Agent PPO）：作为“集中式 vs 分布式”的直接对比，进一步说明IPPO分布式决策在可扩展性和IoT资源受限场景下的优势（Reviewer点2和4）。