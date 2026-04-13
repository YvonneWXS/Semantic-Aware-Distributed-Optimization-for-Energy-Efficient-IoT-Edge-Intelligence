import numpy as np
import math

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        np.random.seed(47)
        self.agent_num = 5 # 设置智能体的个数
        self.obs_dim = 3  # 设置智能体的观测维度 ：task_size, computing_density, max_delay
        self.action_dim = 4 # 设置智能体的动作维度：offload_decision, semantic_factor, resource_allocation, bandwidth_weight

        # 基本参数
        # 频率
        self.Hz = 1
        self.kHz = 1000 * self.Hz
        self.mHz = 1000 * self.kHz
        self.GHz = 1000 * self.mHz
        self.nor = 10**(-7)
        self.nor1 = 10**19

        # 数据大小
        self.bit = 1
        self.B = 8 * self.bit
        self.KB = 1024 * self.B
        self.MB = 1024 * self.KB

        #模拟参数
        self.κ = 10 ** (-27)  # 芯片结构对cpu处理的影响因子
        # self.κ_mec = 10 ** (-26) #mec服务器芯片结构对cpu处理的影响因子
        self.r = 1  #运行语义提取任务的CPU周期数的参数1...."若设为2会导致语义提取的能耗显著增大，不合适。。。"
        self.alpha = 1  #运行语义提取任务的CPU周期数的参数2
        self.beta =2 #运行语义提取任务的CPU周期数的参数3
        self.transmission_bandwidth = 1 * self.mHz   # 传输带宽1MHz
        self.noise_power = 10**(-20) # 噪声功率-170dBm
        #不考虑邻道干扰功率
        self.MEC_f = 20 * self.GHz  # MEC的计算能力
        # 固定传输功率 (W)
        self.transmission_power = 0.1  # 固定传输功率0.1W
        # 资源块带宽 (RB bandwidth) for quantization
        self.RB_bandwidth = 180 * self.kHz  # 5G NR RB bandwidth (180 kHz)

        #随机生成每个UE的计算能力等参数
        self.local_comp = np.zeros(self.agent_num)  # 本地计算能力
        self.distance = np.zeros(self.agent_num)  # 距离
        self.channel_gain = np.zeros(self.agent_num)  # 信道增益

        for i in range(self.agent_num):
            self.local_comp[i] = np.random.randint(1.5 * self.GHz , 2 * self.GHz)    # UE的本地计算能力
            self.distance[i] = np.random.uniform(10, 100)  # 随机生成用户设备和基站之间的距离
            self.channel_gain[i] = 1e-3 * (1.0 / self.distance[i]) **2.5  #计算信道增益

        #定义三个变量，分别表示每个智能体的任务大小，处理任务每比特数据的成本，本地处理任务时间，任务最大容忍时间
        self.task_size = np.zeros(self.agent_num)
        self.computing_density = np.zeros(self.agent_num)
        self.local_delay = np.zeros(self.agent_num)
        self.max_delay = np.zeros(self.agent_num)

        self.cur_agent_obs = []
        self.sub_agent_obs = []

    def _allocate_bandwidth_dwdna(self, bandwidth_weights, offload_decisions):
        """
        DW-DNA bandwidth allocation: B_i = floor(w_i / sum(w_j) * total_bandwidth / Δb) * Δb
        where Δb = 180 kHz (simulating 5G RB)
        """
        total_bandwidth = self.transmission_bandwidth  # Hz
        rb_size = 180 * self.kHz  # 180 kHz per RB

        # Calculate normalized weights (only for offloading UEs)
        offloading_weights = [bw_weight * offload for bw_weight, offload in zip(bandwidth_weights, offload_decisions)]
        total_weight = sum(offloading_weights)

        if total_weight == 0:
            return [0] * self.agent_num

        # Allocate bandwidth with floor quantization
        bandwidths = []
        for weight, offload in zip(bandwidth_weights, offload_decisions):
            if offload == 1 and weight > 0:
                # Normalized allocation
                normalized_share = weight / total_weight
                ideal_bandwidth = normalized_share * total_bandwidth

                # Floor quantization to RB multiples
                num_rbs = max(0, int(ideal_bandwidth // rb_size))
                allocated_bandwidth = num_rbs * rb_size
            else:
                allocated_bandwidth = 0
            bandwidths.append(allocated_bandwidth)

        return bandwidths

    def reset(self): # 重置初始环境
        np.random.seed(1)
        self.sub_agent_obs = []
        for i in range(self.agent_num):
            self.task_size[i] = 256 *8* self.KB  # 任务大小
            self.computing_density[i] = 450  # 处理任务每比特数据的成本
            self.local_delay[i] = self.task_size[i] * self.computing_density[i] / self.local_comp[i]  # 本地处理任务时间
            self.max_delay[i] = np.random.uniform(self.local_delay[i], 2 * self.local_delay[i])  # 任务最大容忍时间随机取
            observation = np.array([self.task_size[i], self.computing_density[i], self.max_delay[i]])
            self.sub_agent_obs.append(observation)
        return self.sub_agent_obs

    def step(self, actions, episode, step):
        """
        动作输入 actions 为 one-hot 拼接向量。
        假设总维度 24 = 2(卸载) + 8(语义) + 10(资源) + 4(带宽权重)

        用离散动作空间one-hot编码值计算reward
        """
        self.cur_agent_obs = self.sub_agent_obs
        cur_agent_reward = []
        sub_agent_done = []
        agent_energy = []
        time_penalty_space = []
        resource_allocation_space = []
        local_energy = []
        bandwidth_weights = []
        offload_num = 0

        # 第一遍循环：解析动作，统计卸载人数，预处理资源分配和带宽权重
        for i in range(self.agent_num):
            #提取每个agent的action
            action = actions[i]

            # 1. 卸载决策 (Indices 0-1)
            offload_decision = np.argmax(action[:2])  # 获取索引并计算对应的值
            offload_num = offload_num + offload_decision

            # 2. 语义因子 (Indices 2-10, 8个值)
            # 范围 [0.3, 1.0]. Index 0 -> 0.3, Index 7 -> 1.0
            semantic_factor = np.argmax(action[2:10]) * 0.1 + 0.3

            # 3. 资源分配 (Indices 10-20, 10个值)
            # 范围 0.1 - 1.0
            resource_allocation= (np.argmax(action[10:20])+1)*0.1*self.MEC_f    # 资源分配    #🌟
            if offload_decision == 1:
                resource_allocation_space.append(resource_allocation)
            else:
                resource_allocation_space.append(0)

            # 4. 带宽权重 (Indices 20-24, 4个值)
            # 范围 0-3
            bw_weight = np.argmax(action[20:24])  # 0,1,2,3
            bandwidth_weights.append(bw_weight)

        # 收集卸载决策列表
        offload_decisions = []
        for i in range(self.agent_num):
            action = actions[i]
            offload_decision = np.argmax(action[:2])
            offload_decisions.append(offload_decision)

        # DW-DNA 带宽分配
        bandwidth_allocation = self._allocate_bandwidth_dwdna(bandwidth_weights, offload_decisions)

        # 如果资源分配总和超过 MEC 资源限制，则归一化
        total_allocated = sum(resource_allocation_space)
        if total_allocated > self.MEC_f and total_allocated > 0:
                resource_allocation_space = [ra * self.MEC_f / total_allocated for ra in resource_allocation_space]

        # 第二遍循环：计算物理指标 (时延、能耗、Reward)
        for i in range(self.agent_num):
            #提取每个agent的action
            action = actions[i]

            # 解析动作
            # 1. 卸载决策
            offload_decision = np.argmax(action[:2])

            # 2. 语义因子
            semantic_factor = np.argmax(action[2:10]) * 0.1 + 0.3

            # 3. 资源分配
            resource_allocation= resource_allocation_space[i]

            # 4. 带宽权重 (已解析)
            # bw_weight already parsed

            # 分配带宽
            W_i = bandwidth_allocation[i]

            # 计算本地能耗
            RL_local_energy = self.κ * self.task_size[i] * self.computing_density[i] * (self.local_comp[i]**2)
            local_energy.append(RL_local_energy)

            if offload_decision == 0:#本地处理
                RL_total_energy = RL_local_energy
                RL_total_delay = self.local_delay[i]
            else:
                # 卸载执行
                # 1. 语义提取能耗 (Local)
                # 论文修正公式: E_sem = kappa * alpha * (D^eta) * (beta^(-k) - 1) * f^2
                # 代码对应: eta -> r, k -> beta
                RL_SEtask_energy = self.κ * self.alpha * ( self.task_size[i] ** self.r) * ((semantic_factor ** (-self.beta) - 1)) * (self.local_comp[i]**2)

                # 2. 传输速率 (Eq. 4) 使用动态分配的带宽 W_i
                if W_i > 0:
                    uplink_rate = W_i * math.log2 (1 + self.transmission_power * self.channel_gain[i] / (W_i * self.noise_power))
                else:
                    uplink_rate = 0.0  # 带宽为0，无法传输（但卸载决策为1时不应出现）
                    # 理论上，如果带宽为0但卸载决策为1，应视为无效，但这里保持鲁棒性

                # 3. 传输能耗 (Eq. 5)
                # 数据量 = task_size * semantic_factor
                if uplink_rate > 0:
                    upload_energy = self.transmission_power *  self.task_size[i] * semantic_factor / uplink_rate
                else:
                    upload_energy = float('inf')  # 无穷大能耗，惩罚

                # 总能耗
                RL_total_energy = RL_SEtask_energy + upload_energy # + mec_energy

                # 总时延 = 语义提取时延 + 传输时延 + MEC处理时延
                se_delay = self.alpha * (self.task_size[i] ** self.r) * ((semantic_factor ** (-1) - 1)) / self.local_comp[i]
                if uplink_rate > 0:
                    tx_delay = (semantic_factor * self.task_size[i]) / uplink_rate
                else:
                    tx_delay = float('inf')
                mec_delay = (semantic_factor * self.task_size[i] * self.computing_density[i]) / resource_allocation

                RL_total_delay = se_delay + tx_delay + mec_delay

            agent_energy.append(RL_total_energy)

            #计算时间约束惩罚
            time_penalty = -max(0,  RL_total_delay-self.max_delay[i])/ self.max_delay[i]
            time_penalty_space.append(time_penalty)


            #其余信息
            sub_agent_done.append(False)#智能体不提前终止
            # sub_agent_info.append({})#附加信息存储

        #-------------------------更新智能体状态-------------------------
        np.random.seed(2+step)
        self.sub_agent_obs = []
        for i in range(self.agent_num):
            # 随机生成每个智能体的观测值
            self.task_size[i] = 256 *8* self.KB  # 任务大小
            self.computing_density[i] = 450  # 处理任务每比特数据的成本
            self.local_delay[i] = self.task_size[i] * self.computing_density[i] / self.local_comp[i]  # 本地处理任务时间
            self.max_delay[i] = np.random.uniform(self.local_delay[i], 2 * self.local_delay[i])  # 任务最大容忍时间随机取
            observation = np.array([self.task_size[i], self.computing_density[i], self.max_delay[i]])
            self.sub_agent_obs.append(observation)

        # -------------------------计算奖励-------------------------
        #min-max归一化能耗(可能会影响收敛)
        E_total = sum(agent_energy)

        #计算资源分配惩罚
        resource_allocation_penalty = -max(0,sum(resource_allocation_space)-self.MEC_f)/self.MEC_f

        #时间约束惩罚
        total_time_penalty = sum(time_penalty_space)

        #奖励权重
        w1,w2 = 1,1
        w3 = 0
        w = w1+w2

        for i in range(self.agent_num):
            # 奖励函数: 能耗效率 + 时延惩罚
            cur_agent_reward.append(w1/w* self.agent_num /E_total + (w2/w)*total_time_penalty)#(w3/w)*resource_allocation_penalty)

        cur_agent_info=[E_total,total_time_penalty,resource_allocation_penalty]
        #绘制rewards曲线
        #所有agent的平均奖励
        # writer.add_scalar('reward/average_reward', np.mean(sub_agent_reward), self.episode_count)
        #每个agent的奖励
        return [self.cur_agent_obs, cur_agent_reward, sub_agent_done, cur_agent_info]


# 测试 DW-DNA 带宽分配逻辑
if __name__ == "__main__":
    print("Testing DW-DNA bandwidth allocation...")

    # 创建环境实例
    env = EnvCore()
    env.transmission_bandwidth = 1e6  # 1 MHz

    # Test 1: Single offloading UE
    print("\nTest 1: Single offloading UE")
    weights = [3, 0, 0, 0, 0]
    offloads = [1, 0, 0, 0, 0]
    bandwidths = env._allocate_bandwidth_dwdna(weights, offloads)
    print(f"  Weights: {weights}")
    print(f"  Offloads: {offloads}")
    print(f"  Bandwidths: {bandwidths}")
    print(f"  Total allocated: {sum(bandwidths)} Hz")

    # Test 2: Equal weights for 3 offloading UEs
    print("\nTest 2: Equal weights for 3 offloading UEs")
    weights = [2, 2, 2, 0, 0]
    offloads = [1, 1, 1, 0, 0]
    bandwidths = env._allocate_bandwidth_dwdna(weights, offloads)
    print(f"  Weights: {weights}")
    print(f"  Offloads: {offloads}")
    print(f"  Bandwidths: {bandwidths}")
    print(f"  Total allocated: {sum(bandwidths)} Hz")

    # Test 3: Different weights
    print("\nTest 3: Different weights")
    weights = [3, 2, 1, 0, 0]
    offloads = [1, 1, 1, 0, 0]
    bandwidths = env._allocate_bandwidth_dwdna(weights, offloads)
    print(f"  Weights: {weights}")
    print(f"  Offloads: {offloads}")
    print(f"  Bandwidths: {bandwidths}")
    print(f"  Total allocated: {sum(bandwidths)} Hz")

    # Test 4: No offloading UEs
    print("\nTest 4: No offloading UEs")
    weights = [3, 2, 1, 0, 0]
    offloads = [0, 0, 0, 0, 0]
    bandwidths = env._allocate_bandwidth_dwdna(weights, offloads)
    print(f"  Weights: {weights}")
    print(f"  Offloads: {offloads}")
    print(f"  Bandwidths: {bandwidths}")
    print(f"  Total allocated: {sum(bandwidths)} Hz")

    # Test 5: Zero weights for offloading UEs
    print("\nTest 5: Zero weights for offloading UEs")
    weights = [0, 0, 0, 0, 0]
    offloads = [1, 1, 1, 0, 0]
    bandwidths = env._allocate_bandwidth_dwdna(weights, offloads)
    print(f"  Weights: {weights}")
    print(f"  Offloads: {offloads}")
    print(f"  Bandwidths: {bandwidths}")
    print(f"  Total allocated: {sum(bandwidths)} Hz")

    print("\nDW-DNA tests completed!")