import numpy as np
import math

class EnvCore(object):
    """
    # 环境中的智能体
    # 动作空间结构（one-hot编码，206维）：
    # - 维度0-1: 卸载决策 (0或1)
    # - 维度2-101: 资源分配 (100个离散值，0.1-1.0)
    # - 维度102-105: 带宽权重 (4个离散值，0,1,2,3)
    # - 维度106-205: 语义因子 (100个离散值，0.1-1.0)
    #
    # DW-DNA带宽分配算法：
    # 1. 基于带宽权重进行归一化分配
    # 2. 向下取整量化模拟RB资源块（180 kHz）
    # 3. 边界情况处理：所有带宽权重为0时平均分配
    """

    def __init__(self):
        np.random.seed(47)
        self.agent_num = 5 # 设置智能体的个数
        self.obs_dim = 3  # 设置智能体的观测维度 ：task_size, computing_density, max_delay
        self.action_dim = 206 # 设置智能体的动作维度：2+100+4+100=206维one-hot编码

        # 动作空间维度常量
        self.OFFLOAD_DECISION_DIM = 2  # 卸载决策维度：0或1
        self.RESOURCE_ALLOCATION_DIM = 100  # 资源分配离散值数量
        self.BW_WEIGHT_DIM = 4  # 带宽权重离散值数量：0,1,2,3
        self.SEMANTIC_FACTOR_DIM = 100  # 语义因子离散值数量

        # 范围限制常量
        self.MIN_FACTOR_VALUE = 0.1  # 最小因子值
        self.MAX_FACTOR_VALUE = 1.0  # 最大因子值
        self.FACTOR_STEP = 0.01  # 因子步长

        self.k = 100

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
        # self.W = self.transmission_bandwidth/self.agent_num  #每个用户设备的带宽分配
        # transmission_power = 0.3  # 固定传输功率0.3W (已移除动作空间中的传输功率维度)
        self.noise_power = 10**(-20) # 噪声功率-170dBm
        self.rb_size = 180 * self.kHz  # RB资源块大小：180 kHz
        #不考虑邻道干扰功率
        self.MEC_f = 20 * self.GHz  # MEC的计算能力
        # self.weight_factor = 0.7  # 能耗和公平性的权重

        #随机生成每个UE的计算能力等参数
        self.UE_params = []  # 初始化UE参数列表
        self.local_comp = np.zeros(self.agent_num)  # 本地计算能力
        self.distance = np.zeros(self.agent_num)  # 距离
        self.channel_gain = np.zeros(self.agent_num)  # 信道增益
        for i in range(self.agent_num):
            self.local_comp[i] = np.random.randint(1.5 * self.GHz , 2 * self.GHz)    # UE的本地计算能力
            self.distance[i] = np.random.uniform(10, 100)  # 随机生成用户设备和基站之间的距离
            self.channel_gain[i] = 1e-3 * (1.0 / self.distance[i]) **2.5  #计算信道增益
            # #将每个UE的参数存储在字典中
            # ue_params = {
            #     'local_comp': local_comp,
            #     'channel_gain': channel_gain,
            # }
            # self.UE_params.append(ue_params)·

        #定义三个变量，分别表示每个智能体的任务大小，处理任务每比特数据的成本，本地处理任务时间，任务最大容忍时间
        self.task_size = np.zeros(self.agent_num)
        self.computing_density = np.zeros(self.agent_num)
        self.local_delay = np.zeros(self.agent_num)
        self.max_delay = np.zeros(self.agent_num)


    def reset(self): # 重置初始环境
        np.random.seed(47)
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
        e.g. actions = [array([...206维...]), ...]
        动作空间结构（one-hot编码）：
        - 维度0-1: 卸载决策 (0或1)
        - 维度2-101: 资源分配 (100个离散值，0.1-1.0)
        - 维度102-105: 带宽权重 (4个离散值，0,1,2,3)
        - 维度106-205: 语义因子 (100个离散值，0.1-1.0)
        用离散动作空间one-hot编码值计算reward，支持DW-DNA带宽分配
        """
        # sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        agent_energy = []
        time_penalty_space = []
        resource_allocation_space = []
        local_energy = []
        offload_num = 0
        offload_decisions = []  # 存储所有智能体的卸载决策
        for i in range(self.agent_num):
            action = actions[i]
            offload_decision = np.argmax(action[:self.OFFLOAD_DECISION_DIM])  # 选择卸载决策
            offload_decisions.append(offload_decision)
            offload_num += offload_decision
            # 资源分配解析：从RESOURCE_ALLOCATION_DIM个离散值中选择，范围应为MIN_FACTOR_VALUE-MAX_FACTOR_VALUE
            # 计算资源分配起始索引：OFFLOAD_DECISION_DIM
            resource_start_idx = self.OFFLOAD_DECISION_DIM
            resource_end_idx = resource_start_idx + self.RESOURCE_ALLOCATION_DIM

            # 原始公式：(np.argmax(action[resource_start_idx:resource_end_idx]) + 1) * self.FACTOR_STEP 产生范围0.01-1.00
            # 通过max(MIN_FACTOR_VALUE, min(MAX_FACTOR_VALUE, ...))限制在MIN_FACTOR_VALUE-MAX_FACTOR_VALUE范围内
            resource_factor_raw = (np.argmax(action[resource_start_idx:resource_end_idx]) + 1) * self.FACTOR_STEP  # 0.01-1.0
            resource_factor = max(self.MIN_FACTOR_VALUE, min(self.MAX_FACTOR_VALUE, resource_factor_raw))  # 限制在0.1-1.0范围内
            resource_allocation = resource_factor * self.MEC_f  # 计算资源分配
            if offload_decision == 1:  # 仅当卸载任务时分配资源
                resource_allocation_space.append(resource_allocation)
            else:
                resource_allocation_space.append(0)

        # 收集所有智能体的带宽权重
        bw_weights = []
        for i in range(self.agent_num):
            action = actions[i]
            offload_decision = np.argmax(action[:self.OFFLOAD_DECISION_DIM])
            if offload_decision == 1:
                # 计算带宽权重起始索引
                bw_start_idx = self.OFFLOAD_DECISION_DIM + self.RESOURCE_ALLOCATION_DIM
                bw_end_idx = bw_start_idx + self.BW_WEIGHT_DIM
                bw_weight = np.argmax(action[bw_start_idx:bw_end_idx])  # 带宽权重
                bw_weights.append(bw_weight)
            else:
                bw_weights.append(0)  # 不卸载的设备带宽权重为0

        # DW-DNA带宽分配：基于带宽权重进行归一化分配，并向下取整量化（模拟RB资源块）
        # 算法：B_i = max(0, floor(β_i * W / Δb)) * Δb
        # 其中β_i是归一化权重：β_i = bw_weight_i / sum_weights
        # Δb = 180 kHz (一个RB资源块的大小)
        if offload_num > 0:
            total_bw_weight = sum(bw_weights)
            if total_bw_weight > 0:
                # 归一化分配带宽：β_i = bw_weight_i / sum_weights
                normalized_weights = [bw_weight / total_bw_weight for bw_weight in bw_weights]
            else:
                # 边界情况：如果所有带宽权重为0但offload_num>0，平均分配带宽给卸载设备
                normalized_weights = [1/offload_num if bw_weights[j] == 0 and offload_decisions[j] == 1 else 0
                                     for j in range(self.agent_num)]

            # 计算理想带宽分配：β_i * W
            ideal_bandwidths = [self.transmission_bandwidth * weight for weight in normalized_weights]

            # 向下取整量化（模拟RB资源块）：max(0, floor(bw / Δb)) * Δb
            W_allocations = [max(0, math.floor(bw / self.rb_size)) * self.rb_size for bw in ideal_bandwidths]
        else:
            W_allocations = [0] * self.agent_num

        # 如果资源分配总和超过 MEC 资源限制，则归一化
        total_allocated = sum(resource_allocation_space)
        if total_allocated > self.MEC_f and total_allocated > 0:
            resource_allocation_space = [ra * self.MEC_f / total_allocated for ra in resource_allocation_space]
        
        # print((sum(resource_allocation_space)-self.MEC_f)/self.MEC_f)
        # print("test")
        for i in range(self.agent_num):
            #提取每个agent的action
            action = actions[i]
            offload_decision = np.argmax(action[:self.OFFLOAD_DECISION_DIM])  # 选择卸载决策
            # 带宽权重解析：从BW_WEIGHT_DIM个离散值中选择
            bw_start_idx = self.OFFLOAD_DECISION_DIM + self.RESOURCE_ALLOCATION_DIM
            bw_end_idx = bw_start_idx + self.BW_WEIGHT_DIM
            bw_weight = np.argmax(action[bw_start_idx:bw_end_idx])  # 带宽权重: 0,1,2,3 (4个值)
            # 移除传输功率，使用固定传输功率
            transmission_power = 0.3  # 固定传输功率0.3W
            resource_allocation = resource_allocation_space[i]    # 资源分配
            # 语义因子解析：从SEMANTIC_FACTOR_DIM个离散值中选择，范围应为MIN_FACTOR_VALUE-MAX_FACTOR_VALUE
            # 计算语义因子起始索引：OFFLOAD_DECISION_DIM + RESOURCE_ALLOCATION_DIM + BW_WEIGHT_DIM
            semantic_start_idx = self.OFFLOAD_DECISION_DIM + self.RESOURCE_ALLOCATION_DIM + self.BW_WEIGHT_DIM
            semantic_end_idx = semantic_start_idx + self.SEMANTIC_FACTOR_DIM

            # 原始公式：(np.argmax(action[semantic_start_idx:semantic_end_idx]) + 1) * self.FACTOR_STEP 产生范围0.01-1.00
            # 通过max(MIN_FACTOR_VALUE, min(MAX_FACTOR_VALUE, ...))限制在MIN_FACTOR_VALUE-MAX_FACTOR_VALUE范围内
            semantic_factor_raw = (np.argmax(action[semantic_start_idx:semantic_end_idx]) + 1) * self.FACTOR_STEP  # 0.01-1.0
            semantic_factor = max(self.MIN_FACTOR_VALUE, min(self.MAX_FACTOR_VALUE, semantic_factor_raw))  # 限制在0.1-1.0范围内

            RL_local_energy = self.κ * self.task_size[i] * self.computing_density[i] * (self.local_comp[i]**2)
            local_energy.append(RL_local_energy)

            if offload_decision == 0:#本地处理
                RL_total_energy = RL_local_energy
                RL_total_delay = self.local_delay[i]
            else:
                # 使用DW-DNA分配的带宽
                W_i = W_allocations[i] if offload_decision == 1 else 0
                if W_i > 0:
                    uplink_rate = W_i * math.log2(1 + transmission_power * self.channel_gain[i] / (W_i * self.noise_power))
                else:
                    # 如果没有分配到带宽，使用最小带宽计算
                    uplink_rate = 0.001 * math.log2(1 + transmission_power * self.channel_gain[i] / (0.001 * self.noise_power))
                RL_SEtask_energy = self.κ * self.alpha * (self.task_size[i] ** self.r) * ((semantic_factor ** (-self.beta) - 1)) * (self.local_comp[i] ** 2)
                upload_energy = transmission_power * self.task_size[i] * semantic_factor / uplink_rate
                # mec_energy = self.κ_mec *(resource_allocation**3) * semantic_factor *  self.task_size[i] * self.computing_density[i]/(resource_allocation)
                RL_total_energy = RL_SEtask_energy + upload_energy # + mec_energy
                RL_total_delay = self.alpha * (self.task_size[i] ** self.r) * ((semantic_factor ** (-1) - 1)) / self.local_comp[i] + semantic_factor * self.task_size[i] / uplink_rate + semantic_factor * self.task_size[i] * self.computing_density[i] / resource_allocation
            # print("local_delay:",self.local_delay[i])
            # print(self.alpha * ( self.task_size[i] ** self.r)*((semantic_factor **(-1)-1)) / self.local_comp[i])
            # print(semantic_factor *  self.task_size[i]/ uplink_rate)
            # print(semantic_factor *  self.task_size[i] * self.computing_density[i]/(self.MEC_f* resource_allocation))
            # print("RL_total_delay",RL_total_delay)
            # print("test")

            agent_energy.append(RL_total_energy)

            #计算时间约束惩罚
            time_penalty = -max(0,  RL_total_delay-self.max_delay[i])/ self.max_delay[i]
            time_penalty_space.append(time_penalty)

            
            #其余信息
            sub_agent_done.append(False)#智能体不提前终止
            # sub_agent_info.append({})#附加信息存储

            #更新智能体状态
            # 随机生成每个智能体的观测值
            # self.task_size[i] = 256 *8 * self.KB  # 任务大小
            # self.computing_density[i] = 450  # 处理任务每比特数据的成本
            # self.local_delay[i] = self.task_size[i] * self.computing_density[i] / self.local_comp[i]  # 本地处理任务时间
            # self.max_delay[i] = np.random.uniform(self.local_delay[i], 2 * self.local_delay[i])  # 任务最大容忍时间随机取
            # observation = np.array([self.task_size[i], self.computing_density[i], self.max_delay[i]])
            # sub_agent_obs.append(observation)

        #min-max归一化能耗(可能会影响收敛)
        E_total = sum(agent_energy)

        #计算资源分配惩罚
        resource_allocation_penalty = -max(0,sum(resource_allocation_space)-self.MEC_f)/self.MEC_f
        # print(resource_allocation_penalty)
        # print(resource_allocation_penalty)

        #时间约束惩罚
        total_time_penalty = sum(time_penalty_space)

        #奖励权重
        w1,w2 = 1,1
        w3 = 1
        w = w1 + w2 + w3
    
        for i in range(self.agent_num):
            # fairness_reward = jain_fairness
            # sub_agent_reward.append(fairness_reward/E_total_norm) run11
            # sub_agent_reward.append(fairness_reward/E_total_norm+time_penalty_space[i]+resource_allocation_penalty)
            sub_agent_reward.append(w1/w*self.agent_num/E_total + (w2/w)*total_time_penalty + (w3/w)*resource_allocation_penalty)
        sub_agent_info=[E_total,total_time_penalty,resource_allocation_penalty]
        #绘制rewards曲线
        #所有agent的平均奖励
        # writer.add_scalar('reward/average_reward', np.mean(sub_agent_reward), self.episode_count)
        #每个agent的奖励
        return [self.sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
