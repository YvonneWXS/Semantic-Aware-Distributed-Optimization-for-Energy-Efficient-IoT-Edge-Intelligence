import numpy as np
import math

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self, agent_num=5):
        np.random.seed(47)
        self.agent_num = agent_num # TODO 设置智能体的个数
        self.obs_dim = 3  # 设置智能体的观测维度 ：task_size, computing_density, max_delay
        self.action_dim = 4  # 设置智能体的动作维度：offload_decision, semantic_factor, resource_allocation
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
        self.transmission_power = 0.2  # 传输功率固定为0.2W (不再从动作空间学习)
        self.noise_power = 10**(-20) # 噪声功率-170dBm
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
            self.task_size[i] = np.random.randint(1.5 * self.MB, 2 * self.MB)  # 任务大小
            self.computing_density[i] = np.random.uniform(300, 500)  # 处理任务每比特数据的成本
            self.local_delay[i] = self.task_size[i] * self.computing_density[i] / self.local_comp[i]  # 本地处理任务时间
            self.max_delay[i] = np.random.uniform(self.local_delay[i], 2 * self.local_delay[i])  # 任务最大容忍时间随机取
            observation = np.array([self.task_size[i], self.computing_density[i], self.max_delay[i]])
            self.sub_agent_obs.append(observation)
        self.cur_agent_obs = self.sub_agent_obs
        return self.sub_agent_obs

    def step(self, actions, episode, step):
        """
        e.g. actions = [array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
       0., 0., 0., 0., 0.]), array([0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 1., 0.]), array([0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 1., 0., 0., 0.])]
        用离散动作空间one-hot编码值计算reward
        """
        self.cur_agent_obs = self.sub_agent_obs
        cur_agent_reward = []
        sub_agent_done = []
        agent_energy = []
        time_penalty_space = []
        resource_allocation_space = []
        bandwidth_weights = []  # 存储带宽权重 (0-3)
        local_energy = []
        offload_num = 0
        for i in range(self.agent_num):
            #提取每个agent的action
            action = actions[i]
            offload_decision = np.argmax(action[:2])  # 获取索引并计算对应的值
            offload_num = offload_num + offload_decision

            # Bandwidth weight extraction (positions 20-24 for values 0-3)
            bw_weight_idx = np.argmax(action[20:24])  # 0, 1, 2, or 3
            bw_weight = bw_weight_idx  # 0-3

            resource_allocation= (np.argmax(action[10:20])+1)*0.1*self.MEC_f    # 资源分配    #🌟
            if offload_decision == 1:
                resource_allocation_space.append(resource_allocation)
                bandwidth_weights.append(bw_weight)
            else:
                resource_allocation_space.append(0)
                bandwidth_weights.append(0)  # 本地执行带宽权重为0

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


        for i in range(self.agent_num):
            #提取每个agent的action
            action = actions[i]
            offload_decision = np.argmax(action[:2])
            semantic_factor = (np.argmax(action[2:10])+1)*0.1 + 0.3 # 语义因子   #🌟
            # 传输功率现在是常量 self.transmission_power = 0.2W (不再从动作空间学习)
            transmission_power = self.transmission_power
            resource_allocation= resource_allocation_space[i]
            #计算上传带宽
            RL_local_energy = self.κ * self.task_size[i] * self.computing_density[i] * (self.local_comp[i]**2)
            local_energy.append(RL_local_energy)
            # print("upload_delay:",upload_delay)

            if offload_decision == 0:#本地处理
                RL_total_energy = RL_local_energy
                RL_total_delay = self.local_delay[i]
                transmission_power = 0 
            else:
                RL_SEtask_energy = self.κ * self.alpha * ( self.task_size[i] ** self.r)*((semantic_factor **(-self.beta)-1)) * (self.local_comp[i]**2)
                if bandwidth_allocation[i] > 0:
                    # 使用分配的带宽计算香农公式
                    uplink_rate = bandwidth_allocation[i] * math.log2(1 + self.transmission_power * self.channel_gain[i] / (bandwidth_allocation[i] * self.noise_power))
                else:
                    uplink_rate = 0  # 无带宽分配
                if bandwidth_allocation[i] > 0 and uplink_rate > 0:
                    upload_energy = self.transmission_power * self.task_size[i] * semantic_factor / uplink_rate
                else:
                    upload_energy = 0
                # mec_energy = self.κ_mec *(resource_allocation**3) * semantic_factor *  self.task_size[i] * self.computing_density[i]/(resource_allocation)
                RL_total_energy = RL_SEtask_energy + upload_energy # + mec_energy
                if offload_decision == 1 and bandwidth_allocation[i] > 0 and uplink_rate > 0:
                    upload_delay = semantic_factor * self.task_size[i] / uplink_rate
                else:
                    upload_delay = 0

                RL_total_delay = self.alpha * (self.task_size[i] ** self.r) * ((semantic_factor **(-1)-1)) / self.local_comp[i] + upload_delay + semantic_factor * self.task_size[i] * self.computing_density[i] / (resource_allocation if resource_allocation > 0 else 1)
            # print("local_delay:",self.local_delay[i])
            # print(self.alpha * ( self.task_size[i] ** self.r)*((semantic_factor **(-1)-1)) / self.local_comp[i])
            # print(semantic_factor *  self.task_size[i]/ uplink_rate)
            # print(semantic_factor *  self.task_size[i] * self.computing_density[i]/(self.MEC_f* resource_allocation))
            # print("RL_total_delay",RL_total_delay)
            # print("test")

            agent_energy.append(RL_total_energy)

            #计算时间约束惩罚
            time_penalty = -max(0,  RL_total_delay-self.max_delay[i])/ max(self.max_delay[i], 1e-10)
            time_penalty_space.append(time_penalty)

            
            #其余信息
            sub_agent_done.append(False)#智能体不提前终止
            # sub_agent_info.append({})#附加信息存储
        
        #-------------------------更新智能体状态-------------------------
        np.random.seed(2+step)
        self.sub_agent_obs = []
        for i in range(self.agent_num):    
            # 随机生成每个智能体的观测值
            self.task_size[i] = np.random.randint(1.5 * self.MB, 2 * self.MB)  # 任务大小
            self.computing_density[i] = np.random.uniform(300, 500)  # 处理任务每比特数据的成本
            self.local_delay[i] = self.task_size[i] * self.computing_density[i] / self.local_comp[i]  # 本地处理任务时间
            self.max_delay[i] = np.random.uniform(self.local_delay[i], 2 * self.local_delay[i])  # 任务最大容忍时间随机取
            observation = np.array([self.task_size[i], self.computing_density[i], self.max_delay[i]])
            self.sub_agent_obs.append(observation)


        #min-max归一化能耗(可能会影响收敛)
        E_total = sum(agent_energy)

        #计算资源分配惩罚
        resource_allocation_penalty = -max(0,sum(resource_allocation_space)-self.MEC_f)/self.MEC_f

        #时间约束惩罚
        total_time_penalty = sum(time_penalty_space)

        #奖励权重
        w1,w2 = 1,5 
        w3 = 0
        w = 1
    
        for i in range(self.agent_num):
            cur_agent_reward.append(w1/w* self.agent_num / max(E_total, 1e-10) + (w2/w)*total_time_penalty)#(w3/w)*resource_allocation_penalty)

        cur_agent_info=[E_total, total_time_penalty, resource_allocation_penalty]
        #绘制rewards曲线
        #所有agent的平均奖励
        # writer.add_scalar('reward/average_reward', np.mean(sub_agent_reward), self.episode_count)
        #每个agent的奖励
        # 确保观测值不为空
        if len(self.cur_agent_obs) == 0:
            self.cur_agent_obs = self.sub_agent_obs
        return [self.cur_agent_obs, cur_agent_reward, sub_agent_done, cur_agent_info]


# 测试代码（可以放在文件末尾，用 if __name__ == "__main__": 包裹）
if __name__ == "__main__":
    env = EnvCore()
    env.transmission_bandwidth = 1e6  # 1 MHz

    # Test 1: Single offloading UE
    weights = [3, 0, 0, 0, 0]
    offloads = [1, 0, 0, 0, 0]
    bandwidths = env._allocate_bandwidth_dwdna(weights, offloads)
    print(f"Test 1 - Single UE: {bandwidths}")  # 应分配约1MHz

    # Test 2: Equal weights
    weights = [2, 2, 2, 0, 0]
    offloads = [1, 1, 1, 0, 0]
    bandwidths = env._allocate_bandwidth_dwdna(weights, offloads)
    print(f"Test 2 - Equal weights: {bandwidths}")  # 应近似等分

    # Test 3: No offloading UEs
    weights = [0, 0, 0, 0, 0]
    offloads = [0, 0, 0, 0, 0]
    bandwidths = env._allocate_bandwidth_dwdna(weights, offloads)
    print(f"Test 3 - No offloading: {bandwidths}")  # 应全为0

    # Test 4: Different weights
    weights = [3, 2, 1, 0, 0]
    offloads = [1, 1, 1, 0, 0]
    bandwidths = env._allocate_bandwidth_dwdna(weights, offloads)
    print(f"Test 4 - Different weights: {bandwidths}")  # 应按权重比例分配

    # Test 5: Test reward calculation with dynamic bandwidth allocation
    print("\n=== Test 5: Reward Calculation with Dynamic Bandwidth ===")

    # Mock parameters for a single agent
    task_size = 1.5 * env.MB
    computing_density = 400
    local_comp = 1.8 * env.GHz
    channel_gain = 1e-3 * (1/50)**2.5  # 50m distance
    semantic_factor = 0.6
    bandwidth = 500e3  # 500 kHz

    # Calculate uplink rate
    if bandwidth > 0:
        uplink_rate = bandwidth * math.log2(1 + env.transmission_power * channel_gain / (bandwidth * env.noise_power))
    else:
        uplink_rate = 0

    print(f"Task size: {task_size/env.MB:.1f} MB")
    print(f"Computing density: {computing_density:.0f} cycles/bit")
    print(f"Local computation: {local_comp/env.GHz:.1f} GHz")
    print(f"Semantic factor: {semantic_factor:.2f}")
    print(f"Bandwidth: {bandwidth/1e3:.0f} kHz")
    print(f"Uplink rate: {uplink_rate:.2f} bps")

    # Test upload delay calculation
    if bandwidth > 0 and uplink_rate > 0:
        upload_delay = semantic_factor * task_size / uplink_rate
        print(f"Upload delay: {upload_delay:.3f} s")
    else:
        print(f"Upload delay: 0 s (no bandwidth or zero uplink rate)")

    # Test upload energy calculation
    if bandwidth > 0 and uplink_rate > 0:
        upload_energy = env.transmission_power * task_size * semantic_factor / uplink_rate
        print(f"Upload energy: {upload_energy:.6f} J")
    else:
        print(f"Upload energy: 0 J (no bandwidth or zero uplink rate)")

    # Test semantic extraction energy
    RL_SEtask_energy = env.κ * env.alpha * (task_size ** env.r) * ((semantic_factor **(-env.beta)-1)) * (local_comp**2)
    print(f"Semantic extraction energy: {RL_SEtask_energy:.6f} J")

    # Test boundary cases
    print("\n=== Test 6: Boundary Cases ===")

    # Case 1: Zero bandwidth
    bandwidth_zero = 0
    uplink_rate_zero = 0
    print(f"Case 1 - Zero bandwidth:")
    print(f"  Bandwidth: {bandwidth_zero} Hz")
    print(f"  Uplink rate: {uplink_rate_zero} bps")
    print(f"  Upload delay: 0 s")
    print(f"  Upload energy: 0 J")

    # Case 2: Very small bandwidth (edge case)
    bandwidth_small = 180e3  # 1 RB
    if bandwidth_small > 0:
        uplink_rate_small = bandwidth_small * math.log2(1 + env.transmission_power * channel_gain / (bandwidth_small * env.noise_power))
    else:
        uplink_rate_small = 0

    print(f"\nCase 2 - Small bandwidth (1 RB):")
    print(f"  Bandwidth: {bandwidth_small/1e3:.0f} kHz")
    print(f"  Uplink rate: {uplink_rate_small:.2f} bps")
    if bandwidth_small > 0 and uplink_rate_small > 0:
        upload_delay_small = semantic_factor * task_size / uplink_rate_small
        upload_energy_small = env.transmission_power * task_size * semantic_factor / uplink_rate_small
        print(f"  Upload delay: {upload_delay_small:.3f} s")
        print(f"  Upload energy: {upload_energy_small:.6f} J")
    else:
        print(f"  Upload delay: 0 s")
        print(f"  Upload energy: 0 J")
