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
        self.action_dim = 3  # 设置智能体的动作维度：offload_decision, semantic_factor, resource_allocation
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
        self.transmission_power = np.random.uniform(0.1, 0.5)  # 传输功率0.1W-0.5W
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
        e.g. actions = [array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
       0., 0., 0., 0., 0.]), array([0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 1., 0.]), array([0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 1., 0., 0., 0.])]
        用离散动作空间one-hot编码值计算reward
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
        for i in range(self.agent_num):
            action = actions[i]
            offload_decision = np.argmax(action[:2])  # 选择卸载决策
            offload_num += offload_decision
            resource_allocation = (np.argmax(action[12:]) + 1) * 0.01 * self.MEC_f  # 计算资源分配
            if offload_decision == 1:  # 仅当卸载任务时分配资源
                resource_allocation_space.append(resource_allocation)
            else:
                resource_allocation_space.append(0)

        if offload_num > 0:
            W = self.transmission_bandwidth/offload_num  #每个用户设备的带宽分配

        # 如果资源分配总和超过 MEC 资源限制，则归一化
        total_allocated = sum(resource_allocation_space)
        if total_allocated > self.MEC_f and total_allocated > 0:
            resource_allocation_space = [ra * self.MEC_f / total_allocated for ra in resource_allocation_space]
        
        # print((sum(resource_allocation_space)-self.MEC_f)/self.MEC_f)
        # print("test")
        for i in range(self.agent_num):
            #提取每个agent的action
            action = actions[i]
            offload_decision = np.argmax(action[:2])  # 选择卸载决策
            # resource_allocation = (np.argmax(action[12:]) + 1) * 0.01 * self.MEC_f  # 计算资源分配
            resource_allocation= resource_allocation_space[i]    # 资源分配   
            # if offload_decision == 1:  # 仅当卸载任务时分配资源
            #     resource_allocation_space.append(resource_allocation)
            semantic_factor = 1 #(np.argmax(action[2:12])+1)*0.01 + 0.3 # 语义因子

            RL_local_energy = self.κ * self.task_size[i] * self.computing_density[i] * (self.local_comp[i]**2)
            local_energy.append(RL_local_energy)

            if offload_decision == 0:#本地处理
                RL_total_energy = RL_local_energy
                RL_total_delay = self.local_delay[i]
            else:
                uplink_rate = W * math.log2 (1 + self.transmission_power * self.channel_gain[i] / (W * self.noise_power))
                RL_SEtask_energy = self.κ * self.alpha * ( self.task_size[i] ** self.r)*((semantic_factor **(-self.beta)-1)) * (self.local_comp[i]**2)
                upload_energy = self.transmission_power *  self.task_size[i] *semantic_factor / uplink_rate
                # mec_energy = self.κ_mec *(resource_allocation**3) * semantic_factor *  self.task_size[i] * self.computing_density[i]/(resource_allocation)
                RL_total_energy = RL_SEtask_energy + upload_energy # + mec_energy
                RL_total_delay = self.alpha * ( self.task_size[i] ** self.r)*((semantic_factor **(-1)-1)) / self.local_comp[i] + semantic_factor *  self.task_size[i]/ uplink_rate + semantic_factor *  self.task_size[i] * self.computing_density[i]/(resource_allocation)
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
        w1,w2 = 1,10
        w3 = 1
        w = 1 # w1+w2+w3
    
        for i in range(self.agent_num):
            # fairness_reward = jain_fairness
            # sub_agent_reward.append(fairness_reward/E_total_norm) run11
            # sub_agent_reward.append(fairness_reward/E_total_norm+time_penalty_space[i]+resource_allocation_penalty)
            sub_agent_reward.append(w1/w*self.agent_num/E_total + (w2/w)*total_time_penalty ) # + 1*resource_allocation_penalty)
        sub_agent_info=[E_total,total_time_penalty,resource_allocation_penalty]
        #绘制rewards曲线
        #所有agent的平均奖励
        # writer.add_scalar('reward/average_reward', np.mean(sub_agent_reward), self.episode_count)
        #每个agent的奖励
        return [self.sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
