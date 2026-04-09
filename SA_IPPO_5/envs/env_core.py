import numpy as np
import math

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        np.random.seed(47)
        self.agent_num = 10 # TODO 设置智能体的个数
        self.obs_dim = 3  # 设置智能体的观测维度 ：task_size, computing_density, max_delay

        # 动作维度：卸载(1) + 语义(1) + 资源(1) + 功率(1) = 4个离散头
        self.action_dim = 4  # 设置智能体的动作维度：offload_decision, semantic_factor, resource_allocation, transmission_power
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
        # self.transmission_power = np.random.uniform(0.02, 0.1)  # 传输功率0.02W-0.1W
        self.noise_power = 10**(-20) # 噪声功率-170dBm
        #不考虑邻道干扰功率
        self.MEC_f = 20 * self.GHz  # MEC的计算能力
        # self.weight_factor = 0.7  # 能耗和公平性的权重


        #随机生成每个UE的计算能力等参数
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
        return self.sub_agent_obs

    def step(self, actions, episode, step):
        """
        动作输入 actions 为 one-hot 拼接向量。
        假设总维度 25 = 2(卸载) + 8(语义) + 10(资源) + 5(功率)

        用离散动作空间one-hot编码值计算reward
        """
        self.cur_agent_obs = self.sub_agent_obs
        cur_agent_reward = []
        sub_agent_done = []
        agent_energy = []
        time_penalty_space = []
        resource_allocation_space = []
        local_energy = []
        offload_num = 0

        # 第一遍循环：解析动作，统计卸载人数，预处理资源分配
        for i in range(self.agent_num):
            #提取每个agent的action
            action = actions[i]

            # 1. 卸载决策 (Indices 0-1)
            offload_decision = np.argmax(action[:2])  # 获取索引并计算对应的值 
            offload_num = offload_num + offload_decision

            # 2. 语义因子 

            # 3. 资源分配 (Indices 10-20, 10个值)
            # 范围 0.1 - 1.0
            resource_allocation= (np.argmax(action[10:20])+1)*0.1*self.MEC_f    # 资源分配    #🌟
            if offload_decision == 1: resource_allocation_space.append(resource_allocation)
            else:
                resource_allocation_space.append(0)
        
        # 带宽均分
        if offload_num > 0:
            W = self.transmission_bandwidth/offload_num  #每个用户设备的带宽分配

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

            # 2. 语义因子 (Indices 2-10, 8个值)
            # 范围 [0.3, 1.0]. Index 0 -> 0.3, Index 7 -> 1.0
            # semantic_factor = (np.argmax(action[2:10])+1)*0.1 + 0.3 # 语义因子   #🌟
            semantic_factor = np.argmax(action[2:10]) * 0.1 + 0.3

            # 4. 发射功率 (Indices 20-25, 5个值)
            # 范围 [0.02, 0.1]. Index 0 -> 0.02, Index 4 -> 0.1
            power_idx = np.argmax(action[20:])  # 50个
            transmission_power = 0.02 + power_idx * 0.02

            resource_allocation= resource_allocation_space[i]

            # 计算上传带宽,本地计算机能耗
            RL_local_energy = self.κ * self.task_size[i] * self.computing_density[i] * (self.local_comp[i]**2)
            local_energy.append(RL_local_energy)
            # print("upload_delay:",upload_delay)

            if offload_decision == 0:#本地处理
                RL_total_energy = RL_local_energy
                RL_total_delay = self.local_delay[i]
                transmission_power = 0 
            else:
                # 卸载执行
                # 1. 语义提取能耗 (Local)
                # 论文修正公式: E_sem = kappa * alpha * (D^eta) * (beta^(-k) - 1) * f^2
                # 代码对应: eta -> r, k -> beta
                RL_SEtask_energy = self.κ * self.alpha * ( self.task_size[i] ** self.r) * ((semantic_factor ** (-self.beta) - 1)) * (self.local_comp[i]**2)

                # 2. 传输速率 (Eq. 4) 
                uplink_rate = W * math.log2 (1 + transmission_power * self.channel_gain[i] / (W * self.noise_power))
                
                # 3. 传输能耗 (Eq. 5) 
                # 数据量 = task_size * semantic_factor
                upload_energy = transmission_power *  self.task_size[i] *semantic_factor / uplink_rate
                # mec_energy = self.κ_mec *(resource_allocation**3) * semantic_factor *  self.task_size[i] * self.computing_density[i]/(resource_allocation)
                
                # 总能耗
                RL_total_energy = RL_SEtask_energy + upload_energy # + mec_energy

                # 总时延 = 语义提取时延 + 传输时延 + MEC处理时延
                # RL_total_delay = self.alpha * ( self.task_size[i] ** self.r)*((semantic_factor **(-1)-1)) / self.local_comp[i] + semantic_factor *  self.task_size[i]/ uplink_rate + semantic_factor *  self.task_size[i] * self.computing_density[i]/(resource_allocation)
                se_delay = self.alpha * (self.task_size[i] ** self.r) * ((semantic_factor ** (-1) - 1)) / self.local_comp[i]
                tx_delay = (semantic_factor * self.task_size[i]) / uplink_rate
                mec_delay = (semantic_factor * self.task_size[i] * self.computing_density[i]) / resource_allocation
                
                RL_total_delay = se_delay + tx_delay + mec_delay
            
            
            
            
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

        # -------------------------计算奖励-------------------------
        #min-max归一化能耗(可能会影响收敛)
        E_total = sum(agent_energy)

        #计算资源分配惩罚
        resource_allocation_penalty = -max(0,sum(resource_allocation_space)-self.MEC_f)/self.MEC_f

        #时间约束惩罚
        total_time_penalty = sum(time_penalty_space)

        #TODO 奖励权重
        w1,w2 = 1, 5 
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
