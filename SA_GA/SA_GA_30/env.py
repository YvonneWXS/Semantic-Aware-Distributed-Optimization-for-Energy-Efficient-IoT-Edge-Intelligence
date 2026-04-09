import copy
import math

import numpy as np

class ENV():
    def __init__(self, UEs, MECs, k):
        np.random.seed(47)
        self.UEs = UEs
        self.MECs = MECs
        self.k = k  # 表示离散化参数
        discrete_step = 1.0 / self.k  # 离散化步长
        # 创建动作空间
        offload_decision = np.array([0, 1]).reshape((2, 1))  # 是否卸载
        semantic_factor = np.arange(discrete_step, 1.0 + discrete_step, discrete_step).reshape(-1, 1) # 语义提取因子--设置为7是假设离散化为7个值，0.4、0.5...
        resource_allocation = np.arange(discrete_step, 1.0 + discrete_step, discrete_step).reshape(-1, 1)  # MEC服务器分配的计算资源比例
        semantic_threshold = 0.3  # 语义提取因子最低阈值

        # 组合动作空间
        actions = []
        for offload in offload_decision:
            if offload[0] == 0:
                # 不卸载时，语义提取因子为1，计算资源比例为0
                actions.append([offload[0], 1, 0])
            else:
                # 卸载时，语义提取因子必须大于0.3
                for semantic in semantic_factor:
                    if semantic[0] > semantic_threshold:  # 语义提取因子大于0.3
                        for resource in resource_allocation:
                            actions.append([offload[0], semantic[0], resource[0]])

        
        actions2 = []
        for offload in offload_decision:
            if offload[0] == 0:
                # 不卸载时，语义提取因子为1，计算资源比例为0
                actions2.append([offload[0], 1])
            else:
                # 卸载时，语义提取因子必须大于0.3
                for semantic in semantic_factor:
                    if semantic[0] > semantic_threshold:  # 语义提取因子大于0.3
                        actions2.append([offload[0], semantic[0]])
        self.actions2 = np.array(actions2)

        self.actions = np.array(actions)
        self.n_actions = len(self.actions)
        # print('动作空间：', self.n_actions)
        self.n_features = 3
        #对于一个agent的环境观察值数量（会变化）：任务大小，处理任务每比特数据的成本，任务的最大处理延迟
        self.discount = 0  # 计算下行链路时间、能耗的折扣因子（本文不考虑，故为0）


        # 基本参数
        # 频率
        self.Hz = 1
        self.kHz = 1000 * self.Hz
        self.mHz = 1000 * self.kHz
        self.GHz = 1000 * self.mHz

        # 数据大小
        self.bit = 1
        self.B = 8 * self.bit
        self.KB = 1024 * self.B
        self.MB = 1024 * self.KB

        #模拟参数
        self.κ = 10 ** (-27)  # 芯片结构对cpu处理的影响因子
        self.r = 1  #运行语义提取任务的CPU周期数的参数1...."若设为2会导致语义提取的能耗显著增大，不合适。。。"
        self.alpha = 1  #运行语义提取任务的CPU周期数的参数2
        self.beta =2 #运行语义提取任务的CPU周期数的参数3
        self.transmission_bandwidth = 1 * self.mHz   # 传输带宽1MHz
        # self.W = self.transmission_bandwidth/self.UEs  #每个用户设备的带宽分配
        # self.transmission_power = np.random.uniform(0.1, 0.5)  # 传输功率0.1W-0.5W
        self.noise_power = 10**(-20) # 噪声功率-170dBm
        #不考虑邻道干扰功率
        self.MEC_f = 20 * self.GHz  # MEC的计算能力
        self.T = 1 #每个时隙时长为1s

        #随机生成每个UE的计算能力等参数
        self.UE_params = []  # 初始化UE参数列表
        for i in range(self.UEs):  
            local_comp = np.random.randint(1.5 * self.GHz, 2 * self.GHz)    # UE的本地计算能力
            distance = np.random.uniform(10, 100)  # 随机生成用户设备和基站之间的距离
            channel_gain = 1e-3 * (1.0 / distance) ** 2.5  #计算信道增益
            #将每个UE的参数存储在字典中
            ue_params = {
                'local_comp': local_comp,
                'channel_gain': channel_gain,
            }
            self.UE_params.append(ue_params)
        

    #更改每step的状态
    def reset(self,step):
        np.random.seed(step+1)
        obs = []
        servers_cap = []
        new_cap = True
        for i in range(self.UEs):
            task_size = np.random.randint(1.5 * self.MB, 2 * self.MB)  # 任务大小
            computing_density = np.random.uniform(300, 500)  # 处理任务每比特数据的成本
            local_comp = self.UE_params[i]['local_comp']
            local_delay = task_size * computing_density / local_comp  # 本地处理任务时间
            local_energy = self.κ * task_size * computing_density * local_comp**2
            max_delay = np.random.uniform(local_delay, 2 * local_delay)  # 任务最大容忍时间随机取
            observation = np.array([task_size, computing_density, max_delay, local_energy])
            obs.append(observation)
            new_cap = False
        return obs
    
    def compute_energy_and_delay(self, offload_decision, semantic_factor, resource_allocation, transmission_power, observation):
        UE_energy=0
        delay_total_penalty = 0
        offload_num = np.sum(offload_decision)
        if offload_num > 0:
            W = self.transmission_bandwidth/offload_num  #每个用户设备的带宽分配            
        for i in range(self.UEs):
            task_size, computing_density, max_delay,local_energy = observation[i]
            local_comp = self.UE_params[i]['local_comp']
            channel_gain = self.UE_params[i]['channel_gain']
            if offload_decision[i] == 0:# 本地处理
                total_energy = local_energy
                total_delay = task_size * computing_density / local_comp
            else:# 卸载服务器处理
                uplink_rate = W * math.log2 (1 + transmission_power[i] * channel_gain /(W *self.noise_power))
                upload_energy = transmission_power[i] * task_size / uplink_rate   
                SEtask_energy = self.κ * self.alpha * (task_size ** self.r)*(semantic_factor[i] **(-self.beta)-1) * local_comp**2
                total_energy = SEtask_energy + upload_energy
                total_delay = self.alpha * (task_size ** self.r)*((semantic_factor[i] **(-1)-1)) / local_comp +\
                                        semantic_factor[i] * task_size/ uplink_rate + \
                                            semantic_factor[i] * task_size * computing_density/(self.MEC_f* resource_allocation[i])
            UE_energy += total_energy
            delay_penalty = max(0, total_delay - max_delay)
            delay_total_penalty += delay_penalty
        return UE_energy,delay_total_penalty
        