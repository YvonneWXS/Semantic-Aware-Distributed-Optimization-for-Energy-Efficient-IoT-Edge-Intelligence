import copy
import math

import numpy as np

class ENV():
    def __init__(self, UEs, MECs, k, lam):
        np.random.seed(47)
        self.UEs = UEs
        self.MECs = MECs
        self.k = k  # 表示离散化参数

        # 创建动作空间
        offload_decision = np.array([0, 1]).reshape((2, 1))  # 是否卸载
        semantic_factor = np.linspace(0.1, 1, k).reshape((k, 1))  # 语义提取因子--设置为7是假设离散化为7个值，0.4、0.5...
        resource_allocation = np.linspace(0.1, 0.5, k).reshape((k, 1))  # MEC服务器分配的计算资源比例
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
        self.transmission_power = np.random.uniform(0.1, 0.5)  # 传输功率0.1W-0.5W
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
    def reset(self):
        np.random.seed(1)
        obs = []
        servers_cap = []
        new_cap = True
        for i in range(self.UEs):
            task_size = np.random.randint(1.5 * self.MB, 2 * self.MB)  # 任务大小
            computing_density = np.random.uniform(300, 500)  # 处理任务每比特数据的成本
            local_comp = self.UE_params[i]['local_comp']
            local_delay = task_size * computing_density / local_comp  # 本地处理任务时间
            max_delay = np.random.uniform(local_delay, 2 * local_delay)  # 任务最大容忍时间随机取
            observation = np.array([task_size, computing_density, max_delay])
            obs.append(observation)
            new_cap = False
        return obs

    def choose_action(self, prob):
        """
        根据概率选择动作
        :param env:
        :param prob:
        :return: [[offload_decision, semantic_factor,resource_allocation]]
        """
        action_choice = np.linspace(0, 1, self.k)
        actions = []
        for i in range(self.UEs):
            # print(len(prob[i]), 1 + self.k * 7)
            prob[i] = np.nan_to_num(prob[i], nan=0.0)  # 将 NaN 替换为 0
            # prob[i] = prob[i] / np.sum(prob[i])  # 确保概率和为 1   
            a = np.random.choice(a=int(1+(self.k)*7*(self.k/10)), p=prob[i])  # 在数组p中从a个数字中以概率p选中一个（探索-利用权衡）
            offload_decision = self.actions[a][0]
            semantic_factor = self.actions[a][1]
            resource_allocation = self.actions[a][2]
            action = [offload_decision, semantic_factor, resource_allocation]
            actions.append(action)
        return actions

    def step(self, observation, actions_prob, epoch, step, is_prob=True, is_compared=True):
        # print(np.random.uniform(1,100))
        if is_prob: #MAPPDG
            actions = self.choose_action(actions_prob)
        else: actions = actions_prob #DQN/D3QN
        new_cap = False

        total_resource_allocation = np.sum([action[2] for action in actions])
        # if(total_resource_allocation > 1): 
        #     for action in actions:
        #         action[2] = action[2] / total_resource_allocation
        #         resource_penalty = - max(0, total_resource_allocation - 1) 
        # else: resource_penalty = 0
        offload_num = np.sum([action[0] for action in actions])
        if offload_num >0: W = self.transmission_bandwidth/offload_num


        # if resource_penalty <0: violation_num = 1
        # else: violation_num = 0
        
        obs_ = []
        rew= []
        dpg_energys, local_energys, ran_energys, mec_energys = [], [], [], []
        total_delay_array=[]
        energy_array = []
        dpg_info=[]

        for i in range(self.UEs):
            if i == self.UEs - 1: new_cap = True
            # 提取信息
            task_size, computing_density, max_delay = observation[i]
            local_comp = self.UE_params[i]['local_comp']
            channel_gain = self.UE_params[i]['channel_gain']
            uplink_rate = W * math.log2 (1 + self.transmission_power * channel_gain /(W *self.noise_power))
            E_max = self.T* self.κ *(local_comp **3)

            action = actions[i]
            offload_decision, semantic_factor, resource_allocation = int(action[0]), action[1], action[2]

            # 全本地
            local_only_energy = self.κ * task_size * computing_density * local_comp**2
            local_only_time = task_size * computing_density / local_comp
            # print("local_only_time:", local_only_time)

            # 全边缘
            upload_energy = self.transmission_power * task_size / uplink_rate
            upload_delay = task_size / uplink_rate + task_size * computing_density / (self.MEC_f /self.UEs)
            # print("upload_delay:", upload_delay)

            # 随机卸载
            offload_decision_random = np.random.choice([0, 1])  # 随机选择 0 或 1
            semantic_factor_random = np.random.uniform(0.3,1)  
            resource_allocation_random = np.random.uniform(0,1)
            random_SEtask_energy = self.κ * self.alpha * (task_size ** self.r)*((semantic_factor_random **(-self.beta)-1)) * local_comp**2 
            random_total_energy = (1-offload_decision_random) * local_only_energy + offload_decision_random * (random_SEtask_energy + upload_energy * semantic_factor_random)

            # 计算奖励
            # 强化学习算法
            RL_SEtask_energy = self.κ * self.alpha * (task_size ** self.r)*((semantic_factor **(-self.beta)-1)) * local_comp**2
            if offload_decision == 0:
                RL_total_energy = local_only_energy
                RL_total_delay = local_only_time
            else:
                RL_total_energy = RL_SEtask_energy + upload_energy * semantic_factor
                RL_total_delay = self.alpha * (task_size ** self.r)*((semantic_factor **(-1)-1)) / local_comp + semantic_factor * task_size/ uplink_rate + semantic_factor * task_size * computing_density/(self.MEC_f* resource_allocation)

            #这里reward要设置成数组，让后面训练时计算正确的奖励（公平性指标+违反约束惩罚）
            delay_penalty=-max(0, RL_total_delay - max_delay) /max_delay #归一化，确保奖励数值范围类似
            total_delay_array.append(delay_penalty)
            energy_array.append(RL_total_energy)
            # reward_energy = RL_total_energy 
            # reward_array = np.array([delay_penalty, reward_energy, resource_penalty])
            # rew.append(reward_array)

            local_energys.append(local_only_energy)
            mec_energys.append(upload_energy)
            ran_energys.append(random_total_energy)
            dpg_energys.append(RL_total_energy)
            # print('local_only_energy:', local_only_energy, 'upload_energy:', upload_energy, 'random_total_energy:', random_total_energy, 'RL_total_energy:', RL_total_energy)
            #更改状态——》不改变状态
            if new_cap:
                np.random.seed(1+1+step)#保证各算法的训练数据一致
                for i in range(self.UEs):
                    task_size = np.random.randint(1.5 * self.MB, 2 * self.MB)  # 任务大小
                    computing_density = np.random.uniform(300, 500)  # 处理任务每比特数据的成本
                    local_comp = self.UE_params[i]['local_comp']
                    local_delay = task_size * computing_density / local_comp  # 本地处理任务时间
                    max_delay = np.random.uniform(local_delay, 2 * local_delay)  # 任务最大容忍时间随机取
                    observation_ = np.array([task_size, computing_density, max_delay])
                    obs_.append(observation_)
            # obs_.append(observation[i])
        
        # 计算奖励
        # w1,w2,w3=2,1,1 converge1
        w1,w2,w3 = 1,20,200
        global_reward=self.UEs/np.sum(energy_array)+np.sum(total_delay_array)
        for i in range(self.UEs):
            rew.append(global_reward)

        dpg_info = [ np.sum(total_delay_array)] 

        if is_compared:
            return obs_, rew, dpg_energys, local_energys, mec_energys, ran_energys, dpg_info
        else:
            return obs_, rew, dpg_energys, dpg_info
            # return obs_, total