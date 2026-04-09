import copy
import math
import numpy as np

# 导入配置文件
import config

class ENV():
    def __init__(self, UEs, MECs, k=None, total_bandwidth_hz=None, mec_capacity_hz=None):
        np.random.seed(47)
        self.UEs = UEs
        self.MECs = MECs
        # 如果提供了k，则使用它，否则从配置中获取
        self.k = k if k is not None else config.k_discretization  # 表示离散化参数
        discrete_step = 1.0 / self.k  # 离散化步长

        # 创建动作空间 - 扩展为包含带宽权重
        offload_decision = np.array([0, 1]).reshape((2, 1))  # 是否卸载
        semantic_factor = np.arange(discrete_step, 1.0 + discrete_step, discrete_step).reshape(-1, 1) # 语义提取因子
        resource_allocation = np.arange(discrete_step, 1.0 + discrete_step, discrete_step).reshape(-1, 1)  # MEC服务器分配的计算资源比例
        semantic_threshold = config.min_semantic_factor_list[0]  # 语义提取因子最低阈值

        # 组合动作空间 [offload, semantic, mec_resource, tx_power, bw_weight]
        actions = []
        for offload in offload_decision:
            if offload[0] == 0:
                # 不卸载时，语义提取因子为1，计算资源比例为0，功率为0，带宽权重为0
                actions.append([offload[0], 1, 0, 0, 0])
            else:
                # 卸载时，语义提取因子必须大于阈值
                for semantic in semantic_factor:
                    if semantic[0] > semantic_threshold:  # 语义提取因子大于阈值
                        for resource in resource_allocation:
                            for power in config.power_list:
                                for bw_weight in config.bw_weights:
                                    if bw_weight > 0:  # 只有带宽权重大于0时才有效
                                        actions.append([offload[0], semantic[0], resource[0], power, bw_weight])


        actions2 = []
        for offload in offload_decision:
            if offload[0] == 0:
                # 不卸载时，语义提取因子为1
                actions2.append([offload[0], 1])
            else:
                # 卸载时，语义提取因子必须大于阈值
                for semantic in semantic_factor:
                    if semantic[0] > semantic_threshold:  # 语义提取因子大于阈值
                        actions2.append([offload[0], semantic[0]])
        self.actions2 = np.array(actions2)

        self.actions = np.array(actions)
        self.n_actions = len(self.actions)
        # print('动作空间：', self.n_actions)
        self.n_features = 4  # 增加为4，包含本地能耗
        #对于一个agent的环境观察值数量（会变化）：任务大小，处理任务每比特数据的成本，任务的最大处理延迟，本地能耗
        self.discount = 0  # 计算下行链路时间、能耗的折扣因子（本文不考虑，故为0）

        # 基本参数 - 从配置文件加载
        self.Hz = 1
        self.kHz = 1000 * self.Hz
        self.mHz = 1000 * self.kHz
        self.GHz = 1000 * self.mHz

        # 数据大小
        self.bit = 1
        self.B = 8 * self.bit
        self.KB = 1024 * self.B
        self.MB = 1024 * self.KB

        # 模拟参数 - 从配置文件加载
        self.κ = config.kappa  # 芯片结构对cpu处理的影响因子
        self.r = config.r_se  # 运行语义提取任务的CPU周期数的参数1
        self.alpha = config.alpha_se  # 运行语义提取任务的CPU周期数的参数2
        self.beta = config.beta_se  # 运行语义提取任务的CPU周期数的参数3
        # 使用传入的参数或配置中的默认值
        self.total_bandwidth = total_bandwidth_hz if total_bandwidth_hz is not None else config.total_bandwidth_kHz[0] * 1000  # 总传输带宽
        self.MEC_f = mec_capacity_hz if mec_capacity_hz is not None else config.MEC_freq_base  # MEC的计算能力
        self.T = config.slot_duration  # 每个时隙时长

        # 随机生成每个UE的计算能力等参数
        self.UE_params = []  # 初始化UE参数列表
        for i in range(self.UEs):
            local_comp = np.random.randint(1.5 * self.GHz, 2 * self.GHz)    # UE的本地计算能力
            distance = np.random.uniform(10, 100)  # 随机生成用户设备和基站之间的距离
            channel_gain = 1e-3 * (1.0 / distance) ** 2.5  # 计算信道增益
            # 将每个UE的参数存储在字典中
            ue_params = {
                'local_comp': local_comp,
                'channel_gain': channel_gain,
            }
            self.UE_params.append(ue_params)


    #更改每step的状态
    def reset(self, step, data_size_bytes=None):
        np.random.seed(step+1)
        obs = []
        for i in range(self.UEs):
            # 如果提供了数据大小，则使用，否则从配置中随机生成
            if data_size_bytes is not None:
                task_size = data_size_bytes
            else:
                # 从配置获取数据大小（注意单位转换：KB to bytes）
                task_size = np.random.uniform(config.DataSize_KB[0] * 1024, config.DataSize_KB[-1] * 1024)  # 任务大小 (Bytes)
            computing_density = np.random.uniform(300, 500)  # 处理任务每比特数据的成本
            local_comp = self.UE_params[i]['local_comp']
            local_delay = task_size * computing_density / local_comp  # 本地处理任务时间
            local_energy = self.κ * task_size * computing_density * local_comp**2
            # 根据配置生成最大时延
            max_delay = np.random.uniform(local_delay, config.max_delay_factor * local_delay)  # 任务最大容忍时间随机取
            observation = np.array([task_size, computing_density, max_delay, local_energy])
            obs.append(observation)
        return obs

    def dynamic_bandwidth_allocation(self, bw_weights):
        """
        动态带宽分配机制 - 基于离散权重的归一化分配
        参数: bw_weights - 每个UE的带宽权重数组
        返回: 每个需要卸载的UE分配到的实际带宽
        """
        # 计算非零权重的总和，用于归一化
        total_weights = sum(w for w in bw_weights if w > 0)

        # 如果总权重为0，说明没有UE需要卸载，返回0带宽
        if total_weights == 0:
            return [0.0] * len(bw_weights)

        # 归一化分配带宽
        allocated_bandwidths = []
        for i in range(len(bw_weights)):
            if bw_weights[i] > 0:
                # 计算该UE应得的带宽比例
                proportion = bw_weights[i] / total_weights
                # 计算理想带宽
                ideal_bw = proportion * self.total_bandwidth
                # 进行向下取整量化，模拟RB资源块
                # 这里假设最小资源单位为10kHz
                min_rb_unit = 10 * 10**3  # 10kHz
                quantized_bw = math.floor(ideal_bw / min_rb_unit) * min_rb_unit
                allocated_bandwidths.append(max(0, quantized_bw))
            else:
                # 如果权重为0，则分配0带宽
                allocated_bandwidths.append(0.0)

        return allocated_bandwidths

    def compute_energy_and_delay(self, offload_decision, semantic_factor, resource_allocation, power_allocation, bw_weight_allocation, observation):
        """
        修改后的能耗和时延计算方法，使用动态带宽分配
        """
        UE_energy = 0
        delay_total_penalty = 0

        # 计算动态带宽分配
        allocated_bandwidths = self.dynamic_bandwidth_allocation(bw_weight_allocation)

        for i in range(self.UEs):
            task_size, computing_density, max_delay, local_energy = observation[i]
            local_comp = self.UE_params[i]['local_comp']
            channel_gain = self.UE_params[i]['channel_gain']

            if offload_decision[i] == 0:  # 本地处理
                total_energy = local_energy
                total_delay = task_size * computing_density / local_comp
            else:  # 卸载服务器处理
                # 使用动态分配的带宽计算上行速率
                assigned_bandwidth = allocated_bandwidths[i]
                if assigned_bandwidth <= 0:
                    # 如果分配的带宽为0，使用一个小的默认值以避免除零错误
                    assigned_bandwidth = 10 * 10**3  # 10kHz

                uplink_rate = assigned_bandwidth * math.log2(1 + power_allocation[i] * channel_gain / (assigned_bandwidth * config.noise_power))
                upload_energy = power_allocation[i] * (semantic_factor[i] * task_size) / uplink_rate   # 修改：使用语义因子影响传输能耗
                SEtask_energy = self.κ * self.alpha * (task_size ** self.r) * (semantic_factor[i] **(-self.beta)-1) * local_comp**2
                total_energy = SEtask_energy + upload_energy

                # 修改时延计算，包括语义提取、传输和MEC计算
                se_delay = self.alpha * (task_size ** self.r) * ((semantic_factor[i] **(-1)-1)) / local_comp
                tx_delay = semantic_factor[i] * task_size / uplink_rate
                mec_delay = semantic_factor[i] * task_size * computing_density / (self.MEC_f * resource_allocation[i])
                total_delay = se_delay + tx_delay + mec_delay

            UE_energy += total_energy
            delay_penalty = max(0, total_delay - max_delay)
            delay_total_penalty += delay_penalty

        return UE_energy, delay_total_penalty