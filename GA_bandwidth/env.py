import copy
import math

import numpy as np


class ENV():
    def __init__(self, UEs, MECs, k, total_bandwidth=1000, mec_capacity=20.0, min_semantic_factor=0.3):
        """
        初始化边缘计算环境

        参数:
            UEs: 用户设备数量
            MECs: MEC服务器数量
            k: 离散化参数
            total_bandwidth: 总带宽 (kHz)
            mec_capacity: MEC计算能力 (Giga Cycles/s)
            min_semantic_factor: 语义提取因子最低阈值
        """
        np.random.seed(47)
        self.UEs = UEs
        self.MECs = MECs
        self.k = k  # 表示离散化参数

        # 新增：带宽相关参数
        self.total_bandwidth = total_bandwidth  # 总带宽 (kHz)
        self.mec_capacity = mec_capacity * 1e9  # MEC计算能力 (cycles/s)
        self.min_semantic_factor = min_semantic_factor  # 语义提取因子最低阈值
        self.bandwidth_weight_set = np.array([0, 1, 2, 3])  # 带宽权重离散集合

        discrete_step = 1.0 / self.k  # 离散化步长
        # 创建动作空间
        offload_decision = np.array([0, 1]).reshape((2, 1))  # 是否卸载
        semantic_factor = np.arange(discrete_step, 1.0 + discrete_step, discrete_step).reshape(-1, 1)  # 语义提取因子
        resource_allocation = np.arange(discrete_step, 1.0 + discrete_step, discrete_step).reshape(-1, 1)  # MEC服务器分配的计算资源比例
        semantic_threshold = min_semantic_factor  # 语义提取因子最低阈值

        # 组合动作空间 (5维: offload, semantic, resource, power, bw_weight)
        actions = []
        for offload in offload_decision:
            if offload[0] == 0:
                # 不卸载时，语义提取因子为1，计算资源比例为0，带宽权重为0
                actions.append([offload[0], 1, 0, 0, 0])
            else:
                # 卸载时，语义提取因子必须大于阈值，带宽权重为1,2,3
                for semantic in semantic_factor:
                    if semantic[0] > semantic_threshold:  # 语义提取因子大于阈值
                        for resource in resource_allocation:
                            for bw_weight in self.bandwidth_weight_set[1:]:  # 1,2,3
                                actions.append([offload[0], semantic[0], resource[0], 0, bw_weight])

        self.actions = np.array(actions)
        self.n_actions = len(self.actions)
        self.n_features = 4  # 观察值数量: task_size, computing_density, max_delay, local_energy
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

        # 模拟参数
        self.κ = 10 ** (-27)  # 芯片结构对cpu处理的影响因子
        self.r = 1  # 运行语义提取任务的CPU周期数的参数1
        self.alpha = 1  # 运行语义提取任务的CPU周期数的参数2
        self.beta = 2  # 运行语义提取任务的CPU周期数的参数3
        self.transmission_bandwidth = 1 * self.mHz  # 传输带宽1MHz
        self.transmission_power = np.random.uniform(0.1, 0.5)  # 传输功率0.1W-0.5W
        self.noise_power = 10**(-20)  # 噪声功率-170dBm
        self.T = 1  # 每个时隙时长为1s

        # 随机生成每个UE的计算能力等参数
        self.UE_params = []  # 初始化UE参数列表
        for i in range(self.UEs):
            local_comp = np.random.randint(1.5 * self.GHz, 2 * self.GHz)  # UE的本地计算能力
            distance = np.random.uniform(10, 100)  # 随机生成用户设备和基站之间的距离
            channel_gain = 1e-3 * (1.0 / distance) ** 2.5  # 计算信道增益
            ue_params = {
                'local_comp': local_comp,
                'channel_gain': channel_gain,
            }
            self.UE_params.append(ue_params)

    # 重置环境状态
    def reset(self, a):
        """
        重置环境并返回初始观察值

        参数:
            a: 任务大小倍数 (用于生成不同任务大小)

        返回:
            obs: 每个UE的观察值列表
        """
        np.random.seed(47)
        obs = []
        for i in range(self.UEs):
            task_size = 256 * a * self.KB  # 任务大小
            computing_density = 450  # 处理任务每比特数据的成本
            local_comp = self.UE_params[i]['local_comp']
            local_delay = task_size * computing_density / local_comp  # 本地处理任务时间
            local_energy = self.κ * task_size * computing_density * local_comp ** 2
            max_delay = np.random.uniform(local_delay, 2 * local_delay)  # 任务最大容忍时间随机取
            observation = np.array([task_size, computing_density, max_delay, local_energy])
            obs.append(observation)
        return obs

    def compute_bandwidth_dwdna(self, bw_weights):
        """
        基于离散权重的动态归一化分配 (DW-DNA)

        参数:
            bw_weights: 每个UE的带宽权重数组

        返回:
            bandwidths: 每个UE分配的带宽数组 (kHz)
        """
        sum_weights = np.sum(bw_weights)

        # 处理边界情况：所有权重为0
        if sum_weights == 0:
            return np.zeros_like(bw_weights, dtype=float)

        # 归一化分配: B_i = (bw_weight_i / sum_weights) × total_bandwidth
        normalized_bw = (bw_weights / sum_weights) * self.total_bandwidth

        # 向下取整量化 (模拟RB资源块，最小单位1kHz)
        quantized_bw = np.floor(normalized_bw).astype(float)

        return quantized_bw

    def compute_energy_and_delay(self, offload_decision, semantic_factor, resource_allocation,
                           transmission_power, bw_weight, observation):
        """
        计算能耗和时延

        参数:
            offload_decision: 卸载决策数组 (0=本地, 1=卸载)
            semantic_factor: 语义提取因子数组
            resource_allocation: MEC资源分配比例数组
            transmission_power: 传输功率数组
            bw_weight: 带宽权重数组
            observation: 观察值列表

        返回:
            total_energy: 总能耗
            delay_penalty: 违反时延约束的惩罚值
        """
        total_energy = 0
        delay_total_penalty = 0

        # 使用DW-DNA计算带宽
        bandwidths = self.compute_bandwidth_dwdna(bw_weight)

        for i in range(self.UEs):
            task_size, computing_density, max_delay, local_energy = observation[i]
            local_comp = self.UE_params[i]['local_comp']
            channel_gain = self.UE_params[i]['channel_gain']

            if offload_decision[i] == 0:
                # 本地处理
                total_energy += local_energy
                total_delay = task_size * computing_density / local_comp
            else:
                # 卸载处理
                # 使用动态带宽计算上行速率
                W_i = bandwidths[i]
                if W_i <= 0:
                    # 带宽为0时，无法传输
                    total_energy += 1e10  # 惩罚值
                    total_delay = 1e10
                else:
                    # SNR = P * |h|^2 / (W * N0)
                    snr = transmission_power[i] * channel_gain / (W_i * self.noise_power)
                    uplink_rate = W_i * math.log2(1 + snr)
                    if uplink_rate <= 0:
                        uplink_rate = 1e-10

                    # 语义提取能耗
                    SE_task_energy = self.κ * self.alpha * (task_size ** self.r) * \
                                  (semantic_factor[i] ** (-self.beta) - 1) * local_comp ** 2

                    # 传输能耗
                    upload_energy = transmission_power[i] * task_size / uplink_rate

                    # 语义提取本地处理时间
                    semantic_local_time = self.alpha * (task_size ** self.r) * \
                                    (semantic_factor[i] ** (-1) - 1) / local_comp

                    # 传输时间
                    upload_time = semantic_factor[i] * task_size / uplink_rate

                    # MEC处理时间
                    if resource_allocation[i] > 0:
                        mec_time = semantic_factor[i] * task_size * computing_density / \
                                  (self.mec_capacity * resource_allocation[i])
                    else:
                        mec_time = 0

                    total_energy += SE_task_energy + upload_energy
                    total_delay = semantic_local_time + upload_time + mec_time

            # 时延违反惩罚
            delay_penalty = max(0, total_delay - max_delay)
            delay_total_penalty += delay_penalty

        return total_energy, delay_total_penalty

    def get_action_space_size(self):
        """获取动作空间大小"""
        return self.n_actions

    def get_feature_size(self):
        """获取特征空间大小"""
        return self.n_features


def create_env(UEs=5, MECs=1, k=100, total_bandwidth=1000,
              mec_capacity=20.0, min_semantic_factor=0.3):
    """创建环境实例的工厂函数"""
    return ENV(UEs, MECs, k, total_bandwidth, mec_capacity, min_semantic_factor)