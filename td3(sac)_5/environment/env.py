import math
from environment.user_equipments import UE
from environment.uavs import UAV
import config
import numpy as np
import pandas as pd


class Env:
    def __init__(self) -> None:
        UE.initialize_ue_class()
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)]
        self._time_step: int = 0

        np.random.seed(47)
        #随机生成每个UE的计算能力等参数
        self.local_comp = np.zeros(config.NUM_UES)  # 本地计算能力
        self.distance = np.zeros(config.NUM_UES)  # 距离
        self.channel_gain = np.zeros(config.NUM_UES)  # 信道增益
        for i in range(config.NUM_UES):
            self.local_comp[i] = np.random.randint(1.5 * config.GHz , 2 * config.GHz)    # UE的本地计算能力
            self.distance[i] = np.random.uniform(10, 100)  # 随机生成用户设备和基站之间的距离
            self.channel_gain[i] = 1e-3 * (1.0 / self.distance[i]) **2.5  #计算信道增益
        
        self.cur_agent_obs = []
        self.sub_agent_obs = []
        
        df = pd.read_csv("obs_{}.csv".format(config.NUM_UES))
        self.data_dict = df.set_index(['step', 'ue_id']).to_dict(orient='index')

    @property
    def uavs(self) -> list[UAV]:
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        return self._ues

    def reset(self) -> list[np.ndarray]:
        # np.random.seed(1)# 是否要去掉注释？
        self.sub_agent_obs = []
        self.task_size = np.zeros(config.NUM_UES)
        self.computing_density = np.zeros(config.NUM_UES)
        self.local_delay = np.zeros(config.NUM_UES)
        self.max_delay = np.zeros(config.NUM_UES)
        step = 0
        for i in range(config.NUM_UES):
            self.task_size[i] = self.data_dict[(step, i)]['task_size']  # 任务大小
            self.computing_density[i] = self.data_dict[(step, i)]['computing_density']  # 处理任务每比特数据的成本
            self.local_delay[i] = self.task_size[i] * self.computing_density[i] / self.local_comp[i]  # 本地处理任务时间
            self.max_delay[i] = self.data_dict[(step, i)]['max_delay']  # 任务最大容忍时间随机取
            observation = np.array([self.task_size[i], self.computing_density[i], self.max_delay[i]])
            self.sub_agent_obs.append(observation)
        return self.sub_agent_obs 
        # return self._get_obs()

    def step(self, actions: np.ndarray, step: int, visualize: bool = False) -> tuple[list[np.ndarray], list[float], tuple[float, float, float]]:
        """Execute one time step of the simulation.""" # step function
        self._time_step += 1

        self.sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        agent_energy = []
        time_penalty_space = []
        resource_allocation_space = []
        local_energy = []

        # action离散化处理
        offloading_decision = np.where(actions[:, 0] >= 0, 1, 0)  # from [-1, 1] to {0, 1}
        resource_allocation_cont = (actions[:, 1] + 1) / 2 + config.MIN_RESOURCE_ALLOCATION #(config.MAX_RESOURCE_ALLOCATION - config.MIN_RESOURCE_ALLOCATION+0.1)1  # from [-1, 1] to[0.1, 1.1]
        transmission_power_cont = (actions[:, 2] + 1) / 2 + config.MIN_TRANSMIT_POWER # (config.MAX_TRANSMIT_POWER - config.MIN_TRANSMIT_POWER+0.1)+0.1 # from [-1, 1] to [0.1,0.6] 
        semantic_factor_cont = (actions[:, 3] + 1) / 2 * (config.MAX_SEMANTIC_EXTRACTION_FACTOR - config.MIN_SEMANTIC_EXTRACTION_FACTOR+0.1)+ config.MIN_SEMANTIC_EXTRACTION_FACTOR # from [-1, 1] to [0.3,1.1]
        # === 离散化并向下取整 === 
        resource_allocation = self.discretize_action(
            resource_allocation_cont,
            config.MIN_RESOURCE_ALLOCATION,
            config.MAX_RESOURCE_ALLOCATION,
            step=0.1
        )

        transmission_power = self.discretize_action(
            transmission_power_cont,
            config.MIN_TRANSMIT_POWER,
            config.MAX_TRANSMIT_POWER,
            step=0.1
        )

        semantic_factor = self.discretize_action(
            semantic_factor_cont,
            config.MIN_SEMANTIC_EXTRACTION_FACTOR,
            config.MAX_SEMANTIC_EXTRACTION_FACTOR,
            step=0.1
        )
        resource_allocation = offloading_decision * resource_allocation  # if offloading_decision is 0, resource_allocation is 0
        #归一化分配资源
        total_resource_allocation = np.sum(resource_allocation)
        if total_resource_allocation > 1.0:
            resource_allocation = resource_allocation / total_resource_allocation  # normalize to make sure the sum is not greater than 1.0
        offload_num = np.sum(offloading_decision)
        #每个用户设备的带宽分配
        if offload_num > 0:
            W = config.transmission_bandwidth / offload_num  

        # print("offloading_decision:", offloading_decision)
        # print("resource_allocation:", resource_allocation)
        # print("transmission_power:", transmission_power)
        # print("semantic_factor:", semantic_factor)
        
        #计算reward,energy,violation rate
        for i in range(config.NUM_UES):
            #计算上传带宽
            RL_local_energy = config.κ * self.task_size[i] * self.computing_density[i] * (self.local_comp[i]**2)
            local_energy.append(RL_local_energy)
            if offloading_decision[i] == 0:#本地处理 
                RL_total_energy = RL_local_energy
                RL_total_delay = self.local_delay[i] 
            else:#边缘处理
                RL_SEtask_energy = config.κ * config.alpha * ( self.task_size[i] ** config.r)*((semantic_factor[i] **(-config.beta)-1)) * (self.local_comp[i]**2)
                uplink_rate = W * math.log2 (1 + transmission_power[i] * self.channel_gain[i] / (W * config.noise_power))
                upload_energy = transmission_power[i] *  self.task_size[i] *semantic_factor[i] / uplink_rate
                # mec_energy = self.κ_mec *(resource_allocation**3) * semantic_factor *  self.task_size[i] * self.computing_density[i]/(resource_allocation)
                RL_total_energy = RL_SEtask_energy + upload_energy # + mec_energy
                RL_total_delay = config.alpha * ( self.task_size[i] ** config.r)*((semantic_factor[i] **(-1)-1)) / self.local_comp[i] + semantic_factor[i] *  self.task_size[i]/ uplink_rate + semantic_factor[i] *  self.task_size[i] * self.computing_density[i]/(resource_allocation[i]*config.MEC_f)            

            agent_energy.append(RL_total_energy)

            #计算时间约束惩罚
            time_penalty = -max(0,  RL_total_delay-self.max_delay[i])/ self.max_delay[i]
            time_penalty_space.append(time_penalty)
            
            #其余信息
            sub_agent_done.append(False)#智能体不提前终止
            # sub_agent_info.append({})#附加信息存储
        
        # np.random.seed(2+step)
        for i in range(config.NUM_UES):    
            #更新智能体状态
            # 随机生成每个智能体的观测值
            self.task_size[i] = self.data_dict[(step+1, i)]['task_size']  # 任务大小
            self.computing_density[i] = self.data_dict[(step+1, i)]['computing_density']  # 处理任务每比特数据的成本
            self.local_delay[i] = self.task_size[i] * self.computing_density[i] / self.local_comp[i]  # 本地处理任务时间
            self.max_delay[i] = self.data_dict[(step+1, i)]['max_delay']  # 任务最大容忍时间随机取
            observation = np.array([self.task_size[i], self.computing_density[i], self.max_delay[i]])
            self.sub_agent_obs.append(observation)
        
        E_total = sum(agent_energy)
        #时间约束惩罚
        total_time_penalty = sum(time_penalty_space)
        
        #奖励权重
        w1,w2 = 1,20 #TODO 调节权重
        w = w1+w2
        cur_agent_reward = []
        for i in range(config.NUM_UES):
            cur_agent_reward.append(w1/w* config.NUM_UES /E_total + (w2/w)*total_time_penalty)#(w3/w)*resource_allocation_penalty)

        return self.sub_agent_obs, cur_agent_reward, E_total, total_time_penalty

    def discretize_action(self, action_values, min_val, max_val, step):
        # Clip to [min_val, max_val]
        action_values = np.clip(action_values, min_val, max_val)

        # Floor instead of ceil
        steps = np.floor((action_values - min_val) / step)

        # Compute discretized values
        discretized = min_val + steps * step

        # Final clip to avoid floating errors
        return np.clip(discretized, min_val, max_val)




    