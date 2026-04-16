"""
# @Time    : 2021/7/2 5:22 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : env_discrete.py
"""

import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore


class DiscreteActionEnv(object):
    """
    对于离散动作环境的封装
    Wrapper for discrete action environment.
    """

    def __init__(self):
        self.env = EnvCore()
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        offload_decision = np.array([0, 1])  # 卸载决策
        k = 100  # 离散化数值
        semantic_factor = np.linspace(0.1, 1, k)  # 语义因子范围: 0.1-1.0
        resource_allocation = np.linspace(0.1, 1, k)
        semantic_threshold = 0.3 #这里暂不考虑
        # 为每个维度定义 [min, max] 范围
        action_space_params = [
            [0, 1],      # offload_decision: 0 or 1
            [0, k-1],    # resource_allocation: 0.1-1.0 (100个离散值)
            [0, 3],      # bw_weight: 0,1,2,3 (4个离散值，用于DW-DNA带宽分配)
            [0, k-1]     # semantic_factor: 0.1-1.0 (100个离散值)
        ]

        # 为每个智能体创建相同的动作空间
        for agent_idx in range(self.num_agent):
            # 动作空间：使用MultiDiscrete定义4个离散动作维度
            self.action_space.append(spaces.MultiDiscrete(action_space_params))

            # 观测空间
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )  # [-inf,inf]

        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def step(self, actions, episode, step):
        """
        输入actions维度假设：
        # actions shape = (5, 2, 206)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的206维编码
        # 动作空间结构（one-hot编码）：
        # - 维度0-1: 卸载决策 (0或1)
        # - 维度2-101: 资源分配 (100个离散值，0.1-1.0)
        # - 维度102-105: 带宽权重 (4个离散值，0,1,2,3)
        # - 维度106-205: 语义因子 (100个离散值，0.1-1.0)
        Input actions dimension assumption:
        # actions shape = (5, 2, 206)
        # 5 threads of the environment, with 2 intelligent agents inside, and each intelligent agent's action is a 206-dimensional one_hot encoding
        """

        results = self.env.step(actions, episode, step)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass




if __name__ == "__main__":
    DiscreteActionEnv().step(actions=None)
