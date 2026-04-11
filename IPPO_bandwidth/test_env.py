#!/usr/bin/env python
"""
快速测试环境是否可正常初始化与执行一步
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.env_discrete import DiscreteActionEnv
import numpy as np

def test_env():
    print("Testing DiscreteActionEnv...")
    env = DiscreteActionEnv()
    print(f"Number of agents: {env.num_agent}")
    print(f"Observation space: {env.observation_space[0].shape}")
    print(f"Action space: {env.action_space}")

    # 重置环境
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    # 生成随机动作（符合动作空间维度）
    # 动作空间是 MultiDiscrete，每个agent输出4个离散索引
    # 但环境期望的输入是 one‑hot 向量（总维度24）
    # 这里我们直接构造一个合法的 one‑hot 动作（全部取第一个选项）
    actions = []
    for i in range(env.num_agent):
        action_onehot = np.zeros(24)
        action_onehot[0] = 1   # offload_decision = 0 (本地处理)
        action_onehot[2] = 1   # semantic_factor 第一个选项 (0.3)
        action_onehot[10] = 1  # resource_allocation 第一个选项 (0.1)
        action_onehot[20] = 1  # bandwidth_weight 第一个选项 (0)
        actions.append(action_onehot)
    actions = np.array(actions)

    # 执行一步
    obs, rewards, dones, infos = env.step(actions, episode=0, step=0)
    print(f"Step completed.")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Rewards: {rewards}")
    print(f"Infos: {infos}")

    print("Environment test passed!")

if __name__ == "__main__":
    try:
        test_env()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)