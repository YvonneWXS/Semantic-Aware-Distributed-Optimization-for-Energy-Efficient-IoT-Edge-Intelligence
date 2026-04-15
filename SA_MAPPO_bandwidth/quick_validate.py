#!/usr/bin/env python3
"""
快速验证脚本 - 验证SA-MAPPO关键修改

这个脚本用于快速检查关键修改是否正常工作，不进行复杂的测试。
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.env_discrete import DiscreteActionEnv
from envs.env_core import EnvCore
import config


def validate_action_space():
    """验证动作空间修改"""
    print("=" * 60)
    print("验证动作空间修改")
    print("=" * 60)

    env = DiscreteActionEnv()

    # 检查动作空间维度
    print(f"1. 智能体数量: {env.num_agent}")
    print(f"2. 信号观测维度: {env.signal_obs_dim}")
    print(f"3. 信号动作维度: {env.signal_action_dim}")

    # 检查动作空间类型
    if hasattr(env, 'action_space'):
        print(f"3. 动作空间类型: {type(env.action_space).__name__}")

        if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
            print(f"4. 动作空间维度数: {len(env.action_space.low)}")
            print(f"5. 动作空间范围:")
            print(f"   - offload_decision: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
            print(f"   - semantic_factor: [{env.action_space.low[1]}, {env.action_space.high[1]}]")
            print(f"   - resource_allocation: [{env.action_space.low[2]}, {env.action_space.high[2]}]")
            print(f"   - bandwidth_weight: [{env.action_space.low[3]}, {env.action_space.high[3]}]")

            # 验证带宽权重范围
            if env.action_space.high[3] == 3:
                print("   ✅ 带宽权重范围正确 (0-3)")
            else:
                print(f"   ❌ 带宽权重范围不正确: 期望0-3, 实际{env.action_space.low[3]}-{env.action_space.high[3]}")

    print()


def validate_dwdna_allocation():
    """验证DW-DNA带宽分配"""
    print("=" * 60)
    print("验证DW-DNA带宽分配")
    print("=" * 60)

    env_core = EnvCore()

    # 测试用例1: 基本分配
    print("测试用例1: 基本分配")
    bandwidth_weights = [1, 2, 3, 1, 2]
    offload_decisions = [1, 1, 1, 1, 1]

    bandwidths = env_core._allocate_bandwidth_dwdna(bandwidth_weights, offload_decisions)

    print(f"带宽权重: {bandwidth_weights}")
    print(f"卸载决策: {offload_decisions}")
    print(f"分配的带宽: {bandwidths}")
    print(f"总分配带宽: {sum(bandwidths):.0f} Hz")
    print(f"总可用带宽: {env_core.transmission_bandwidth:.0f} Hz")

    if sum(bandwidths) <= env_core.transmission_bandwidth:
        print("✅ 总分配带宽不超过总可用带宽")
    else:
        print("❌ 总分配带宽超过总可用带宽")

    # 检查RB量化
    rb_size = 180 * env_core.kHz
    all_rb_multiple = all(bw % rb_size == 0 for bw in bandwidths if bw > 0)
    if all_rb_multiple:
        print("✅ 所有带宽分配都是180kHz的倍数")
    else:
        print("❌ 有带宽分配不是180kHz的倍数")

    # 测试用例2: 部分卸载
    print("\n测试用例2: 部分卸载")
    bandwidth_weights = [1, 2, 0, 3, 1]
    offload_decisions = [1, 1, 0, 1, 0]

    bandwidths = env_core._allocate_bandwidth_dwdna(bandwidth_weights, offload_decisions)

    print(f"带宽权重: {bandwidth_weights}")
    print(f"卸载决策: {offload_decisions}")
    print(f"分配的带宽: {bandwidths}")

    if bandwidths[2] == 0 and bandwidths[4] == 0:
        print("✅ 未卸载用户带宽为0")
    else:
        print("❌ 未卸载用户带宽不为0")

    print()


def validate_environment_step():
    """验证环境步骤"""
    print("=" * 60)
    print("验证环境步骤")
    print("=" * 60)

    env = DiscreteActionEnv()

    # 重置环境
    obs = env.reset()
    print(f"1. 环境重置成功")
    print(f"   观测形状: {obs.shape}")
    print(f"   智能体数量: {env.num_agent}")
    print(f"   观测维度: {env.signal_obs_dim}")

    # 创建测试动作
    num_agents = env.num_agent
    test_actions = []

    for i in range(num_agents):
        # 创建有意义的测试动作
        action = [
            1,  # offload_decision: 卸载
            3,  # semantic_factor: 中等语义因子
            5,  # resource_allocation: 中等资源分配
            2   # bandwidth_weight: 中等带宽权重
        ]
        test_actions.append(action)

    test_actions = np.array(test_actions)
    print(f"\n2. 创建测试动作")
    print(f"   动作形状: {test_actions.shape}")
    print(f"   动作示例: {test_actions[0]}")

    # 执行步骤
    episode = 0
    step = 0
    obs, rewards, dones, infos = env.step(test_actions, episode, step)

    print(f"\n3. 环境步骤执行成功")
    print(f"   新观测形状: {obs.shape}")
    print(f"   奖励形状: {rewards.shape}")
    print(f"   奖励值: {rewards}")
    print(f"   完成标志: {dones}")
    print(f"   信息类型: {type(infos)}")

    print()


def validate_config_parameters():
    """验证配置参数"""
    print("=" * 60)
    print("验证配置参数")
    print("=" * 60)

    parser = config.get_config()

    print("1. 配置解析器创建成功")

    # 检查关键参数
    test_args = [
        '--algorithm_name', 'rmappo',
        '--env_name', 'MPE',
        '--num_agents', '5',
        '--episode_length', '200',
        '--num_env_steps', '1000000'
    ]

    try:
        args = parser.parse_args(test_args)

        print("2. 关键参数解析成功:")
        print(f"   算法名称: {args.algorithm_name}")
        print(f"   环境名称: {args.env_name}")
        print(f"   智能体数量: {args.num_agents}")
        print(f"   回合长度: {args.episode_length}")
        print(f"   环境步数: {args.num_env_steps}")

        # 检查批量实验参数
        print("\n3. 检查批量实验参数:")
        if hasattr(args, 'batch_mode'):
            print(f"   批量模式: {args.batch_mode}")
        else:
            print("   批量模式参数未定义")

        if hasattr(args, 'grid_size'):
            print(f"   网格大小: {args.grid_size}")
        else:
            print("   网格大小参数未定义")

    except SystemExit:
        print("❌ 参数解析失败")

    print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("SA-MAPPO 快速验证")
    print("=" * 60 + "\n")

    # 运行所有验证
    validate_action_space()
    validate_dwdna_allocation()
    validate_environment_step()
    validate_config_parameters()

    print("=" * 60)
    print("验证完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()