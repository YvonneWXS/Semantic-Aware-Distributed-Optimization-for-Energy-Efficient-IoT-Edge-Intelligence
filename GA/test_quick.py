"""
快速测试脚本 - 验证代码修改是否正确
"""
import numpy as np
from env import ENV
from genetic_algorithm import genetic_algorithm

def test_environment_params():
    """测试环境参数是否正确设置"""
    print("测试环境参数...")
    env = ENV(UEs=3, MECs=1, k=10)  # 使用少量用户进行测试

    # 验证参数
    assert env.transmission_bandwidth == 2 * env.mHz, f"带宽应为2MHz, 实际: {env.transmission_bandwidth/env.mHz}MHz"
    assert env.transmission_power == 0.1, f"发射功率应为0.1W, 实际: {env.transmission_power}W"
    assert env.MEC_f == 20 * env.GHz, f"MEC计算能力应为20GHz, 实际: {env.MEC_f/env.GHz}GHz"

    # 验证语义阈值
    assert env.actions2[1][1] >= 0.5, f"语义因子应>=0.5, 实际: {env.actions2[1][1]}"

    print("环境参数测试通过")

def test_reset_function():
    """测试重置函数"""
    print("\n测试重置函数...")
    env = ENV(UEs=3, MECs=1, k=10)

    # 测试不同数据大小
    test_sizes = [250, 500, 1000]
    for size in test_sizes:
        obs = env.reset(size)
        assert len(obs) == 3, f"应有3个用户的观察值, 实际: {len(obs)}"

        # 验证任务大小
        for i, ob in enumerate(obs):
            task_size = ob[0]
            expected = size * env.KB
            assert abs(task_size - expected) < 1e-6, f"用户{i}任务大小应为{expected}, 实际: {task_size}"

            # 验证最大延迟
            max_delay = ob[2]
            assert abs(max_delay - 0.1) < 1e-6, f"最大延迟应为0.1s, 实际: {max_delay}"

    print("✓ 重置函数测试通过")

def test_genetic_algorithm_quick():
    """快速测试遗传算法（小规模）"""
    print("\n快速测试遗传算法...")

    env = ENV(UEs=5, MECs=1, k=10)  # 使用少量用户
    data_size_kb = 500  # 中等数据大小
    observation = env.reset(data_size_kb)

    print(f"运行遗传算法: UEs=5, DataSize={data_size_kb}KB")
    try:
        # 运行小规模遗传算法
        best_solution, energy_history = genetic_algorithm(
            env, observation, data_size_kb,
            pop_size=20,  # 小种群
            generations=50,  # 少代数
            early_stop_threshold=0.1,
            patience=10,
            output_dir="test_output"
        )

        # 验证解
        offload_decision, resource_allocation, transmission_power = best_solution

        print(f"最优解信息:")
        print(f"  卸载决策: {offload_decision}")
        print(f"  资源分配: {resource_allocation}")
        print(f"  发射功率: {transmission_power}")
        print(f"  能量历史长度: {len(energy_history)}")

        # 验证约束
        final_energy, final_penalty = env.compute_energy_and_delay(*best_solution, observation)
        print(f"  总能量: {final_energy:.6f} J")
        print(f"  延迟惩罚: {final_penalty}")

        # 验证发射功率约束
        assert np.all(transmission_power == 0.1), f"发射功率应全为0.1W, 实际: {transmission_power}"

        print("✓ 遗传算法快速测试通过")

    except Exception as e:
        print(f"遗传算法测试失败: {e}")
        raise

def main():
    print("开始快速测试...")
    print("="*50)

    try:
        test_environment_params()
        test_reset_function()
        test_genetic_algorithm_quick()

        print("\n" + "="*50)
        print("所有测试通过！代码修改正确。")
        print("可以运行 experiment_runner.py 进行完整实验。")

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()