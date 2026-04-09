"""
遗传算法 (Genetic Algorithm) for Semantic-Aware Distributed Optimization

优化目标：在满足时延约束的前提下最小化总能耗
决策变量：5维 [offload, semantic, mec_resource, tx_power, bw_weight]

Author: GA Implementation
Date: 2026
"""

import numpy as np
import random
import csv
import copy
from env import ENV

# 动作空间离散集合
semantic_factor_set = np.round(np.linspace(0.3, 1.0, 8), 1)
resource_allocation_set = np.round(np.linspace(0.1, 1.0, 10), 1)
transmission_power_set = np.round(np.linspace(0.1, 0.5, 5), 1)
bw_weight_set = np.array([0, 1, 2, 3])  # 带宽权重离散集合


def initialize_population(pop_size, K, env, observation, max_attempts=10000):
    """
    初始化种群，生成合法的个体

    参数:
        pop_size: 种群大小
        K: UE数量
        env: 环境实例
        observation: 观察值列表
        max_attempts: 最大尝试次数

    返回:
        population: 个体列表，每个个体为 (offload, semantic, resource, power, bw_weight)
    """
    population = []
    attempts = 0

    # 策略1: 全部本地处理 (最安全)
    offload_decision = np.zeros(K, dtype=int)
    semantic_factor = np.ones(K)
    resource_allocation = np.zeros(K)
    transmission_power = np.full(K, 0.1)
    bw_weight = np.zeros(K, dtype=int)

    _, delay_penalty = env.compute_energy_and_delay(
        offload_decision, semantic_factor, resource_allocation,
        transmission_power, bw_weight, observation)

    if delay_penalty == 0:
        population.append((
            offload_decision.copy(),
            semantic_factor.copy(),
            resource_allocation.copy(),
            transmission_power.copy(),
            bw_weight.copy()
        ))

    # 策略2: 随机初始化
    while len(population) < pop_size and attempts < max_attempts:
        attempts += 1

        # 随机生成卸载决策
        offload_decision = np.random.choice([0, 1], size=K)

        # 生成语义提取因子 (卸载决策=1时有效)
        semantic_factor = np.random.choice(semantic_factor_set, size=K)
        # 不卸载时设为1
        semantic_factor = np.where(offload_decision == 1, semantic_factor, 1.0)

        # 生成MEC资源分配 (卸载决策=1时有效)
        resource_allocation = np.random.choice(resource_allocation_set, size=K)
        resource_allocation = resource_allocation * offload_decision
        if np.sum(resource_allocation) > 0:
            resource_allocation = resource_allocation / np.sum(resource_allocation)

        # 生成传输功率
        transmission_power = np.random.choice(transmission_power_set, size=K)

        # 生成带宽权重 (卸载决策=1时为1,2,3; 不卸载时为0)
        bw_weight = np.zeros(K, dtype=int)
        offload_indices = np.where(offload_decision == 1)[0]
        if len(offload_indices) > 0:
            bw_weight[offload_indices] = np.random.choice(bw_weight_set[1:], size=len(offload_indices))

        # 验证约束
        _, delay_penalty = env.compute_energy_and_delay(
            offload_decision, semantic_factor, resource_allocation,
            transmission_power, bw_weight, observation)

        if delay_penalty == 0:
            population.append((
                offload_decision.copy(),
                semantic_factor.copy(),
                resource_allocation.copy(),
                transmission_power.copy(),
                bw_weight.copy()
            ))

        # 生成语义提取因子 (卸载决策=1时有效)
        semantic_factor = np.random.choice(semantic_factor_set, size=K)
        # 不卸载时设为1
        semantic_factor = np.where(offload_decision == 1, semantic_factor, 1.0)

        # 生成MEC资源分配 (卸载决策=1时有效)
        resource_allocation = np.random.choice(resource_allocation_set, size=K)
        resource_allocation = resource_allocation * offload_decision
        if np.sum(resource_allocation) > 0:
            resource_allocation = resource_allocation / np.sum(resource_allocation)

        # 生成传输功率
        transmission_power = np.random.choice(transmission_power_set, size=K)

        # 生成带宽权重 (卸载决策=1时为1,2,3; 不卸载时为0)
        bw_weight = np.zeros(K, dtype=int)
        offload_indices = np.where(offload_decision == 1)[0]
        if len(offload_indices) > 0:
            bw_weight[offload_indices] = np.random.choice(bw_weight_set[1:], size=len(offload_indices))

        # 验证约束
        _, delay_penalty = env.compute_energy_and_delay(
            offload_decision, semantic_factor, resource_allocation,
            transmission_power, bw_weight, observation)

        if delay_penalty == 0:
            population.append((
                offload_decision.copy(),
                semantic_factor.copy(),
                resource_allocation.copy(),
                transmission_power.copy(),
                bw_weight.copy()
            ))

    if len(population) == 0:
        raise ValueError("无法生成满足约束的初始种群！")

    return population


def fitness(individual, env, observation, generation=0, max_generations=10000):
    """
    计算适应度函数

    参数:
        individual: 个体 (offload, semantic, resource, power, bw_weight)
        env: 环境实例
        observation: 观察值列表
        generation: 当前代数
        max_generations: 总代数

    返回:
        fitness: 适应度值 (负的总能耗，违反约束时返回极负值)
    """
    offload, semantic, resource, power, bw_weight = individual

    total_energy, delay_penalty = env.compute_energy_and_delay(
        offload, semantic, resource, power, bw_weight, observation)

    # 违反时延约束的惩罚
    if delay_penalty > 0:
        return -1e10

    # 检查数值稳定性
    if np.any(np.isnan(total_energy)) or np.any(np.isinf(total_energy)):
        print(f"Warning: Total energy contains NaN or inf! Energy: {total_energy}")
        return -1e10

    # 适应度 = 负的总能耗 (最小化能耗)
    return -total_energy


def selection(population, fitness_values):
    """
    轮盘赌选择

    参数:
        population: 种群列表
        fitness_values: 适应度值数组

    返回:
        selected_population: 选中的种群
    """
    # 适应度平移到正区间
    min_fitness = np.min(fitness_values)
    if min_fitness < 0:
        weights = fitness_values - min_fitness + 1
    else:
        weights = fitness_values + 1

    # 选择
    selected = random.choices(population, weights=weights, k=len(population))
    return selected


def crossover(parent1, parent2, crossover_rate=0.8):
    """
    单点交叉

    参数:
        parent1: 父个体1
        parent2: 父个体2
        crossover_rate: 交叉概率

    返回:
        child1, child2: 子个体
    """
    if np.random.rand() < crossover_rate:
        # 随机选择交叉点
        K = len(parent1[0])
        point = np.random.randint(1, K)

        child1 = (
            np.concatenate((parent1[0][:point], parent2[0][point:])),
            np.concatenate((parent1[1][:point], parent2[1][point:])),
            np.concatenate((parent1[2][:point], parent2[2][point:])),
            np.concatenate((parent1[3][:point], parent2[3][point:])),
            np.concatenate((parent1[4][:point], parent2[4][point:]))
        )
        child2 = (
            np.concatenate((parent2[0][:point], parent1[0][point:])),
            np.concatenate((parent2[1][:point], parent1[1][point:])),
            np.concatenate((parent2[2][:point], parent1[2][point:])),
            np.concatenate((parent2[3][:point], parent1[3][point:])),
            np.concatenate((parent2[4][:point], parent1[4][point:]))
        )
        return child1, child2

    return parent1, parent2


def mutation(individual, env, observation, mutation_rate=0.3, max_attempts=100):
    """
    变异操作 (保持约束合法)

    参数:
        individual: 个体
        env: 环境实例
        observation: 观察值列表
        mutation_rate: 变异概率
        max_attempts: 最大尝试次数

    返回:
        individual: 变异后的个体
    """
    offload, semantic, resource, power, bw_weight = individual
    K = len(offload)

    for _ in range(max_attempts):
        # 变异卸载决策
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(K)
            offload[idx] = 1 - offload[idx]

            # 同步更新语义因子和带宽权重
            if offload[idx] == 0:
                semantic[idx] = 1.0
                resource[idx] = 0
                bw_weight[idx] = 0
            else:
                semantic[idx] = np.random.choice(semantic_factor_set)
                bw_weight[idx] = np.random.choice(bw_weight_set[1:])

        # 变异语义提取因子
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(K)
            if offload[idx] == 1:
                current_val = semantic[idx]
                candidates = semantic_factor_set[semantic_factor_set != current_val]
                semantic[idx] = np.random.choice(candidates)

        # 变异MEC资源分配
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(K)
            if offload[idx] == 1:
                current_val = resource[idx]
                candidates = resource_allocation_set[resource_allocation_set != current_val]
                if len(candidates) > 0:
                    resource[idx] = np.random.choice(candidates)

        # 变异传输功率
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(K)
            current_val = power[idx]
            candidates = transmission_power_set[transmission_power_set != current_val]
            power[idx] = np.random.choice(candidates)

        # 变异带宽权重
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(K)
            if offload[idx] == 1:
                current_val = bw_weight[idx]
                candidates = bw_weight_set[bw_weight_set != current_val]
                bw_weight[idx] = np.random.choice(candidates)

        # 重新归一化资源分配
        resource = resource * offload
        if np.sum(resource) > 0:
            resource = resource / np.sum(resource)

        # 验证约束
        _, delay_penalty = env.compute_energy_and_delay(
            offload, semantic, resource, power, bw_weight, observation)

        if delay_penalty == 0:
            return offload, semantic, resource, power, bw_weight

    # 如果无法找到合法变异，返回原始个体
    return individual


def genetic_algorithm(env, observation, a, pop_size=50, generations=5000,
                     early_stop_threshold=0.001, patience=500, verbose=True):
    """
    遗传算法主函数

    参数:
        env: 环境实例
        observation: 观察值列表
        a: 实验编号 (用于保存结果)
        pop_size: 种群大小
        generations: 最大代数
        early_stop_threshold: 早停阈值
        patience: 早停耐心值
        verbose: 是否打印详细信息

    返回:
        best_solution: 最优解
        best_energy_per_generation: 每代最优能耗
    """
    K = env.UEs

    # 初始化种群
    population = initialize_population(pop_size, K, env, observation)

    best_solution = None
    best_fitness = float('-inf')
    best_energy_per_generation = []
    no_improve_counter = 0

    for gen in range(generations):
        # 计算适应度
        fitness_values = np.array([
            fitness(ind, env, observation, gen, generations)
            for ind in population
        ])

        # 选择
        selected_population = selection(population, fitness_values)

        # 生成新种群
        new_population = []

        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[min(i + 1, len(selected_population) - 1)]

            # 交叉
            child1, child2 = crossover(parent1, parent2)

            # 变异
            new_population.append(
                mutation(child1, env, observation, mutation_rate=0.3)
            )
            if len(new_population) < pop_size:
                new_population.append(
                    mutation(child2, env, observation, mutation_rate=0.3)
                )

        # 保持种群大小
        population = new_population[:pop_size]

        # 记录最优解
        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_solution = copy.deepcopy(population[current_best_idx])
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        best_energy_per_generation.append(-best_fitness)

        if verbose and gen % 100 == 0:
            print(f'Generation {gen + 1}, Best Energy: {-best_fitness:.6f}')

        # 早停检查
        if no_improve_counter >= patience:
            if len(best_energy_per_generation) >= patience:
                recent_energy = np.mean(best_energy_per_generation[-patience:])
                if abs(best_energy_per_generation[-1] - recent_energy) < early_stop_threshold:
                    if verbose:
                        print(f'Early stopping at generation {gen + 1}. Best Energy: {-best_fitness:.6f}')
                    break

    final_energy = -best_fitness
    if verbose:
        print(f'Optimal Solution Found with Total Energy: {final_energy:.6f}')

    return best_solution, best_energy_per_generation


def run_single_experiment(data_size, num_UEs, bandwidth, mec_capacity,
                          min_semantic_factor, a=1, pop_size=50,
                          generations=5000, verbose=True):
    """
    运行单次实验

    参数:
        data_size: 数据大小 (KB)
        num_UEs: 用户数量
        bandwidth: 总带宽 (kHz)
        mec_capacity: MEC计算能力 (Giga Cycles/s)
        min_semantic_factor: 最小语义提取因子
        a: 实验编号
        pop_size: 种群大小
        generations: 进化代数
        verbose: 是否打印详细信息

    返回:
        result: 结果字典
    """
    # 创建环境
    env = ENV(
        UEs=num_UEs,
        MECs=1,
        k=100,
        total_bandwidth=bandwidth,
        mec_capacity=mec_capacity,
        min_semantic_factor=min_semantic_factor
    )

    # 计算任务大小倍数 (data_size KB / 256 KB)
    task_multiplier = data_size / 256.0
    observation = env.reset(task_multiplier)

    # 运行遗传算法
    best_solution, energy_history = genetic_algorithm(
        env, observation, a, pop_size, generations, verbose=verbose
    )

    # 计算最终能耗和时延
    offload, semantic, resource, power, bw_weight = best_solution
    final_energy, final_penalty = env.compute_energy_and_delay(
        offload, semantic, resource, power, bw_weight, observation
    )

    # 统计结果
    offload_count = np.sum(offload)
    local_count = len(offload) - offload_count

    result = {
        'data_size': data_size,
        'num_UEs': num_UEs,
        'bandwidth': bandwidth,
        'mec_capacity': mec_capacity,
        'min_semantic_factor': min_semantic_factor,
        'total_energy': final_energy,
        'delay_penalty': final_penalty,
        'offload_count': int(offload_count),
        'local_count': int(local_count),
        'best_generation': len(energy_history)
    }

    return result, best_solution, energy_history


if __name__ == '__main__':
    # 单次测试运行
    result, solution, history = run_single_experiment(
        data_size=256,
        num_UEs=5,
        bandwidth=1000,
        mec_capacity=20.0,
        min_semantic_factor=0.3,
        a=1,
        pop_size=30,
        generations=1000,
        verbose=True
    )

    print("\n=== Result ===")
    print(f"Total Energy: {result['total_energy']:.6f}")
    print(f"Offloaded: {result['offload_count']}, Local: {result['local_count']}")
    print(f"Delay Penalty: {result['delay_penalty']:.6f}")