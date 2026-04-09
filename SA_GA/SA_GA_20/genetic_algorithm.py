import numpy as np
import random
import csv
from env import ENV
import pandas as pd

semantic_factor_set = np.round(np.linspace(0.3, 1.0, 8), 1)
resource_allocation_set = np.round(np.linspace(0.1, 1.0, 10), 1)
transmission_power_set = np.round(np.linspace(0.1, 0.5, 5), 1)

def initialize_population(pop_size, K ,env, observation, max_attempts=100):
    population = []
    while len(population) < pop_size:
        for _ in range(max_attempts):
            offload_decision = np.random.choice([0, 1], size=K)
            semantic_factor = np.random.choice(semantic_factor_set, size=K)
            resource_allocation = np.random.choice(resource_allocation_set, size=K)
            transmission_power = np.random.choice(transmission_power_set, size=K)
            resource_allocation = resource_allocation * offload_decision
            if np.sum(resource_allocation) > 1:
                resource_allocation /= np.sum(resource_allocation)  # 归一化资源分配

            total_energy, delay_penalty = env.compute_energy_and_delay(offload_decision, semantic_factor,resource_allocation, transmission_power, observation)
            if delay_penalty == 0:
                population.append((offload_decision, semantic_factor, resource_allocation,transmission_power))
                break
    return population

def fitness(offload_decision,semantic_factor, resource_allocation, transmission_power, env, observation, generation, max_generations):
    total_energy, delay_penalty = env.compute_energy_and_delay(offload_decision, semantic_factor, resource_allocation, transmission_power, observation)
    
    # 如果 delay_penalty 不为 0，说明不满足约束，直接返回非常差的适应度
    if delay_penalty > 0:
        return -1e10  # 非法解，极低适应度，淘汰

    # 只考虑合法解（delay_penalty == 0）的能耗
    if np.any(np.isnan(total_energy)) or np.any(np.isinf(total_energy)):
        print("Total energy contains NaN or inf!")
        print(total_energy)
    
    return -total_energy  

def selection(population, fitness_values):
    """轮盘赌选择或锦标赛选择"""
    selected = random.choices(population, weights=fitness_values - min(fitness_values) + 1, k=len(population))
    return selected

def crossover(parent1, parent2, crossover_rate=0.89):
    """单点交叉"""
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1[0]))
        child1 = (
            np.concatenate((parent1[0][:point], parent2[0][point:])),
            np.concatenate((parent1[1][:point], parent2[1][point:])),
            np.concatenate((parent1[2][:point], parent2[2][point:])),
            np.concatenate((parent1[3][:point], parent2[3][point:])),
        )
        child2 = (
            np.concatenate((parent2[0][:point], parent1[0][point:])),
            np.concatenate((parent2[1][:point], parent1[1][point:])),
            np.concatenate((parent2[2][:point], parent1[2][point:])),
            np.concatenate((parent2[3][:point], parent1[3][point:])),
        )
        return child1, child2
    return parent1, parent2

def mutation(individual, env, observation, mutation_rate=0.9, min_resource=1e-6, max_attempts=50):
    mu, beta, f_mec, p_tx = individual
    for _ in range(max_attempts):
        # 变异offload_decision
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(len(mu))
            mu[idx] = 1 - mu[idx]

        # 变异semantic_factor，从离散集合随机选不同值
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(len(beta))
            current_val = beta[idx]
            # 从集合中过滤掉当前值后随机选一个
            candidates = semantic_factor_set[semantic_factor_set != current_val]
            beta[idx] = np.random.choice(candidates)

        # 变异resource_allocation，同样从离散集合选
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(len(f_mec))
            current_val = f_mec[idx]
            candidates = resource_allocation_set[resource_allocation_set != current_val]
            f_mec[idx] = np.random.choice(candidates)

        # 变异transmission_power，从集合选
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(len(p_tx))
            current_val = p_tx[idx]
            candidates = transmission_power_set[transmission_power_set != current_val]
            p_tx[idx] = np.random.choice(candidates)

        # 资源归一化
        f_mec = f_mec * mu
        if np.sum(f_mec) > 0:
            f_mec = f_mec / np.sum(f_mec)

        # 计算并检查约束
        _, delay_penalty = env.compute_energy_and_delay(mu, beta, f_mec, p_tx, observation)
        if delay_penalty == 0:
            return mu, beta, f_mec, p_tx
    # 超出 max_attempts 次都不合法，返回原个体
    return individual

def genetic_algorithm(env, observation, a,pop_size=100, generations=3000, early_stop_threshold=0.01, patience=500):
    population = initialize_population(pop_size, env.UEs, env, observation)
    best_solution = None
    best_fitness = float('-inf')
    best_energy_per_generation = []  # 记录每代的最佳能耗
    no_improve_counter = 0  # 记录连续未改进的代数

    for gen in range(generations):
        fitness_values = np.array([fitness(*ind, env, observation, gen, generations) for ind in population])
        selected_population = selection(population, fitness_values)
        new_population = []
        
        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[min(i+1, len(selected_population)-1)]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutation(child1, env, observation), mutation(child2, env, observation)])
        
        population = new_population
        current_best_idx = np.argmax(fitness_values)
        
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_solution = population[current_best_idx]
            no_improve_counter = 0  # 重新计数
        else:
            no_improve_counter += 1

        best_energy_per_generation.append(-best_fitness)
        print(f'Generation {gen+1}, Best Energy: {-best_fitness}')
        
        # 早停机制
        if no_improve_counter >= patience and abs(best_energy_per_generation[-1] - best_energy_per_generation[-patience]) < early_stop_threshold:
            print(f'Early stopping at generation {gen+1} due to minimal improvement. Best Energy: {-best_fitness}')
            break
    
    total_energy = -best_fitness
    print(f'Optimal Solution Found with Total Energy: {total_energy}')

    # # 保存能耗数据到 CSV
    # csv_filename = f"SA_GA_energy_log_{a}_UEs.csv"
    # with open(csv_filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Generation", "Best Energy"])
    #     for gen, energy in enumerate(best_energy_per_generation):
    #         writer.writerow([gen + 1, energy])
    
    # print(f'Energy log saved to {csv_filename}')

    return best_solution, total_energy

if __name__ == '__main__':
    energy_records = []
    # for a in range (5,31,5):
    a = 20
    env = ENV(UEs=a, MECs=1, k=100)  
    total_energy_list = []
    step_total_energy = []
    for step in range(0,50):
        observation = env.reset(step)
        best_solution, final_energy = genetic_algorithm(env, observation,a)
        step_total_energy.append(final_energy)
    
    step_average_energy = sum(step_total_energy) / len(step_total_energy)
    print ("step average energy:",step_average_energy)
    energy_records.append({'Number of UEs': a, 'Average Energy Consumption': step_average_energy})
        
    # 将数据转换成 DataFrame
    df = pd.DataFrame(energy_records)

    # 保存为 CSV 文件
    df.to_csv("GA_{}.csv".format(a), index=False)
    print("Results saved to GA_{}.csv".format(a))
    print(df)


