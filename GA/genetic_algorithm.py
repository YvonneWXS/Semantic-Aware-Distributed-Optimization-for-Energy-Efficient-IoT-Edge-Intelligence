import numpy as np
import random
import csv
import os
from env import ENV

semantic_factor_set = np.round(np.linspace(0.5, 1.0, 6), 1)  # 从0.5开始，6个值
resource_allocation_set = np.round(np.linspace(0.1, 1.0, 10), 1)
transmission_power_set = np.round(np.linspace(0.1, 0.5, 5), 1)  # 传输功率范围[0.1, 0.5]W，5个离散值

def initialize_population(pop_size, K, env, observation, max_attempts=100):
    population = []
    while len(population) < pop_size:
        for _ in range(max_attempts):
            offload_decision = np.random.choice([0, 1], size=K)
            resource_allocation = np.random.choice(resource_allocation_set, size=K)
            transmission_power = np.random.choice(transmission_power_set, size=K)
            resource_allocation = resource_allocation * offload_decision
            if np.sum(resource_allocation) > 0:
                resource_allocation = resource_allocation / np.sum(resource_allocation)

            total_energy, delay_penalty = env.compute_energy_and_delay(
                offload_decision, resource_allocation, transmission_power, observation)

            if delay_penalty == 0:
                population.append((offload_decision, resource_allocation, transmission_power))
                break
    return population

def fitness(offload_decision, resource_allocation, transmission_power, env, observation, generation, max_generations):
    total_energy, delay_penalty = env.compute_energy_and_delay(
        offload_decision, resource_allocation, transmission_power, observation)

    if delay_penalty > 0:
        return -1e10  # 违反约束的劣质解

    if np.any(np.isnan(total_energy)) or np.any(np.isinf(total_energy)):
        print("Total energy contains NaN or inf!")
        print(total_energy)

    return -total_energy

def selection(population, fitness_values):
    weights = fitness_values - min(fitness_values) + 1
    selected = random.choices(population, weights=weights, k=len(population))
    return selected

def crossover(parent1, parent2, crossover_rate=0.99):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1[0]))

        child1 = (
            np.concatenate((parent1[0][:point], parent2[0][point:])),
            np.concatenate((parent1[1][:point], parent2[1][point:])),
            np.concatenate((parent1[2][:point], parent2[2][point:]))
        )
        child2 = (
            np.concatenate((parent2[0][:point], parent1[0][point:])),
            np.concatenate((parent2[1][:point], parent1[1][point:])),
            np.concatenate((parent2[2][:point], parent1[2][point:]))
        )
        return child1, child2
    return parent1, parent2

def mutation(individual, env, observation, mutation_rate=0.9, min_resource=1e-6, max_attempts=50):
    mu, f_mec, p_tx = individual
    for _ in range(max_attempts):
        # 变异offload_decision
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(len(mu))
            mu[idx] = 1 - mu[idx]

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

        f_mec = f_mec * mu
        if np.sum(f_mec) > 0:
            f_mec = f_mec / np.sum(f_mec)

        _, delay_penalty = env.compute_energy_and_delay(mu, f_mec, p_tx, observation)
        if delay_penalty == 0:
            return mu, f_mec, p_tx

    return individual

def genetic_algorithm(env, observation, data_size_kb, pop_size=100, generations=10000, early_stop_threshold=0.01, patience=2000, output_dir="."):
    population = initialize_population(pop_size, env.UEs, env, observation)
    best_solution = None
    best_fitness = float('-inf')
    best_energy_per_generation = []
    no_improve_counter = 0

    for gen in range(generations):
        fitness_values = np.array([fitness(*ind, env, observation, gen, generations) for ind in population])
        selected_population = selection(population, fitness_values)
        new_population = []

        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[min(i + 1, len(selected_population) - 1)]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([
                mutation(child1, env, observation),
                mutation(child2, env, observation)
            ])

        population = new_population
        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_solution = population[current_best_idx]
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        best_energy_per_generation.append(-best_fitness)
        print(f'Generation {gen + 1}, Best Energy: {-best_fitness}')

        if no_improve_counter >= patience and abs(
            best_energy_per_generation[-1] - best_energy_per_generation[-patience]
        ) < early_stop_threshold:
            print(f'Early stopping at generation {gen + 1}. Best Energy: {-best_fitness}')
            break

    total_energy = -best_fitness
    print(f'Optimal Solution Found with Total Energy: {total_energy}')

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    csv_filename = os.path.join(output_dir, f"GA_energy_log_{data_size_kb}KB.csv")
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Generation", "Best Energy"])
        for gen, energy in enumerate(best_energy_per_generation):
            writer.writerow([gen + 1, energy])

    print(f'Energy log saved to {csv_filename}')

    return best_solution, best_energy_per_generation

if __name__ == '__main__':
    # 测试用的小规模运行
    env = ENV(UEs=20, MECs=1, k=100)
    data_size_kb = 1000  # 测试用
    observation = env.reset(data_size_kb)
    best_solution, energy_history = genetic_algorithm(env, observation, data_size_kb)
    final_energy, final_penalty = env.compute_energy_and_delay(*best_solution, observation)
    assert final_penalty == 0, f"Final solution violates delay constraint! Penalty: {final_penalty}"
