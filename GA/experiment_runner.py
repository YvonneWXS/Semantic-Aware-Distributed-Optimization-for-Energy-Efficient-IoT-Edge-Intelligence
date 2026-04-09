import numpy as np
import csv
import os
import shutil
from env import ENV
from genetic_algorithm import genetic_algorithm

def run_experiment_for_data_size(data_size_kb, result_dir="result"):
    """
    运行单个数据大小的实验
    """
    # 创建实验文件夹
    exp_dir = os.path.join(result_dir, f"DataSize_{data_size_kb}KB")
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"开始运行实验: DataSize = {data_size_kb} KB")
    print(f"结果将保存到: {exp_dir}")
    print(f"{'='*60}")

    # 初始化环境
    env = ENV(UEs=20, MECs=1, k=100)
    observation = env.reset(data_size_kb)

    # 运行遗传算法
    best_solution, energy_history = genetic_algorithm(
        env, observation, data_size_kb,
        pop_size=100, generations=10000,
        early_stop_threshold=0.01, patience=2000,
        output_dir=exp_dir
    )

    # 计算最终结果
    final_energy, final_penalty = env.compute_energy_and_delay(*best_solution, observation)

    # 验证约束
    if final_penalty > 0:
        print(f"警告: 最终解违反了延迟约束! 惩罚值: {final_penalty}")
    else:
        print(f"最终解满足所有约束。")

    # 保存详细结果
    save_results(exp_dir, data_size_kb, best_solution, energy_history,
                 final_energy, final_penalty, env, observation)

    return final_energy, final_penalty

def save_results(exp_dir, data_size_kb, best_solution, energy_history,
                 final_energy, final_penalty, env, observation):
    """
    保存实验结果
    """
    offload_decision, resource_allocation, transmission_power = best_solution

    # 1. 保存参数设置
    params_file = os.path.join(exp_dir, "experiment_parameters.txt")
    with open(params_file, 'w') as f:
        f.write("实验参数设置:\n")
        f.write("="*50 + "\n")
        f.write(f"数据大小 (DataSize): {data_size_kb} KB\n")
        f.write(f"用户数量 (UEs): 20\n")
        f.write(f"总带宽 (B): 2000 kHz (2 MHz)\n")
        f.write(f"MEC服务器计算能力 (Fmec): 20.0 GHz\n")
        f.write(f"任务数据量 (Dn): {data_size_kb} KB\n")
        f.write(f"最小语义提取因子 (beta_min): 0.5\n")
        f.write(f"最大发射功率限制 (Pmax): 0.5 W\n")
        f.write(f"最大容忍时延 (Tmax): 100 ms\n")
        f.write(f"结果文件夹: {exp_dir}\n")

    # 2. 保存最终解
    solution_file = os.path.join(exp_dir, "best_solution.csv")
    with open(solution_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["UE_Index", "Offload_Decision", "Resource_Allocation", "Transmission_Power", "Channel_Gain", "Local_Comp"])

        for i in range(len(offload_decision)):
            writer.writerow([
                i,
                offload_decision[i],
                resource_allocation[i],
                transmission_power[i],
                env.UE_params[i]['channel_gain'],
                env.UE_params[i]['local_comp']
            ])

    # 3. 保存能量历史
    energy_file = os.path.join(exp_dir, "energy_history.csv")
    with open(energy_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Generation", "Best_Energy"])
        for gen, energy in enumerate(energy_history):
            writer.writerow([gen + 1, energy])

    # 4. 保存汇总结果
    summary_file = os.path.join(exp_dir, "summary_results.txt")
    with open(summary_file, 'w') as f:
        f.write("实验汇总结果:\n")
        f.write("="*50 + "\n")
        f.write(f"数据大小: {data_size_kb} KB\n")
        f.write(f"总能量消耗: {final_energy:.6f} J\n")
        f.write(f"延迟惩罚: {final_penalty}\n")
        f.write(f"卸载决策统计: 卸载 {np.sum(offload_decision)} / {len(offload_decision)} 用户\n")
        f.write(f"资源分配总和: {np.sum(resource_allocation):.4f}\n")
        f.write(f"平均传输功率: {np.mean(transmission_power):.4f} W\n")

        # 计算能量分布
        local_energy_total = 0
        upload_energy_total = 0
        SEtask_energy_total = 0

        for i in range(len(offload_decision)):
            task_size, computing_density, max_delay, local_energy = observation[i]
            local_comp = env.UE_params[i]['local_comp']
            channel_gain = env.UE_params[i]['channel_gain']

            if offload_decision[i] == 0:
                local_energy_total += local_energy
            else:
                # 计算上传能量
                offload_num = np.sum(offload_decision)
                W = env.transmission_bandwidth / offload_num if offload_num > 0 else 0
                uplink_rate = W * np.log2(1 + transmission_power[i] * channel_gain / (W * env.noise_power))
                upload_energy = transmission_power[i] * task_size / uplink_rate
                upload_energy_total += upload_energy

                # 计算语义提取任务能量
                SEtask_energy = env.κ * env.alpha * (task_size ** env.r) * (1 ** (-env.beta) - 1) * local_comp**2
                SEtask_energy_total += SEtask_energy

        f.write(f"\n能量分布:\n")
        f.write(f"  本地处理能量: {local_energy_total:.6f} J\n")
        f.write(f"  上传能量: {upload_energy_total:.6f} J\n")
        f.write(f"  语义提取任务能量: {SEtask_energy_total:.6f} J\n")

        f.write(f"\n解验证:\n")
        if final_penalty == 0:
            f.write(f"  ✓ 满足延迟约束 (Tmax = 100ms)\n")
        else:
            f.write(f"  ✗ 违反延迟约束, 惩罚值: {final_penalty}\n")

        if np.sum(resource_allocation) <= 1 + 1e-6:
            f.write(f"  ✓ 满足资源约束 (总和 <= 1)\n")
        else:
            f.write(f"  ✗ 违反资源约束, 总和: {np.sum(resource_allocation):.4f}\n")

        if np.all(transmission_power <= 0.5 + 1e-6):
            f.write(f"  ✓ 满足功率约束 (<= 0.5W)\n")
        else:
            f.write(f"  ✗ 违反功率约束, 最大值: {np.max(transmission_power):.4f}W\n")

    print(f"实验结果已保存到: {exp_dir}")

def main():
    """主函数：运行所有数据大小实验"""
    # 数据大小列表
    data_sizes = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]

    print("开始运行多数据大小实验")
    print(f"数据大小范围: {data_sizes} KB")
    print(f"结果将保存到: GA/result/ 文件夹")
    print(f"{'='*60}")

    results = []

    for data_size in data_sizes:
        try:
            final_energy, final_penalty = run_experiment_for_data_size(data_size)
            results.append({
                'data_size': data_size,
                'energy': final_energy,
                'penalty': final_penalty,
                'success': final_penalty == 0
            })
        except Exception as e:
            print(f"运行数据大小 {data_size} KB 时出错: {e}")
            results.append({
                'data_size': data_size,
                'energy': None,
                'penalty': None,
                'success': False,
                'error': str(e)
            })

    # 保存总结果汇总
    save_final_summary(results)

def save_final_summary(results):
    """保存所有实验的总结果汇总"""
    summary_file = os.path.join("result", "all_experiments_summary.csv")

    with open(summary_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["DataSize_KB", "Total_Energy_J", "Delay_Penalty", "Success", "Result_Folder"])

        for result in results:
            if result['success']:
                folder_name = f"DataSize_{result['data_size']}KB"
                writer.writerow([
                    result['data_size'],
                    result['energy'],
                    result['penalty'],
                    "YES" if result['success'] else "NO",
                    folder_name
                ])
            else:
                writer.writerow([
                    result['data_size'],
                    "N/A" if result['energy'] is None else result['energy'],
                    "N/A" if result['penalty'] is None else result['penalty'],
                    "NO",
                    "FAILED"
                ])

    print(f"\n所有实验汇总已保存到: {summary_file}")

    # 打印汇总报告
    print(f"\n{'='*60}")
    print("实验汇总报告:")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r['success'])
    print(f"成功完成: {successful}/{len(results)} 个实验")

    if successful > 0:
        print("\n成功实验的能量消耗:")
        for result in results:
            if result['success']:
                print(f"  DataSize {result['data_size']} KB: {result['energy']:.6f} J")

if __name__ == '__main__':
    main()