"""
批量实验运行入口

功能:
1. 根据配置运行多组实验
2. 自动保存结果到CSV文件
3. 支持单变量遍历实验
4. 实时显示进度

使用方式:
1. 在 config.py 中设置要遍历的变量和固定参数
2. 修改下方 VAR_TO_COMPARE 变量选择对比参数
3. 运行本脚本

Author: GA Implementation
Date: 2026
"""

import os
import csv
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any

from genetic_algorithm import run_single_experiment
from config import (
    DEFAULT_PARAMS, GA_PARAMS, RESULT_DIR,
    get_single_var_config, get_full_factorial_config,
    print_config_info
)


# ==================== 配置 ====================

# 选择要对比的变量 (修改此参数切换实验)
# 可选: 'data_size', 'num_UEs', 'bandwidth', 'mec_capacity', 'min_semantic_factor', None(全因子)
VAR_TO_COMPARE: str = 'num_UEs'  # 默认对比用户数量

# 是否运行全因子实验
RUN_FULL_FACTORIAL: bool = False


# ==================== 结果保存 ====================

def save_results_to_csv(results: List[Dict[str, Any]], filename: str):
    """
    保存结果到CSV文件

    参数:
        results: 结果列表
        filename: 文件名
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    filepath = os.path.join(RESULT_DIR, filename)

    if len(results) == 0:
        print("Warning: 没有结果可保存!")
        return

    # 获取字段名
    fieldnames = list(results[0].keys())

    # 写入CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"结果已保存到: {filepath}")


def save_best_solutions(solutions: List[Dict[str, Any]], filename: str):
    """
    保存最优解详情到CSV文件

    参数:
        solutions: 最优解列表
        filename: 文件名
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    filepath = os.path.join(RESULT_DIR, filename)

    if len(solutions) == 0:
        return

    # 获取字段名
    fieldnames = list(solutions[0].keys())

    # 写入CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(solutions)

    print(f"最优解已保存到: {filepath}")


# ==================== 实验运行 ====================

def run_experiments(var_name: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    运行单变量对比实验

    参数:
        var_name: 变量名
        verbose: 是否打印详细信息

    返回:
        results: 结果列表
    """
    # 获取配置
    configs = get_single_var_config(var_name)
    total_experiments = len(configs)

    if verbose:
        print(f"\n{'='*60}")
        print(f"开始实验: 对比变量 = {var_name}")
        print(f"总实验数: {total_experiments}")
        print(f"{'='*60}\n")

    results = []
    start_time = time.time()

    for i, config in enumerate(configs):
        # 打印进度
        if verbose:
            print(f"\n[{i+1}/{total_experiments}] ", end="")
            print(f"data_size={config['data_size']}KB, ", end="")
            print(f"num_UEs={config['num_UEs']}, ", end="")
            print(f"bandwidth={config['bandwidth']}kHz, ", end="")
            print(f"mec={config['mec_capacity']}Gcps, ", end="")
            print(f"min_se={config['min_semantic_factor']}")

        # 运行实验
        try:
            result, solution, energy_history = run_single_experiment(
                data_size=config['data_size'],
                num_UEs=config['num_UEs'],
                bandwidth=config['bandwidth'],
                mec_capacity=config['mec_capacity'],
                min_semantic_factor=config['min_semantic_factor'],
                a=i+1,
                pop_size=GA_PARAMS['pop_size'],
                generations=GA_PARAMS['generations'],
                verbose=False  # 关闭详细信息避免输出过多
            )
            results.append(result)

            if verbose:
                print(f"  -> Energy: {result['total_energy']:.6f}, " , end="")
                print(f"Offload: {result['offload_count']}/{result['num_UEs']}, " , end="")
                print(f"Delay: {result['delay_penalty']:.6f}")

        except Exception as e:
            print(f"  -> Error: {str(e)}")
            results.append({
                'data_size': config['data_size'],
                'num_UEs': config['num_UEs'],
                'bandwidth': config['bandwidth'],
                'mec_capacity': config['mec_capacity'],
                'min_semantic_factor': config['min_semantic_factor'],
                'total_energy': -1,
                'delay_penalty': -1,
                'offload_count': -1,
                'local_count': -1,
                'best_generation': -1
            })

    # 统计时间
    elapsed = time.time() - start_time

    if verbose:
        print(f"\n{'='*60}")
        print(f"实验完成! 总耗时: {elapsed:.1f}秒")
        print(f"{'='*60}\n")

    return results


def run_full_factorial_experiments(verbose: bool = True) -> List[Dict[str, Any]]:
    """
    运行全因子实验

    参数:
        verbose: 是否打印详细信息

    返回:
        results: 结果列表
    """
    # 获取配置
    configs = get_full_factorial_config()
    total_experiments = len(configs)

    if verbose:
        print(f"\n{'='*60}")
        print(f"开始全因子实验")
        print(f"总实验数: {total_experiments}")
        print(f"警告: 全因子实验可能需要较长时间!")
        print(f"{'='*60}\n")

    results = []
    start_time = time.time()

    for i, config in enumerate(configs):
        # 打印进度
        if verbose and i % 10 == 0:
            print(f"\n[{i+1}/{total_experiments}] ", end="")
            print(f"data_size={config['data_size']}KB, ", end="")
            print(f"num_UEs={config['num_UEs']}, ", end="")
            print(f"bandwidth={config['bandwidth']}kHz, ", end="")
            print(f"mec={config['mec_capacity']}Gcps, ", end="")
            print(f"min_se={config['min_semantic_factor']}")

        # 运行实验
        try:
            result, solution, energy_history = run_single_experiment(
                data_size=config['data_size'],
                num_UEs=config['num_UEs'],
                bandwidth=config['bandwidth'],
                mec_capacity=config['mec_capacity'],
                min_semantic_factor=config['min_semantic_factor'],
                a=i+1,
                pop_size=GA_PARAMS['pop_size'],
                generations=GA_PARAMS['generations'],
                verbose=False
            )
            results.append(result)

            if verbose and i % 10 == 0:
                print(f"  -> Energy: {result['total_energy']:.6f}")

        except Exception as e:
            print(f"\n  -> Error at config {i+1}: {str(e)}")
            results.append({
                'data_size': config['data_size'],
                'num_UEs': config['num_UEs'],
                'bandwidth': config['bandwidth'],
                'mec_capacity': config['mec_capacity'],
                'min_semantic_factor': config['min_semantic_factor'],
                'total_energy': -1,
                'delay_penalty': -1,
                'offload_count': -1,
                'local_count': -1,
                'best_generation': -1
            })

    # 统计时间
    elapsed = time.time() - start_time

    if verbose:
        print(f"\n{'='*60}")
        print(f"实验完成! 总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        print(f"{'='*60}\n")

    return results


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 打印配置信息
    print_config_info()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GA批量实验')
    parser.add_argument('--var', type=str, default=VAR_TO_COMPARE,
                   help='要对比的变量名')
    parser.add_argument('--full', action='store_true',
                   help='运行全因子实验')
    parser.add_argument('--quiet', action='store_true',
                   help='安静模式，减少输出')

    args = parser.parse_args()

    verbose = not args.quiet

    # 运行实验
    if args.full or RUN_FULL_FACTORIAL:
        # 全因子实验
        results = run_full_factorial_experiments(verbose=verbose)
        filename = f"GA_results_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        # 单变量实验
        var_name = args.var if args.var else VAR_TO_COMPARE
        results = run_experiments(var_name, verbose=verbose)
        filename = f"GA_results_{var_name}.csv"

    # 保存结果
    save_results_to_csv(results, filename)

    # 打印统计信息
    if len(results) > 0:
        successful_results = [r for r in results if r['total_energy'] > 0]
        print(f"\n统计: 成功 {len(successful_results)}/{len(results)}")

        if len(successful_results) > 0:
            energies = [r['total_energy'] for r in successful_results]
            print(f"  能耗范围: {min(energies):.6f} - {max(energies):.6f}")
            print(f"  平均能耗: {sum(energies)/len(energies):.6f}")


if __name__ == '__main__':
    main()