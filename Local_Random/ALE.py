"""
ALE (All Local Execution) baseline
所有 UE 的计算任务在本地执行，不发生卸载。
"""

import os
import csv
import numpy as np

from env import ENV
from parameter import (
    SYSTEM_PARAMS, EXPERIMENT_PARAMS, convert_units,
    DEFAULT_DATA_SIZE_KB, DEFAULT_BANDWIDTH_KHZ, DEFAULT_MEC_CAPACITY_GHZ,
)

EPISODES_PER_RUN = 500


def run(num_ues, run_id=0, output_dir="results"):
    """
    执行一次 ALE 实验
    - num_ues: UE 数量
    - run_id:  运行编号
    - output_dir: 结果输出根目录
    """
    # 构建保存路径
    save_dir = os.path.join(output_dir, f"ALE_{num_ues}UEs", f"run_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    # 保存配置
    _save_config(save_dir, num_ues)

    # 创建环境
    k = SYSTEM_PARAMS["k"]
    env = ENV(UEs=num_ues, MECs=1, k=k)

    log_rows = []

    for ep in range(EPISODES_PER_RUN):
        obs = env.reset()
        actions = _generate_actions(num_ues)
        obs_, total_energy, energy_per_ue, delay_per_ue, violation_per_ue = env.step(obs, actions)

        violation_rate = np.mean(violation_per_ue)
        log_rows.append({
            "episode": ep,
            "energy": total_energy,
            "violation_rate": violation_rate,
        })

    # 保存 train_log.csv
    log_path = os.path.join(save_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "energy", "violation_rate"])
        writer.writeheader()
        writer.writerows(log_rows)

    # 计算汇总
    energies = [r["energy"] for r in log_rows]
    violations = [r["violation_rate"] for r in log_rows]
    summary = {
        "num_ues": num_ues,
        "avg_energy": np.mean(energies),
        "std_energy": np.std(energies),
        "min_energy": np.min(energies),
        "max_energy": np.max(energies),
        "avg_violation": np.mean(violations),
        "std_violation": np.std(violations),
    }

    print(f"  [ALE] UEs={num_ues}, run={run_id}: "
          f"energy={summary['avg_energy']:.6e}, violation={summary['avg_violation']:.4f}")

    return summary


def _generate_actions(num_ues):
    """
    ALE: 所有 UE 强制本地执行
    返回: [[offload=0, semantic_factor=1.0, resource_allocation=0, w_bw=0], ...]
    """
    return [[0, 1.0, 0, 0] for _ in range(num_ues)]


def _save_config(save_dir, num_ues):
    """保存实验配置到 config.txt"""
    lines = [
        "=== ALE (All Local Execution) Config ===",
        f"num_ues = {num_ues}",
        f"episodes = {EPISODES_PER_RUN}",
        f"data_size_kb = {DEFAULT_DATA_SIZE_KB}",
        f"bandwidth_khz = {DEFAULT_BANDWIDTH_KHZ}",
        f"mec_capacity_ghz = {DEFAULT_MEC_CAPACITY_GHZ}",
        "",
        "--- SYSTEM_PARAMS ---",
    ]
    for k, v in SYSTEM_PARAMS.items():
        lines.append(f"{k} = {v}")
    lines.append("")
    lines.append(f"num_runs = {EXPERIMENT_PARAMS['num_runs']}")

    with open(os.path.join(save_dir, "config.txt"), "w") as f:
        f.write("\n".join(lines))
