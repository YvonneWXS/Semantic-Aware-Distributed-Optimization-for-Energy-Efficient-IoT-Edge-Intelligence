"""
ROE (All Random Execution) baseline
卸载决策、语义提取因子、资源分配、带宽权重均为随机选择。
"""

import os
import csv
import numpy as np

from env import ENV
from parameter import (
    SYSTEM_PARAMS, EXPERIMENT_PARAMS,
    DEFAULT_DATA_SIZE_KB, DEFAULT_BANDWIDTH_KHZ, DEFAULT_MEC_CAPACITY_GHZ,
)

EPISODES_PER_RUN = 500


def run(num_ues, run_id=0, output_dir="results"):
    """
    执行一次 ROE 实验
    - num_ues: UE 数量
    - run_id:  运行编号
    - output_dir: 结果输出根目录
    """
    save_dir = os.path.join(output_dir, f"ROE_{num_ues}UEs", f"run_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    _save_config(save_dir, num_ues)

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

    # 汇总
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

    print(f"  [ROE] UEs={num_ues}, run={run_id}: "
          f"energy={summary['avg_energy']:.6e}, violation={summary['avg_violation']:.4f}")

    return summary


def _generate_actions(num_ues):
    """
    ROE: 为每个 UE 随机生成动作
    - offload_decision: 50% 概率 0 或 1
    - offload=0 → 其他三维强制归零
    - offload=1 → semantic ∈ [0.3,1], resource ∈ [0.01,1], w_bw ∈ {1,2,3,4}
    """
    actions = []
    for _ in range(num_ues):
        offload = np.random.choice([0, 1])
        if offload == 0:
            actions.append([0, 0, 0, 0])
        else:
            semantic = np.random.uniform(0.3, 1.0)
            resource = np.random.uniform(0.01, 1.0)
            w_bw = np.random.choice([1, 2, 3, 4])
            actions.append([offload, semantic, resource, w_bw])
    return actions


def _save_config(save_dir, num_ues):
    """保存实验配置到 config.txt"""
    lines = [
        "=== ROE (All Random Execution) Config ===",
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
