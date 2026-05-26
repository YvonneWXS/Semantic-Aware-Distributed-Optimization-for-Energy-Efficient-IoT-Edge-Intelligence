"""
ALE / ROE Baseline 统一入口
支持命令行参数选择运行 ALE 或 ROE，支持加法遍历 num_of_ues。
"""

import argparse
import os
import csv
import sys

import numpy as np

from parameter import num_of_ues, EXPERIMENT_PARAMS


def main():
    parser = argparse.ArgumentParser(description="ALE / ROE Baseline 实验入口")
    parser.add_argument(
        "--baseline", type=str, default="all", choices=["ALE", "ROE", "all"],
        help="选择运行的 baseline (默认: all，同时运行 ALE 和 ROE)"
    )
    args = parser.parse_args()

    run_ale = args.baseline in ("ALE", "all")
    run_roe = args.baseline in ("ROE", "all")

    output_dir = EXPERIMENT_PARAMS["output_dir"]
    num_runs = EXPERIMENT_PARAMS["num_runs"]

    ale_summaries = []
    roe_summaries = []

    # ========== 加法遍历：依次改变 num_of_ues ==========
    for n_ues in num_of_ues:
        print(f"\n{'='*50}")
        print(f"实验: num_of_ues = {n_ues}")
        print(f"{'='*50}")

        for run_id in range(num_runs):
            if run_ale:
                from ALE import run as run_ale_baseline
                s = run_ale_baseline(num_ues=n_ues, run_id=run_id, output_dir=output_dir)
                ale_summaries.append(s)

            if run_roe:
                from ROE import run as run_roe_baseline
                s = run_roe_baseline(num_ues=n_ues, run_id=run_id, output_dir=output_dir)
                roe_summaries.append(s)

    # ========== 写入汇总 CSV ==========
    if run_ale and ale_summaries:
        _write_summary(os.path.join(output_dir, "ALE_summary.csv"), ale_summaries)
        print(f"\nALE 汇总已保存: {output_dir}/ALE_summary.csv")

    if run_roe and roe_summaries:
        _write_summary(os.path.join(output_dir, "ROE_summary.csv"), roe_summaries)
        print(f"ROE 汇总已保存: {output_dir}/ROE_summary.csv")

    print("\n实验完成。")


def _write_summary(path, summaries):
    """将汇总列表写入 CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "num_ues", "avg_energy", "std_energy", "min_energy", "max_energy",
        "avg_violation", "std_violation",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
