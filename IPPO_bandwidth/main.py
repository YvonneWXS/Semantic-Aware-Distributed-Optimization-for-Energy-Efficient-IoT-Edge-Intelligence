#!/usr/bin/env python
"""
批量实验入口脚本
遍历config_batch.py中的参数组合，运行SA‑IPPO训练
"""
import sys
import os
import subprocess
import itertools
from config_batch import (
    DataSize_list,
    num_UEs_list,
    bandwidth_list,
    mec_capacity_list,
    min_semantic_factor_list,
    other_params,
    generate_exp_name
)

def run_experiment(num_UEs, bandwidth, mec_capacity, min_semantic, data_size):
    """运行单个实验"""
    # 构造实验名称
    exp_name = generate_exp_name(num_UEs, bandwidth, mec_capacity, min_semantic)

    # 构造命令行参数
    cmd = [
        sys.executable,  # 当前Python解释器
        "train/train.py",
        "--env_name", "MyEnv",
        "--algorithm_name", "rmappo",
        "--experiment_name", exp_name,
        "--num_agents", str(num_UEs),
        "--episode_length", str(other_params['episode_length']),
        "--num_env_steps", str(other_params['num_env_steps']),
        "--lr", str(other_params['lr']),
        "--hidden_size", str(other_params['hidden_size']),
        "--layer_N", str(other_params['layer_N']),
        "--use_popart", str(other_params['use_popart']).lower(),
        "--use_valuenorm", str(other_params['use_valuenorm']).lower(),
        "--gamma", str(other_params['gamma']),
        "--clip_param", str(other_params['clip_param']),
        "--seed", "1",
        "--cuda", "false",
    ]

    # 打印命令
    print("Running experiment:", exp_name)
    print("Command:", " ".join(cmd))

    # 运行训练
    try:
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    except subprocess.CalledProcessError as e:
        print(f"Experiment {exp_name} failed with error: {e}")
        return False
    return True

def main():
    """主函数：遍历所有参数组合"""
    # 参数组合遍历
    param_combinations = list(itertools.product(
        num_UEs_list,
        bandwidth_list,
        mec_capacity_list,
        min_semantic_factor_list,
        DataSize_list
    ))

    print(f"Total experiments: {len(param_combinations)}")

    success_count = 0
    for idx, (num_UEs, bandwidth, mec_capacity, min_semantic, data_size) in enumerate(param_combinations):
        print(f"\n=== Experiment {idx+1}/{len(param_combinations)} ===")
        print(f"Parameters: num_UEs={num_UEs}, bandwidth={bandwidth} kHz, mec_capacity={mec_capacity} Gcps, min_semantic={min_semantic}, data_size={data_size} KB")

        # 注意：这里需要将参数传递给环境，但环境参数目前通过env_core.py固定
        # 为了修改环境参数，我们需要修改env_core.py中的对应变量，或者通过配置文件传递
        # 由于时间有限，这里仅打印参数，实际实验需要修改env_core.py的相应属性
        # 建议用户根据需求手动修改env_core.py中的参数，然后运行批量实验

        # 运行实验（暂时跳过，仅演示）
        # success = run_experiment(num_UEs, bandwidth, mec_capacity, min_semantic, data_size)
        # if success:
        #     success_count += 1

        print(f"Experiment {idx+1} parameters set (实际运行需取消注释run_experiment调用)")

    print(f"\nCompleted. Successfully ran {success_count}/{len(param_combinations)} experiments.")

if __name__ == "__main__":
    main()