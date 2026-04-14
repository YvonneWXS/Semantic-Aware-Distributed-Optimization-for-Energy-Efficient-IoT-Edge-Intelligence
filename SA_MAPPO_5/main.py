#!/usr/bin/env python
"""
Main entry point for SA-MAPPO batch experiments.
Supports both single experiment mode and batch experiment mode.
"""

import os
import sys
import argparse
import itertools
import subprocess
import json
from datetime import datetime
from pathlib import Path


def run_batch_experiments():
    """
    Run batch experiments with parameter grid.
    """
    print("=" * 80)
    print("Starting batch experiments for SA-MAPPO")
    print("=" * 80)

    # Fixed parameters for all experiments
    fixed_params = {
        "algorithm_name": "mappo",
        "experiment_name": "batch_experiment",
        "seed": 1,
        "cuda": False,
        "n_training_threads": 2,
        "n_rollout_threads": 1,
        "num_env_steps": 800000,
        "env_name": "MyEnv",
        "use_obs_instead_of_state": False,
        "episode_length": 50,
        "share_policy": False,
        "use_centralized_V": True,
        "hidden_size": 64,
        "layer_N": 1,
        "use_ReLU": True,
        "use_valuenorm": True,
        "use_feature_normalization": True,
        "use_orthogonal": True,
        "gain": 0.01,
        "use_recurrent_policy": False,
        "lr": 5e-5,
        "critic_lr": 5e-5,
        "opti_eps": 1e-5,
        "ppo_epoch": 15,
        "clip_param": 0.2,
        "num_mini_batch": 1,
        "entropy_coef": 0.01,
        "value_loss_coef": 1,
        "use_max_grad_norm": True,
        "max_grad_norm": 10.0,
        "use_gae": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "use_huber_loss": True,
        "huber_delta": 10.0,
        "use_linear_lr_decay": False,
        "save_interval": 1,
        "log_interval": 5,
        "use_eval": True,
        "eval_interval": 25,
        "eval_episodes": 32,
        "save_gifs": False,
        "use_render": False,
        "scenario_name": "MyEnv",
        "num_landmarks": 3,
    }

    # Parameter grid (using default values from config.py)
    param_grid = {
        "data_size": [128, 256, 512, 1024],  # KB
        "num_ues": [5, 10, 15, 20, 25, 30],
        "bandwidth": [750, 1000, 1500, 2000],  # kHz
        "mec_capacity": [10.0, 12.5, 15.0, 17.5, 20.0],  # Gcps
        "min_semantic_factor": [0.2, 0.3, 0.4, 0.5],
    }

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / "batch_experiments" / f"batch_study_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {results_dir}")
    print(f"Parameter grid size: {len(param_grid['data_size'])} x {len(param_grid['num_ues'])} x "
          f"{len(param_grid['bandwidth'])} x {len(param_grid['mec_capacity'])} x "
          f"{len(param_grid['min_semantic_factor'])} = "
          f"{len(param_grid['data_size']) * len(param_grid['num_ues']) * len(param_grid['bandwidth']) * len(param_grid['mec_capacity']) * len(param_grid['min_semantic_factor'])} combinations")

    # Save experiment configuration
    config = {
        "fixed_params": fixed_params,
        "param_grid": param_grid,
        "timestamp": timestamp,
        "results_dir": str(results_dir),
    }

    config_file = results_dir / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Experiment configuration saved to: {config_file}")

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    print(f"\nTotal experiments to run: {len(param_combinations)}")
    print("-" * 80)

    # Run each experiment
    experiment_results = []

    for i, combination in enumerate(param_combinations):
        exp_params = dict(zip(param_names, combination))

        # Create experiment name
        exp_name = f"exp_{i+1:04d}_ds{exp_params['data_size']}_ue{exp_params['num_ues']}_bw{exp_params['bandwidth']}_mc{exp_params['mec_capacity']}_sf{exp_params['min_semantic_factor']}"

        print(f"\nExperiment {i+1}/{len(param_combinations)}: {exp_name}")
        print(f"  Parameters: data_size={exp_params['data_size']}KB, num_ues={exp_params['num_ues']}, "
              f"bandwidth={exp_params['bandwidth']}kHz, mec_capacity={exp_params['mec_capacity']}Gcps, "
              f"min_semantic_factor={exp_params['min_semantic_factor']}")

        # Build command
        cmd = [
            "python", "train/train.py",
            f"--experiment_name={exp_name}",
            f"--num_agents={exp_params['num_ues']}",
            f"--data_size={exp_params['data_size']}",
            f"--bandwidth={exp_params['bandwidth']}",
            f"--mec_capacity={exp_params['mec_capacity']}",
            f"--min_semantic_factor={exp_params['min_semantic_factor']}",
        ]

        # Add fixed parameters
        for key, value in fixed_params.items():
            if key not in ["experiment_name", "num_agents"]:  # Already added
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                else:
                    cmd.append(f"--{key}={value}")

        print(f"  Command: {' '.join(cmd[:10])}...")  # Show first 10 args

        # Run experiment
        try:
            # Create experiment directory
            exp_dir = results_dir / exp_name
            exp_dir.mkdir(exist_ok=True)

            # Save experiment parameters
            exp_config = {
                "experiment_name": exp_name,
                "parameters": exp_params,
                "fixed_params": fixed_params,
                "command": cmd,
                "start_time": datetime.now().isoformat(),
            }

            exp_config_file = exp_dir / "config.json"
            with open(exp_config_file, "w") as f:
                json.dump(exp_config, f, indent=2)

            # Run the experiment
            print(f"  Starting experiment...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            # Save output
            stdout_file = exp_dir / "stdout.log"
            stderr_file = exp_dir / "stderr.log"

            with open(stdout_file, "w") as f:
                f.write(result.stdout)

            with open(stderr_file, "w") as f:
                f.write(result.stderr)

            # Record result
            exp_result = {
                "experiment_name": exp_name,
                "parameters": exp_params,
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "end_time": datetime.now().isoformat(),
                "stdout_file": str(stdout_file),
                "stderr_file": str(stderr_file),
            }

            experiment_results.append(exp_result)

            if result.returncode == 0:
                print(f"  ✓ Experiment completed successfully")
            else:
                print(f"  ✗ Experiment failed with return code {result.returncode}")
                print(f"    See logs in: {exp_dir}")

        except Exception as e:
            print(f"  ✗ Error running experiment: {e}")
            experiment_results.append({
                "experiment_name": exp_name,
                "parameters": exp_params,
                "error": str(e),
                "success": False,
            })

        print("-" * 80)

    # Save summary of all experiments
    summary = {
        "total_experiments": len(param_combinations),
        "successful_experiments": sum(1 for r in experiment_results if r.get("success", False)),
        "failed_experiments": sum(1 for r in experiment_results if not r.get("success", True)),
        "experiment_results": experiment_results,
        "end_time": datetime.now().isoformat(),
    }

    summary_file = results_dir / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("Batch experiments completed!")
    print(f"Results directory: {results_dir}")
    print(f"Successful experiments: {summary['successful_experiments']}/{summary['total_experiments']}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 80)

    return results_dir


def main():
    parser = argparse.ArgumentParser(description="SA-MAPPO Batch Experiment Runner")
    parser.add_argument("--batch", action="store_true", default=False,
                        help="Run batch experiments with parameter grid")
    parser.add_argument("--single", action="store_true", default=False,
                        help="Run single experiment (default behavior)")

    args = parser.parse_args()

    if args.batch:
        # Run batch experiments
        run_batch_experiments()
    else:
        # Run single experiment (default behavior - pass through to train.py)
        print("Running single experiment mode...")
        print("Passing arguments to train/train.py")

        # Pass all arguments except --batch and --single to train.py
        import sys
        cmd_args = [arg for arg in sys.argv[1:] if arg not in ["--batch", "--single"]]

        cmd = ["python", "train/train.py"] + cmd_args
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        sys.exit(result.returncode)


if __name__ == "__main__":
    main()