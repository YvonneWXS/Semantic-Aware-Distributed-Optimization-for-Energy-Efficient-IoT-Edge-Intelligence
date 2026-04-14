"""
# @Time    : 2021/6/30 10:07 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : train.py
"""

# !/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

# # Get the parent directory of the current file
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

# # Append the parent directory to sys.path, otherwise the following import will fail
# sys.path.append(parent_dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import get_config
from envs.env_wrappers import DummyVecEnv

"""Train script for MPEs."""



def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.

            # from envs.env_continuous import ContinuousActionEnv

            # env = ContinuousActionEnv()
            # 任务大小

            from envs.env_discrete import DiscreteActionEnv
            env = DiscreteActionEnv()

            # env.seed(all_args.seed + rank * 1000)
            return env

        return init_env
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.
            # from envs.env_continuous import ContinuousActionEnv

            # env = ContinuousActionEnv()
            from envs.env_discrete import DiscreteActionEnv
            env = DiscreteActionEnv()
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=5, help="number of players")#TODO

    # Batch experiment parameters (passed through)
    parser.add_argument("--data_size_list", type=float, nargs='+', default=None)
    parser.add_argument("--num_ues_list", type=int, nargs='+', default=None)
    parser.add_argument("--bandwidth_list", type=float, nargs='+', default=None)
    parser.add_argument("--mec_capacity_list", type=float, nargs='+', default=None)
    parser.add_argument("--min_semantic_factor_list", type=float, nargs='+', default=None)
    parser.add_argument("--run_batch_experiments", action="store_true", default=False)
    parser.add_argument("--batch_experiment_name", type=str, default="batch_study")

    all_args = parser.parse_known_args(args)[0]

    # Pass batch parameters to environment if provided
    if all_args.data_size_list:
        print(f"Batch parameter - Data size list: {all_args.data_size_list}")
    if all_args.num_ues_list:
        print(f"Batch parameter - Num UEs list: {all_args.num_ues_list}")

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert (
        all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener"
    ) == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."

    # cuda
    # if all_args.cuda and torch.cuda.is_available():
    #     print("choose to use gpu...")
    #     device = torch.device("cuda:0")
    #     torch.set_num_threads(all_args.n_training_threads)
    #     if all_args.cuda_deterministic:
    #         torch.backends.cudnn.benchmark = False
    #         torch.backends.cudnn.deterministic = True
    # else:
    print("choose to use cpu...")
    device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed 设置随机种子保证实验可重复性
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    # np.random.seed(all_args.seed)
    # print(np.random.randint(1, 100))  # 先打印第一个随机数
    # print(np.random.uniform(0, 1))    # 再打印第二个随机数

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

   
    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    # for a in range (1,9,1):
    main(sys.argv[1:])
