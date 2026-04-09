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
import time 

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
    parser.add_argument("--num_agents", type=int, default=10, help="number of players") #TODO

    all_args = parser.parse_known_args(args)[0]

    return all_args

class Logger(object):
    def __init__(self, filename="default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8') # 追加模式

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 每次写入都强制刷新，防止丢失

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    print(f"\n[Config Check] Experiment Name: '{all_args.experiment_name}' (Type: {type(all_args.experiment_name)})")
    print(f"[Config Check] Seed: {all_args.seed}")
    print(f"[Config Check] Device: {torch.device('cuda:0' if torch.cuda.is_available() and all_args.cuda else 'cpu')}")

    if "," in str(all_args.experiment_name) or "(" in str(all_args.experiment_name):
        raise ValueError(f"检测到非法的 experiment_name: {all_args.experiment_name}。请检查是否传入了元组！")

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
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
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

    print(f"[System] 初始化 Runner...")
    runner = Runner(config)

    print(f"[System] 开始训练 (Run Directory: {run_dir})")
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

   
    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()
    print(f"[System] 训练正常结束。")


if __name__ == "__main__":
    sys.stdout = Logger("master_log.txt")

    print(f"\n\n{'='*50}")
    print(f"启动批量训练任务：时间 {time.strftime('%Y-%m-%d %H:%M:%S') if 'time' in locals() else ''}")
    print(f"{'='*50}")
    # for a in range (1,9,1):
    

    try:
            main(sys.argv[1:])
            
            
    except Exception as e:
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("详细堆栈追踪 (Traceback):")
        import traceback
        traceback.print_exc() # 这行代码会把报错的具体行数写进日志
        
        # 强制清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"--------------------------------------------------")
        print(f"跳过此 Run，尝试继续下一个...")
        sys.exit(1)