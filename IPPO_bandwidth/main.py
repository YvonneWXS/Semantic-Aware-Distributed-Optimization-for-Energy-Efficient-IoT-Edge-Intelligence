#!/usr/bin/env python
"""
无线通信与边缘计算批量实验入口文件

功能：
1. 支持批量参数遍历实验
2. 使用config.py中的BatchExperimentConfig类获取参数组合
3. 对每个参数组合运行IPPO训练
4. 保存实验结果（奖励曲线、能耗、时延等）
5. 提供实验进度跟踪和结果汇总

作者：无线通信与边缘计算算法工程师
日期：2026/04/15
"""

import os
import sys
import time
import json
import pickle
import argparse
import subprocess
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 可选依赖
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("警告: numpy未安装，某些功能可能受限")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("警告: pandas未安装，汇总报告功能可能受限")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，可视化功能可能受限")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("警告: tqdm未安装，进度条功能不可用")
    # 创建简单的进度条替代
    class SimpleProgressBar:
        def __init__(self, total, desc=""):
            self.total = total
            self.desc = desc
            self.current = 0
            self.start_time = time.time()

        def update(self, n=1):
            self.current += n
            elapsed = time.time() - self.start_time
            percent = self.current / self.total * 100
            print(f"\r{self.desc}: {self.current}/{self.total} ({percent:.1f}%) 耗时: {elapsed:.1f}s", end="")

        def set_postfix(self, **kwargs):
            # 简单实现，忽略后缀
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            print()  # 换行

    tqdm = SimpleProgressBar

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config import BatchExperimentConfig, get_config


class ExperimentRunner:
    """批量实验运行器"""

    def __init__(self, output_dir: str = "batch_experiments",
                 num_workers: int = 1,
                 resume: bool = False):
        """
        初始化实验运行器

        参数：
            output_dir: 输出目录
            num_workers: 并行工作进程数
            resume: 是否恢复中断的实验
        """
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.resume = resume

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 实验状态文件
        self.status_file = self.output_dir / "experiment_status.json"
        self.results_file = self.output_dir / "experiment_results.pkl"
        self.summary_file = self.output_dir / "experiment_summary.csv"

        # 加载或初始化实验状态
        self.experiment_status = self._load_experiment_status()

        # 获取所有参数组合
        self.param_combinations = BatchExperimentConfig.generate_param_combinations()

        # 实验统计信息
        self.stats = {
            'total': len(self.param_combinations),
            'completed': 0,
            'failed': 0,
            'pending': 0,
            'start_time': datetime.now().isoformat()
        }

    def _load_experiment_status(self) -> Dict:
        """加载实验状态"""
        if self.resume and self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"警告：无法加载实验状态文件，将重新开始: {e}")

        # 初始化状态
        return {
            'experiments': {},
            'last_update': datetime.now().isoformat(),
            'version': '1.0'
        }

    def _save_experiment_status(self):
        """保存实验状态"""
        self.experiment_status['last_update'] = datetime.now().isoformat()
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_status, f, indent=2, ensure_ascii=False)

    def _save_results(self, results: List[Dict]):
        """保存实验结果"""
        with open(self.results_file, 'wb') as f:
            pickle.dump(results, f)

    def _load_results(self) -> List[Dict]:
        """加载实验结果"""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"警告：无法加载结果文件: {e}")
        return []

    def _create_experiment_dir(self, experiment_id: str) -> Path:
        """创建实验目录"""
        exp_dir = self.output_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _update_env_core_params(self, params: Dict, env_core_path: Path) -> bool:
        """
        更新env_core.py中的参数

        注意：这是一个示例实现，实际使用时需要根据env_core.py的具体结构进行调整
        """
        try:
            # 读取env_core.py文件
            with open(env_core_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 这里需要根据env_core.py的实际结构来更新参数
            # 例如，如果env_core.py中有类似 self.agent_num = 5 的代码
            # 我们可以用正则表达式或字符串替换来更新这些值

            # 由于env_core.py的结构可能比较复杂，这里提供一个模板方法
            # 实际使用时需要根据具体代码结构进行调整

            # 示例：更新agent_num
            if 'num_UEs' in params:
                # 查找并替换agent_num
                import re
                pattern = r'self\.agent_num\s*=\s*\d+'
                replacement = f'self.agent_num = {params["num_UEs"]}'
                content = re.sub(pattern, replacement, content)

            # 示例：更新任务大小
            if 'data_size' in params:
                # 查找并替换task_size
                pattern = r'self\.task_size\[i\]\s*=\s*\d+\s*\*\s*8\s*\*\s*self\.KB'
                data_size_bits = params['data_size'] * 8 * 1024  # KB to bits
                replacement = f'self.task_size[i] = {data_size_bits}  # {params["data_size"]} KB'
                content = re.sub(pattern, replacement, content)

            # 示例：更新MEC计算能力
            if 'mec_capacity' in params:
                # 查找并替换MEC_f
                pattern = r'self\.MEC_f\s*=\s*[\d\.]+\s*\*\s*self\.GHz'
                mec_capacity_hz = params['mec_capacity'] * 1e9  # Gcps to Hz
                replacement = f'self.MEC_f = {mec_capacity_hz}  # {params["mec_capacity"]} Gcps'
                content = re.sub(pattern, replacement, content)

            # 示例：更新传输带宽
            if 'bandwidth' in params:
                # 查找并替换transmission_bandwidth
                pattern = r'self\.transmission_bandwidth\s*=\s*[\d\.]+\s*\*\s*self\.mHz'
                bandwidth_hz = params['bandwidth'] * 1000  # kHz to Hz
                replacement = f'self.transmission_bandwidth = {bandwidth_hz}  # {params["bandwidth"]} kHz'
                content = re.sub(pattern, replacement, content)

            # 写入更新后的文件
            with open(env_core_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True

        except Exception as e:
            print(f"更新env_core.py参数失败: {e}")
            return False

    def _run_single_experiment(self, params: Dict, experiment_id: str) -> Dict:
        """
        运行单个实验

        参数：
            params: 实验参数
            experiment_id: 实验ID

        返回：
            Dict: 实验结果
        """
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始实验 {experiment_id}")
        print(f"参数: data_size={params['data_size']}KB, num_UEs={params['num_UEs']}, "
              f"bandwidth={params['bandwidth']}kHz, mec_capacity={params['mec_capacity']}Gcps")

        start_time = time.time()
        result = {
            'experiment_id': experiment_id,
            'params': params.copy(),
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'error': None,
            'metrics': {}
        }

        try:
            # 创建实验目录
            exp_dir = self._create_experiment_dir(experiment_id)

            # 保存参数配置
            params_file = exp_dir / "params.json"
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)

            # 方法1：直接调用train.py（推荐）
            # 通过命令行参数传递配置
            cmd = [
                sys.executable,  # 使用当前Python解释器
                str(Path(project_root) / "train" / "train.py"),
                f"--experiment_name={experiment_id}",
                f"--num_agents={params['num_UEs']}",
                f"--num_env_steps={params.get('num_env_steps', 200000)}",
                f"--episode_length={params.get('episode_length', 50)}",
                f"--hidden_size={params.get('hidden_size', 64)}",
                f"--lr={params.get('lr', 5e-4)}",
                f"--seed={params.get('seed', 42)}"
            ]

            # 添加更多参数（如果需要）
            if 'algorithm_name' in params:
                cmd.append(f"--algorithm_name={params['algorithm_name']}")

            # 设置工作目录
            env = os.environ.copy()
            env['PYTHONPATH'] = project_root

            # 运行训练
            print(f"运行命令: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=params.get('max_runtime', 3600)  # 超时时间
            )

            # 检查运行结果
            if process.returncode == 0:
                result['status'] = 'completed'
                result['stdout'] = process.stdout
                result['stderr'] = process.stderr

                # 解析训练输出，提取关键指标
                # 这里需要根据train.py的实际输出来解析
                metrics = self._parse_training_output(process.stdout)
                result['metrics'].update(metrics)

                # 尝试从结果目录加载训练日志
                self._load_training_metrics(exp_dir, result)

            else:
                result['status'] = 'failed'
                result['error'] = f"训练过程返回非零退出码: {process.returncode}"
                result['stdout'] = process.stdout
                result['stderr'] = process.stderr
                print(f"实验 {experiment_id} 失败: {result['error']}")
                print(f"标准错误输出:\n{process.stderr}")

        except subprocess.TimeoutExpired:
            result['status'] = 'timeout'
            result['error'] = f"训练超时（超过{params.get('max_runtime', 3600)}秒）"
            print(f"实验 {experiment_id} 超时")

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            print(f"实验 {experiment_id} 异常: {e}")

        # 计算运行时间
        result['end_time'] = datetime.now().isoformat()
        result['duration'] = time.time() - start_time

        # 保存实验结果
        result_file = exp_dir / "result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 实验 {experiment_id} 完成，状态: {result['status']}, "
              f"耗时: {result['duration']:.2f}秒")

        return result

    def _parse_training_output(self, output: str) -> Dict:
        """解析训练输出，提取关键指标"""
        metrics = {}

        # 这里需要根据train.py的实际输出来解析
        # 示例：查找平均奖励
        import re

        # 查找平均奖励
        reward_pattern = r'average_reward.*?([\d\.-]+)'
        matches = re.findall(reward_pattern, output)
        if matches:
            try:
                rewards = [float(m) for m in matches]
                metrics['final_average_reward'] = rewards[-1] if rewards else 0
                metrics['max_average_reward'] = max(rewards) if rewards else 0
            except:
                pass

        # 查找能耗
        energy_pattern = r'energy.*?([\d\.eE+-]+)'
        matches = re.findall(energy_pattern, output)
        if matches:
            try:
                energies = [float(m) for m in matches]
                metrics['final_energy'] = energies[-1] if energies else 0
                metrics['min_energy'] = min(energies) if energies else 0
            except:
                pass

        # 查找时延
        delay_pattern = r'delay.*?([\d\.eE+-]+)'
        matches = re.findall(delay_pattern, output)
        if matches:
            try:
                delays = [float(m) for m in matches]
                metrics['final_delay'] = delays[-1] if delays else 0
                metrics['min_delay'] = min(delays) if delays else 0
            except:
                pass

        return metrics

    def _load_training_metrics(self, exp_dir: Path, result: Dict):
        """从训练结果目录加载指标"""
        try:
            # 查找训练结果目录
            results_dir = Path(project_root) / "results"
            if results_dir.exists():
                # 查找最新的运行目录
                run_dirs = list(results_dir.glob("**/run*"))
                if run_dirs:
                    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)

                    # 查找TensorBoard日志
                    tb_logs = list(latest_run.glob("events.out.tfevents.*"))
                    if tb_logs:
                        # 这里可以添加TensorBoard日志解析代码
                        pass

                    # 查找JSON摘要文件
                    json_files = list(latest_run.glob("*.json"))
                    for json_file in json_files:
                        if json_file.name == "summary.json":
                            with open(json_file, 'r', encoding='utf-8') as f:
                                summary = json.load(f)
                                result['metrics']['tensorboard_summary'] = summary
        except Exception as e:
            print(f"加载训练指标失败: {e}")

    def _worker_function(self, task_queue: multiprocessing.Queue,
                        result_queue: multiprocessing.Queue):
        """工作进程函数"""
        while True:
            try:
                # 获取任务
                task = task_queue.get(timeout=1)
                if task is None:  # 结束信号
                    break

                experiment_id, params = task
                result = self._run_single_experiment(params, experiment_id)
                result_queue.put(result)

            except multiprocessing.queues.Empty:
                continue
            except Exception as e:
                print(f"工作进程异常: {e}")
                break

    def run_experiments(self, start_id: int = 1, end_id: Optional[int] = None):
        """运行批量实验"""
        print("=" * 80)
        print("无线通信与边缘计算批量实验")
        print("=" * 80)

        # 获取实验摘要
        summary = BatchExperimentConfig.get_experiment_summary()
        print(f"总实验数量: {summary['total_experiments']}")
        print(f"数据大小范围: {summary['data_size_range']}")
        print(f"UE数量范围: {summary['num_UEs_range']}")
        print(f"带宽范围: {summary['bandwidth_range']}")
        print(f"MEC容量范围: {summary['mec_capacity_range']}")
        print(f"输出目录: {self.output_dir}")
        print(f"并行工作进程数: {self.num_workers}")
        print(f"恢复模式: {self.resume}")
        print("=" * 80)

        # 确定要运行的实验范围
        if end_id is None:
            end_id = len(self.param_combinations)

        experiments_to_run = []
        for i in range(start_id - 1, end_id):
            experiment_id = f"exp_{i+1:04d}"
            params = self.param_combinations[i]

            # 检查是否已经完成
            if self.resume and experiment_id in self.experiment_status['experiments']:
                status = self.experiment_status['experiments'][experiment_id].get('status', 'pending')
                if status in ['completed', 'running']:
                    print(f"实验 {experiment_id} 已{status}，跳过")
                    continue

            experiments_to_run.append((experiment_id, params))

        if not experiments_to_run:
            print("没有需要运行的实验")
            return

        print(f"需要运行的实验数量: {len(experiments_to_run)}")

        # 准备任务队列
        if self.num_workers > 1:
            # 使用多进程
            task_queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()

            # 添加任务到队列
            for task in experiments_to_run:
                task_queue.put(task)

            # 添加结束信号
            for _ in range(self.num_workers):
                task_queue.put(None)

            # 创建工作进程
            workers = []
            for i in range(self.num_workers):
                worker = multiprocessing.Process(
                    target=self._worker_function,
                    args=(task_queue, result_queue)
                )
                worker.start()
                workers.append(worker)

            # 收集结果
            results = []
            with tqdm(total=len(experiments_to_run), desc="运行实验") as pbar:
                for _ in range(len(experiments_to_run)):
                    result = result_queue.get()
                    results.append(result)

                    # 更新状态
                    self.experiment_status['experiments'][result['experiment_id']] = {
                        'status': result['status'],
                        'end_time': result['end_time'],
                        'duration': result['duration']
                    }
                    self._save_experiment_status()

                    pbar.update(1)
                    pbar.set_postfix({
                        '完成': f"{len(results)}/{len(experiments_to_run)}",
                        '状态': result['status']
                    })

            # 等待工作进程结束
            for worker in workers:
                worker.join()

        else:
            # 单进程运行
            results = []
            with tqdm(total=len(experiments_to_run), desc="运行实验") as pbar:
                for experiment_id, params in experiments_to_run:
                    result = self._run_single_experiment(params, experiment_id)
                    results.append(result)

                    # 更新状态
                    self.experiment_status['experiments'][result['experiment_id']] = {
                        'status': result['status'],
                        'end_time': result['end_time'],
                        'duration': result['duration']
                    }
                    self._save_experiment_status()

                    pbar.update(1)
                    pbar.set_postfix({
                        '完成': f"{len(results)}/{len(experiments_to_run)}",
                        '状态': result['status']
                    })

        # 保存所有结果
        all_results = self._load_results()
        all_results.extend(results)
        self._save_results(all_results)

        # 生成汇总报告
        self.generate_summary_report(all_results)

        print("\n" + "=" * 80)
        print("批量实验完成!")
        print(f"总实验数: {len(experiments_to_run)}")
        print(f"成功: {sum(1 for r in results if r['status'] == 'completed')}")
        print(f"失败: {sum(1 for r in results if r['status'] == 'failed')}")
        print(f"超时: {sum(1 for r in results if r['status'] == 'timeout')}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 80)

    def generate_summary_report(self, results: List[Dict]):
        """生成汇总报告"""
        if not results:
            print("没有实验结果可生成报告")
            return

        # 提取关键数据
        data = []
        for result in results:
            if result['status'] != 'completed':
                continue

            row = {
                'experiment_id': result['experiment_id'],
                'data_size_KB': result['params']['data_size'],
                'num_UEs': result['params']['num_UEs'],
                'bandwidth_kHz': result['params']['bandwidth'],
                'mec_capacity_Gcps': result['params']['mec_capacity'],
                'duration_seconds': result['duration'],
                'status': result['status']
            }

            # 添加指标
            metrics = result.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = value

            data.append(row)

        if not data:
            print("没有成功的实验可生成报告")
            return

        # 创建DataFrame
        df = pd.DataFrame(data)

        # 保存为CSV
        df.to_csv(self.summary_file, index=False, encoding='utf-8-sig')

        # 生成统计摘要
        summary_stats = {
            'total_experiments': len(results),
            'completed_experiments': len(data),
            'success_rate': len(data) / len(results) * 100 if results else 0,
            'avg_duration_seconds': df['duration_seconds'].mean() if not df.empty else 0,
            'total_duration_hours': df['duration_seconds'].sum() / 3600 if not df.empty else 0
        }

        # 按参数分组统计
        if not df.empty:
            # 按数据大小分组
            data_size_stats = df.groupby('data_size_KB').agg({
                'duration_seconds': 'mean',
                'final_average_reward': 'mean' if 'final_average_reward' in df.columns else None
            }).reset_index()

            # 按UE数量分组
            ue_stats = df.groupby('num_UEs').agg({
                'duration_seconds': 'mean',
                'final_average_reward': 'mean' if 'final_average_reward' in df.columns else None
            }).reset_index()

        # 保存统计摘要
        stats_file = self.output_dir / "experiment_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)

        # 生成可视化图表
        self._generate_visualizations(df)

        print(f"\n汇总报告已生成:")
        print(f"  - 详细结果: {self.summary_file}")
        print(f"  - 统计摘要: {stats_file}")
        print(f"  - 实验状态: {self.status_file}")
        print(f"  - 原始结果: {self.results_file}")

        # 打印关键统计信息
        print(f"\n关键统计信息:")
        print(f"  总实验数: {summary_stats['total_experiments']}")
        print(f"  成功实验数: {summary_stats['completed_experiments']}")
        print(f"  成功率: {summary_stats['success_rate']:.1f}%")
        print(f"  平均实验时长: {summary_stats['avg_duration_seconds']:.1f}秒")
        print(f"  总实验时长: {summary_stats['total_duration_hours']:.1f}小时")

    def _generate_visualizations(self, df: pd.DataFrame):
        """生成可视化图表"""
        if df.empty:
            return

        try:
            # 创建可视化目录
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # 1. 奖励随数据大小的变化
            if 'data_size_KB' in df.columns and 'final_average_reward' in df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(df['data_size_KB'], df['final_average_reward'], alpha=0.6)
                plt.xlabel('数据大小 (KB)')
                plt.ylabel('最终平均奖励')
                plt.title('奖励随数据大小的变化')
                plt.grid(True, alpha=0.3)
                plt.savefig(viz_dir / "reward_vs_data_size.png", dpi=150, bbox_inches='tight')
                plt.close()

            # 2. 奖励随UE数量的变化
            if 'num_UEs' in df.columns and 'final_average_reward' in df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(df['num_UEs'], df['final_average_reward'], alpha=0.6)
                plt.xlabel('UE数量')
                plt.ylabel('最终平均奖励')
                plt.title('奖励随UE数量的变化')
                plt.grid(True, alpha=0.3)
                plt.savefig(viz_dir / "reward_vs_num_UEs.png", dpi=150, bbox_inches='tight')
                plt.close()

            # 3. 实验时长分布
            if 'duration_seconds' in df.columns:
                plt.figure(figsize=(10, 6))
                plt.hist(df['duration_seconds'], bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('实验时长 (秒)')
                plt.ylabel('频数')
                plt.title('实验时长分布')
                plt.grid(True, alpha=0.3)
                plt.savefig(viz_dir / "experiment_duration_distribution.png", dpi=150, bbox_inches='tight')
                plt.close()

            # 4. 参数组合热图（示例：数据大小 vs UE数量）
            if all(col in df.columns for col in ['data_size_KB', 'num_UEs', 'final_average_reward']):
                # 创建透视表
                pivot_table = df.pivot_table(
                    values='final_average_reward',
                    index='data_size_KB',
                    columns='num_UEs',
                    aggfunc='mean'
                )

                if not pivot_table.empty:
                    plt.figure(figsize=(12, 8))
                    import seaborn as sns
                    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlOrRd")
                    plt.title('平均奖励热图（数据大小 vs UE数量）')
                    plt.xlabel('UE数量')
                    plt.ylabel('数据大小 (KB)')
                    plt.savefig(viz_dir / "reward_heatmap.png", dpi=150, bbox_inches='tight')
                    plt.close()

            print(f"  可视化图表: {viz_dir}/")

        except Exception as e:
            print(f"生成可视化图表失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="无线通信与边缘计算批量实验入口",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="batch_experiments",
        help="输出目录（默认: batch_experiments）"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="并行工作进程数（默认: 1）"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="恢复中断的实验"
    )

    parser.add_argument(
        "--start_id",
        type=int,
        default=1,
        help="起始实验ID（默认: 1）"
    )

    parser.add_argument(
        "--end_id",
        type=int,
        default=None,
        help="结束实验ID（默认: 所有实验）"
    )

    parser.add_argument(
        "--list_params",
        action="store_true",
        help="列出所有参数组合而不运行实验"
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="显示实验配置摘要"
    )

    args = parser.parse_args()

    # 显示实验配置摘要
    if args.summary:
        summary = BatchExperimentConfig.get_experiment_summary()
        print("实验配置摘要:")
        print(f"  总实验数量: {summary['total_experiments']}")
        print(f"  数据大小范围: {summary['data_size_range']}")
        print(f"  UE数量范围: {summary['num_UEs_range']}")
        print(f"  带宽范围: {summary['bandwidth_range']}")
        print(f"  MEC容量范围: {summary['mec_capacity_range']}")
        print(f"  固定参数: {json.dumps(summary['fixed_params'], indent=2, ensure_ascii=False)}")
        return

    # 列出参数组合
    if args.list_params:
        combinations = BatchExperimentConfig.generate_param_combinations()
        print(f"总参数组合数: {len(combinations)}")
        print("\n前5个参数组合示例:")
        for i, params in enumerate(combinations[:5], 1):
            print(f"\n组合 {i} (实验ID: {params['experiment_id']}):")
            print(f"  数据大小: {params['data_size']} KB")
            print(f"  UE数量: {params['num_UEs']}")
            print(f"  带宽: {params['bandwidth']} kHz")
            print(f"  MEC容量: {params['mec_capacity']} Gcps")
            print(f"  种群大小: {params['population_size']}")
            print(f"  迭代次数: {params['iterations']}")
        return

    # 运行批量实验
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        resume=args.resume
    )

    runner.run_experiments(
        start_id=args.start_id,
        end_id=args.end_id
    )


if __name__ == "__main__":
    main()