# SA-GA 批量实验入口文件
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

# 导入项目模块
from env import ENV
from genetic_algorithm import genetic_algorithm
import config

class ExperimentRunner:
    """批量实验运行器"""

    def __init__(self, output_dir=None):
        """初始化实验运行器

        Args:
            output_dir: 结果输出目录，默认为 ./results_YYYYMMDD_HHMMSS
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./results_{timestamp}"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 创建结果文件
        self.result_file = os.path.join(output_dir, "experiment_results.csv")
        self.detail_file = os.path.join(output_dir, "experiment_details.csv")

        # 初始化结果存储
        self.results = []
        self.details = []

    def create_env_with_config(self, config_dict: Dict[str, Any]) -> ENV:
        """根据配置创建环境实例

        Args:
            config_dict: 配置字典

        Returns:
            配置好的ENV实例
        """
        # 从配置中获取参数
        num_UEs = config_dict['num_UEs']
        bandwidth_kHz = config_dict['bandwidth']
        mec_capacity_ghz = config_dict['mec_capacity']
        min_semantic_factor = config_dict['min_semantic_factor']
        fixed_config = config_dict['fixed']

        # 创建环境实例
        env = ENV(UEs=num_UEs, MECs=1, k=100)

        # 修改环境参数以匹配配置
        # 注意：env.py需要被修改以支持这些参数，这里仅作示例
        # 实际使用时需要根据env.py的具体实现调整

        # 设置带宽（假设env有transmission_bandwidth属性）
        # 单位转换：kHz -> Hz
        env.transmission_bandwidth = bandwidth_kHz * env.kHz

        # 设置MEC计算能力
        # 单位转换：Giga Cycles/s -> GHz
        env.MEC_f = mec_capacity_ghz * env.GHz

        # 设置最小语义提取因子阈值
        # 需要修改env中的semantic_threshold
        if hasattr(env, 'semantic_threshold'):
            env.semantic_threshold = min_semantic_factor
        else:
            # 如果env没有该属性，可以在初始化时传入
            print(f"Warning: env does not have semantic_threshold attribute")

        # 设置任务数据大小范围（如果配置了data_size）
        if 'data_size' in config_dict:
            data_size_kb = config_dict['data_size']
            # 转换为MB
            data_size_mb = data_size_kb / 1024
            # 假设任务大小范围为中心值±10%
            lower_bound = data_size_mb * 0.9
            upper_bound = data_size_mb * 1.1
            # 这里需要在env.reset中调整，暂时跳过

        return env

    def run_single_experiment(self, config_dict: Dict[str, Any], trial_id: int) -> Dict[str, Any]:
        """运行单个实验配置

        Args:
            config_dict: 配置字典
            trial_id: 试验ID

        Returns:
            实验结果字典
        """
        print(f"\n{'='*60}")
        print(f"Running experiment - Trial {trial_id + 1}")
        print(f"Config: {config_dict}")
        print(f"{'='*60}")

        # 创建环境
        try:
            env = self.create_env_with_config(config_dict)
        except Exception as e:
            print(f"Error creating environment: {e}")
            return None

        # 获取固定配置
        fixed_config = config_dict['fixed']

        # 运行多次试验求平均
        trial_energies = []
        best_solutions = []

        for step in range(fixed_config.num_trials):
            try:
                # 重置环境状态
                observation = env.reset(step)

                # 运行遗传算法
                best_solution, final_energy = genetic_algorithm(
                    env=env,
                    observation=observation,
                    a=config_dict['num_UEs'],  # 传入UE数量
                    pop_size=fixed_config.pop_size,
                    generations=fixed_config.generations,
                    early_stop_threshold=fixed_config.early_stop_threshold,
                    patience=fixed_config.early_stop_patience
                )

                trial_energies.append(final_energy)
                best_solutions.append(best_solution)

                print(f"  Trial {step + 1}/{fixed_config.num_trials}: Energy = {final_energy:.4f}")

            except Exception as e:
                print(f"  Error in trial {step + 1}: {e}")
                trial_energies.append(np.nan)

        # 计算统计结果
        valid_energies = [e for e in trial_energies if not np.isnan(e)]
        if not valid_energies:
            print(f"Warning: All trials failed for config {config_dict}")
            return None

        avg_energy = np.mean(valid_energies)
        std_energy = np.std(valid_energies)
        min_energy = np.min(valid_energies)
        max_energy = np.max(valid_energies)

        # 构建结果记录
        result_record = {
            'experiment_id': f"{config_dict.get('data_size', 0)}_{config_dict['num_UEs']}_{config_dict['bandwidth']}_{config_dict['mec_capacity']}_{config_dict['min_semantic_factor']}",
            'data_size_kb': config_dict.get('data_size', 'N/A'),
            'num_UEs': config_dict['num_UEs'],
            'bandwidth_kHz': config_dict['bandwidth'],
            'mec_capacity_ghz': config_dict['mec_capacity'],
            'min_semantic_factor': config_dict['min_semantic_factor'],
            'avg_energy': avg_energy,
            'std_energy': std_energy,
            'min_energy': min_energy,
            'max_energy': max_energy,
            'num_successful_trials': len(valid_energies),
            'total_trials': fixed_config.num_trials,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"  Results: Avg Energy = {avg_energy:.4f} ± {std_energy:.4f}")

        return result_record

    def run_batch_experiments(self, configs: List[Dict[str, Any]]):
        """运行批量实验

        Args:
            configs: 配置列表
        """
        print(f"Starting batch experiments with {len(configs)} configurations")
        print(f"Output directory: {self.output_dir}")

        total_experiments = len(configs)

        for idx, config_dict in enumerate(configs):
            print(f"\n{'#'*60}")
            print(f"Progress: {idx + 1}/{total_experiments} ({((idx + 1) / total_experiments * 100):.1f}%)")
            print(f"{'#'*60}")

            # 运行单个配置的多次试验
            result = self.run_single_experiment(config_dict, idx)

            if result is not None:
                self.results.append(result)

                # 保存中间结果
                self.save_results()

            # 打印进度
            print(f"Completed {idx + 1}/{total_experiments} configurations")

        print(f"\n{'='*60}")
        print(f"All experiments completed!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}")

    def save_results(self):
        """保存结果到文件"""
        if self.results:
            # 保存汇总结果
            df_results = pd.DataFrame(self.results)
            df_results.to_csv(self.result_file, index=False)

            # 保存详细结果（如果有）
            if self.details:
                df_details = pd.DataFrame(self.details)
                df_details.to_csv(self.detail_file, index=False)

    def generate_summary_report(self):
        """生成实验总结报告"""
        if not self.results:
            print("No results to summarize")
            return

        report_file = os.path.join(self.output_dir, "experiment_summary.txt")

        with open(report_file, 'w') as f:
            f.write("SA-GA Batch Experiment Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total experiments: {len(self.results)}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")

            # 按参数分析结果
            df = pd.DataFrame(self.results)

            # 分析每个参数的影响
            parameters = ['data_size_kb', 'num_UEs', 'bandwidth_kHz', 'mec_capacity_ghz', 'min_semantic_factor']

            for param in parameters:
                if param in df.columns:
                    f.write(f"\nAnalysis by {param}:\n")
                    f.write("-" * 30 + "\n")

                    # 分组统计
                    grouped = df.groupby(param)['avg_energy'].agg(['mean', 'std', 'count'])

                    for value, row in grouped.iterrows():
                        f.write(f"  {param} = {value}: ")
                        f.write(f"Avg Energy = {row['mean']:.4f} ± {row['std']:.4f} ")
                        f.write(f"(n={row['count']})\n")

            # 最佳配置
            if 'avg_energy' in df.columns:
                best_idx = df['avg_energy'].idxmin()
                best_config = df.loc[best_idx]

                f.write("\nBest Configuration:\n")
                f.write("-" * 30 + "\n")
                for param in parameters:
                    if param in best_config:
                        f.write(f"  {param}: {best_config[param]}\n")
                f.write(f"  Average Energy: {best_config['avg_energy']:.4f}\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("End of report\n")

        print(f"Summary report saved to: {report_file}")

def main():
    """主函数"""
    print("SA-GA Semantic-Aware Genetic Algorithm Batch Experiment")
    print("=" * 60)

    # 生成所有实验配置
    experiment_configs = config.generate_experiment_configs()

    # 打印配置摘要
    config.print_experiment_summary()

    # 确认是否继续 - 这部分可以根据需要启用或禁用
    # print("\n" + "=" * 60)
    # response = input("Do you want to continue with batch experiments? (yes/no): ")

    # if response.lower() not in ['yes', 'y']:
    #     print("Experiment cancelled.")
    #     return

    # 创建实验运行器
    runner = ExperimentRunner()

    # 运行批量实验
    runner.run_batch_experiments(experiment_configs)

    # 生成总结报告
    runner.generate_summary_report()

    print(f"\nAll experiments completed successfully!")
    print(f"Results saved to: {runner.output_dir}")

if __name__ == "__main__":
    main()