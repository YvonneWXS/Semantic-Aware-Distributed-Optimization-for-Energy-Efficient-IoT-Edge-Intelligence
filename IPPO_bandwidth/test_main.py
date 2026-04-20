#!/usr/bin/env python
"""
测试main.py批量实验入口文件

功能：
1. 测试参数组合生成
2. 测试实验状态管理
3. 测试汇总报告生成
4. 测试命令行参数解析
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from main import ExperimentRunner
from config import BatchExperimentConfig


def test_batch_experiment_config():
    """测试BatchExperimentConfig类"""
    print("测试BatchExperimentConfig类...")

    # 测试参数组合生成
    combinations = BatchExperimentConfig.generate_param_combinations()
    print(f"  生成的参数组合数: {len(combinations)}")

    # 测试第一个参数组合
    if combinations:
        first_combo = combinations[0]
        print(f"  第一个参数组合:")
        print(f"    实验ID: {first_combo['experiment_id']}")
        print(f"    数据大小: {first_combo['data_size']} KB")
        print(f"    UE数量: {first_combo['num_UEs']}")
        print(f"    带宽: {first_combo['bandwidth']} kHz")
        print(f"    MEC容量: {first_combo['mec_capacity']} Gcps")

        # 测试单位转换
        assert 'data_size_bits' in first_combo, "缺少data_size_bits字段"
        assert 'bandwidth_hz' in first_combo, "缺少bandwidth_hz字段"
        assert 'mec_capacity_cps' in first_combo, "缺少mec_capacity_cps字段"
        print(f"    单位转换测试通过")

    # 测试实验总数计算
    total = BatchExperimentConfig.get_total_experiments()
    print(f"  总实验数计算: {total}")
    assert total == len(combinations), "总实验数计算错误"

    # 测试参数验证
    if combinations:
        is_valid, message = BatchExperimentConfig.validate_params(combinations[0])
        print(f"  参数验证测试: {is_valid}, {message}")
        assert is_valid, "参数验证失败"

    print("  BatchExperimentConfig测试通过 [OK]")


def test_experiment_runner_basic():
    """测试ExperimentRunner基本功能"""
    print("\n测试ExperimentRunner基本功能...")

    # 使用临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir, num_workers=1, resume=False)

        # 测试目录创建
        assert Path(tmpdir).exists(), "输出目录未创建"
        print(f"  输出目录创建测试通过: {tmpdir}")

        # 测试状态文件创建
        status_file = Path(tmpdir) / "experiment_status.json"
        assert status_file.exists(), "状态文件未创建"
        print(f"  状态文件创建测试通过: {status_file}")

        # 测试参数组合加载
        assert len(runner.param_combinations) > 0, "参数组合未加载"
        print(f"  参数组合加载测试通过: {len(runner.param_combinations)}个组合")

        # 测试实验状态管理
        initial_status = runner.experiment_status
        assert 'experiments' in initial_status, "实验状态格式错误"
        assert 'last_update' in initial_status, "实验状态缺少最后更新时间"
        print(f"  实验状态管理测试通过")

    print("  ExperimentRunner基本功能测试通过 [OK]")


def test_experiment_runner_methods():
    """测试ExperimentRunner方法"""
    print("\n测试ExperimentRunner方法...")

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir, num_workers=1, resume=False)

        # 测试实验目录创建
        exp_dir = runner._create_experiment_dir("test_exp_0001")
        assert exp_dir.exists(), "实验目录未创建"
        print(f"  实验目录创建测试通过: {exp_dir}")

        # 测试状态保存
        runner._save_experiment_status()
        status_file = Path(tmpdir) / "experiment_status.json"
        assert status_file.stat().st_size > 0, "状态文件为空"
        print(f"  状态保存测试通过")

        # 测试结果保存和加载
        test_results = [{'test': 'data'}]
        runner._save_results(test_results)
        loaded_results = runner._load_results()
        assert loaded_results == test_results, "结果保存/加载不一致"
        print(f"  结果保存/加载测试通过")

    print("  ExperimentRunner方法测试通过 [OK]")


def test_summary_generation():
    """测试汇总报告生成"""
    print("\n测试汇总报告生成...")

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(output_dir=tmpdir, num_workers=1, resume=False)

        # 创建测试结果
        test_results = [
            {
                'experiment_id': 'exp_0001',
                'params': {
                    'data_size': 128,
                    'num_UEs': 5,
                    'bandwidth': 750,
                    'mec_capacity': 10.0
                },
                'status': 'completed',
                'duration': 120.5,
                'metrics': {
                    'final_average_reward': 0.85,
                    'final_energy': 150.2,
                    'final_delay': 0.45
                }
            },
            {
                'experiment_id': 'exp_0002',
                'params': {
                    'data_size': 256,
                    'num_UEs': 10,
                    'bandwidth': 1000,
                    'mec_capacity': 12.5
                },
                'status': 'completed',
                'duration': 180.3,
                'metrics': {
                    'final_average_reward': 0.72,
                    'final_energy': 220.8,
                    'final_delay': 0.62
                }
            }
        ]

        # 生成汇总报告
        runner.generate_summary_report(test_results)

        # 检查生成的文件
        summary_file = Path(tmpdir) / "experiment_summary.csv"
        stats_file = Path(tmpdir) / "experiment_stats.json"

        assert summary_file.exists(), "汇总CSV文件未创建"
        assert stats_file.exists(), "统计JSON文件未创建"

        # 检查JSON文件内容
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        assert 'total_experiments' in stats, "统计缺少total_experiments字段"
        assert 'completed_experiments' in stats, "统计缺少completed_experiments字段"

        print(f"  汇总报告生成测试通过")
        print(f"    CSV文件: {summary_file}")
        print(f"    JSON统计: {stats_file}")

    print("  汇总报告生成测试通过 [OK]")


def test_command_line_interface():
    """测试命令行接口"""
    print("\n测试命令行接口...")

    import subprocess

    # 测试帮助信息
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        cwd=project_root
    )

    assert result.returncode == 0, "帮助命令失败"
    assert "无线通信与边缘计算批量实验入口" in result.stdout, "帮助信息不正确"
    print(f"  帮助命令测试通过")

    # 测试参数列表
    result = subprocess.run(
        [sys.executable, "main.py", "--list_params"],
        capture_output=True,
        text=True,
        cwd=project_root
    )

    assert result.returncode == 0, "参数列表命令失败"
    assert "总参数组合数" in result.stdout, "参数列表输出不正确"
    print(f"  参数列表命令测试通过")

    # 测试摘要显示
    result = subprocess.run(
        [sys.executable, "main.py", "--summary"],
        capture_output=True,
        text=True,
        cwd=project_root
    )

    assert result.returncode == 0, "摘要命令失败"
    assert "实验配置摘要" in result.stdout, "摘要输出不正确"
    print(f"  摘要命令测试通过")

    print("  命令行接口测试通过 [OK]")


def main():
    """运行所有测试"""
    print("=" * 80)
    print("测试main.py批量实验入口文件")
    print("=" * 80)

    try:
        test_batch_experiment_config()
        test_experiment_runner_basic()
        test_experiment_runner_methods()
        test_summary_generation()
        test_command_line_interface()

        print("\n" + "=" * 80)
        print("所有测试通过! [OK]")
        print("=" * 80)

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()