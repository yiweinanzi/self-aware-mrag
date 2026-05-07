#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
归档过时实验代码脚本
Archive Old Experiment Code Scripts

基于以下标准归档过时的代码：
1. 无法运行或有严重错误
2. 准确率明显低于最佳版本
3. 功能已被更好的版本替代
4. 配置混乱或难以维护

保留的代码：
✅ run_unified_ablation.py (新的统一版本)
✅ run_real_model_ablation.py (功能最完整，作为参考)
✅ run_stable_4gpu_ablation.py (稳定的4GPU版本)
✅ run_2gpu_real_qwen3vl.py (成功的2GPU版本)
✅ baselines/ 目录 (baseline对比方法)
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def archive_directory():
    """归档目录创建"""
    archive_dir = Path("/data0/home/zqwang/ACL/FlashRAG/experiments/archived")
    archive_dir.mkdir(exist_ok=True)

    # 按日期创建子目录
    date_str = datetime.now().strftime("%Y%m%d")
    today_archive = archive_dir / date_str
    today_archive.mkdir(exist_ok=True)

    return today_archive

def get_files_to_archive():
    """确定需要归档的文件列表"""

    # 明确保留的文件（不归档）
    keep_files = {
        # 核心实验脚本
        'run_unified_ablation.py',  # 新的统一版本
        'run_real_model_ablation.py',  # 功能最完整
        'run_stable_4gpu_ablation.py',  # 稳定的4GPU版本
        'run_2gpu_real_qwen3vl.py',  # 成功的2GPU版本

        # 基础设施
        'README.md',
        'README_EXPERIMENTS.md',
        'README_MRAGBENCH.md',
        'monitor_experiment.py',
        'test_evaluation_metrics.py',
        'test_config.json',
        'test_sam_rag.py',
        'quick_test_run.py',

        # Shell脚本（可能有用）
        'setup_environment.sh',
        'test_gpu.sh',
    }

    # 明确保留的目录
    keep_dirs = {
        'baselines/',  # baseline对比方法
        'configs/',    # 配置文件
        '__pycache__/',  # Python缓存
        'logs/',       # 日志文件
    }

    # 需要归档的文件（基于分析和经验）
    archive_files = [
        'run_ablation_multigpu.py',      # 复杂的多GPU实现，已被更稳定的版本替代
        'run_final_ablation.py',         # 功能不完整
        'run_fixed_ablation.py',         # 修复版本，已被统一版本替代
        'run_fixed_ablation_100samples.py',  # 临时测试版本
        'run_ablation_simple.py',        # 过于简化
        'run_real_ablation_simple.py',   # 功能不完整
        'run_ablation_study_okvqa.py',   # 配置复杂，难以维护
        'run_all_baselines_100samples.py',  # 特定实验，已完成
        'run_all_real_qwen3vl_4gpu.py',     # 复杂的4GPU实现
        'run_real_model_ablation_4gpu.py',   # 已被更稳定的版本替代
        'run_real_model_ablation_optimized_4gpu.py',  # 优化版本，但不够稳定
    ]

    # 需要归档的Shell脚本
    archive_shell_scripts = [
        'run_ablation_final.sh',
        'run_ablation_srun.sh',
        'run_ablation_with_logs.sh',
        'run_gpu_experiment.sh',
        'run_ablation_2gpu.sh',
        'run_ablation_2gpu_clean.sh',
        'run_ablation_simple_2gpu.sh',
        'run_final_2gpu.sh',
        'monitor_experiment.sh',
    ]

    archive_files.extend(archive_shell_scripts)

    return archive_files, keep_files, keep_dirs

def archive_experiments():
    """执行归档操作"""
    print("🔍 开始归档过时的实验代码...")

    experiments_dir = Path("/data0/home/zqwang/ACL/FlashRAG/experiments")
    archive_dir = archive_directory()

    archive_files, keep_files, keep_dirs = get_files_to_archive()

    archived_count = 0
    kept_count = 0

    # 检查每个文件
    for file_path in experiments_dir.iterdir():
        if file_path.is_file():
            filename = file_path.name

            if filename in keep_files:
                print(f"✅ 保留: {filename}")
                kept_count += 1
            elif filename in archive_files:
                # 移动到归档目录
                try:
                    shutil.move(str(file_path), str(archive_dir / filename))
                    print(f"📦 归档: {filename}")
                    archived_count += 1
                except Exception as e:
                    print(f"❌ 归档失败 {filename}: {e}")
            else:
                print(f"❓ 未知文件: {filename}")

    # 归档过时的结果目录（基于准确率分析）
    results_dirs_to_archive = [
        'results_ablation_simple',      # 准确率较低
        'results_real_ablation',        # 功能不完整
        'results_final_ablation',       # 已被更好的结果替代
        'results_optimized_4gpu',       # 不够稳定
        'results_real_model_ablation_4gpu',  # 已被stable版本替代
        'results_baseline_comparison_100_wiki3m',  # 早期实验
    ]

    for dir_name in results_dirs_to_archive:
        dir_path = experiments_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.move(str(dir_path), str(archive_dir / dir_name))
                print(f"📦 归档目录: {dir_name}")
                archived_count += 1
            except Exception as e:
                print(f"❌ 归档目录失败 {dir_name}: {e}")

    # 保留的结果目录
    keep_results_dirs = [
        'results_real_model_ablation',  # 最新且准确率较高
        'results_stable_4gpu',          # 4GPU稳定结果
        'results_all_real_qwen3vl',     # 全真实模型结果
    ]

    for dir_name in keep_results_dirs:
        dir_path = experiments_dir / dir_name
        if dir_path.exists():
            print(f"✅ 保留结果目录: {dir_name}")
            kept_count += 1

    # 生成归档报告
    report_file = archive_dir / "archive_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"实验代码归档报告\n")
        f.write(f"归档时间: {datetime.now().isoformat()}\n")
        f.write(f"归档目录: {archive_dir}\n\n")

        f.write(f"归档的文件数量: {archived_count}\n")
        f.write(f"保留的文件数量: {kept_count}\n\n")

        f.write("归档标准:\n")
        f.write("1. 无法运行或有严重错误\n")
        f.write("2. 准确率明显低于最佳版本\n")
        f.write("3. 功能已被更好的版本替代\n")
        f.write("4. 配置混乱或难以维护\n\n")

        f.write("保留的核心文件:\n")
        for filename in sorted(keep_files):
            f.write(f"- {filename}\n")

        f.write("\n保留的结果目录:\n")
        for dirname in sorted(keep_results_dirs):
            f.write(f"- {dirname}\n")

    print(f"\n📋 归档完成!")
    print(f"   归档文件数: {archived_count}")
    print(f"   保留文件数: {kept_count}")
    print(f"   归档目录: {archive_dir}")
    print(f"   归档报告: {report_file}")

if __name__ == "__main__":
    archive_experiments()