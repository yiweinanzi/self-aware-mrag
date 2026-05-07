#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验监控脚本
Monitor Ablation Experiment Progress
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

def check_job_status():
    """检查SLURM作业状态"""
    try:
        result = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', 'zqwang')],
            capture_output=True,
            text=True
        )
        return result.stdout
    except:
        return "无法获取作业状态"

def check_latest_log():
    """检查最新的实验日志"""
    log_dir = Path("/data0/home/zqwang/ACL/FlashRAG/experiments/logs")

    # 查找最新的ablation日志文件
    log_files = list(log_dir.glob("srun_main_*.out"))
    if not log_files:
        return None, "未找到日志文件"

    latest_log = max(log_files, key=os.path.getmtime)

    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 返回最后50行
        return str(latest_log), ''.join(lines[-50:])
    except Exception as e:
        return str(latest_log), f"读取日志失败: {e}"

def check_results():
    """检查实验结果目录"""
    results_dir = Path("/data0/home/zqwang/ACL/FlashRAG/experiments/results_ablation_okvqa")

    if not results_dir.exists():
        return "结果目录不存在"

    items = list(results_dir.rglob("*"))
    return f"结果文件数: {len(items)}\n最新文件: {get_latest_files(items, 5)}"

def get_latest_files(files, n=5):
    """获取最新的n个文件"""
    sorted_files = sorted(files, key=os.path.getmtime, reverse=True)
    return "\n".join([f"  {f.relative_to(files[0].parents[2])}" for f in sorted_files[:n]])

def monitor():
    """主监控函数"""
    print("="*80)
    print("消融实验监控")
    print("="*80)
    print(f"开始监控: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        while True:
            # 清屏
            os.system('clear' if os.name == 'posix' else 'cls')

            print("="*80)
            print("消融实验实时监控")
            print("="*80)
            print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()

            # 显示作业状态
            print("📊 SLURM作业状态:")
            job_status = check_job_status()
            print(job_status)
            print()

            # 显示日志
            print("📋 最新日志 (最后50行):")
            log_file, log_content = check_latest_log()
            if log_file:
                print(f"日志文件: {log_file}")
                print("-" * 60)
                print(log_content)
            print()

            # 显示结果
            print("📁 实验结果:")
            results_info = check_results()
            print(results_info)
            print()

            # 显示GPU状态（如果在节点上）
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("🎮 GPU状态:")
                    print("-" * 60)
                    print(result.stdout)
            except:
                print("🎮 GPU状态: 无法获取")

            print("-" * 80)
            print("按 Ctrl+C 退出监控")
            print(f"下次更新: 30秒后 ({datetime.now()})")

            # 等待30秒
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n监控停止")

if __name__ == '__main__':
    monitor()