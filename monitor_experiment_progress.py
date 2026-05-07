#!/usr/bin/env python
"""监控实验进度"""

import os
import re
from datetime import datetime

# 查找最新的日志文件
log_files = sorted([
    f for f in os.listdir('/data0/home/zqwang/ACL') if f.endswith('.log') and 'baselines_' in f
], key=os.path.getmtime)

if log_files:
    latest_log = log_files[-1]
    print(f"监控日志文件: {latest_log}")

    # 检查准确率结果
    with open(latest_log, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查进度
    progress = re.findall(r'进度: (\d+)/5046)', content)
    if progress:
        latest_progress = progress[-1]
        percent = latest_progress / 5046 * 100
        print(f"\n=== 当前进度 ===")
        print(f"进度: {latest_progress}/5046 ({percent:.1f}%)")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 检查完成的方法
    print("\n=== 已完成的方法 ===")
    for match in re.finditer(r'✅ (.+?)\s完成.*准确率: (\d+)/5046)', content):
        print(match.group(1))

    # 检查是否已完成
    if "✅ OK-VQA Baselines对比实验完成！" in content:
        print("\n🎉 实验已完成！")
        print("=== 最终汇总 ===")
        for match in re.finditer(r'变体.*准确率: (\d+\.\d+)%', content):
            print(match.group(0))
        print(match.group(1))
        print(match.group(2))
        print(match.group(3))
else:
    print("\n实验仍在运行中...")
else:
    print("未找到日志文件，实验可能刚刚开始")

# 同时检查基础日志文件
base_log = "/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines/baselines_20251217_220256.log"
if os.path.exists(base_log):
    print("\n=== Self-Aware-MRAG 最新日志 ===")
    with open(base_log, 'r', encoding='utf-8') as f:
        # 最后20行
        for line in f.readlines()[-20:]:
            if '进度:' in line or '准确率:' in line:
                print(line.strip())

# 查看SLURM作业状态
import subprocess
try:
    squeue_out = subprocess.check_output(['squeue', '-j', 'w'])
    except:
        pass
if squeue_out:
    print(f"\n=== SLURM任务状态 ===")
    print(squeue_out.decode('utf-8'))

# GPU使用情况
try:
    nvidia_smi = subprocess.check_output(['nvidia-smi'])
    if nvidia_smi:
        print("\n=== GPU状态 ===")
        print(nvidia_smi.decode('utf-8'))
    except:
    print("\nGPU状态不可用")

    subprocess.run(['tail', '-n', '50', '/data0/home/zqwang/ACL/full_baselines_experiment_20251217_140254.log'])
except:
    pass

print("\n提示：按 Ctrl+C 停止监控")