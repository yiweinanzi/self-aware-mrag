#!/usr/bin/env python
"""监控全样本实验进度"""

import os
import re
from datetime import datetime

# 查找最新的日志文件
import glob
log_files = glob.glob("/data0/home/zqwang/ACL/full_baselines_experiment_*.log")
if log_files:
    latest_log = max(log_files, key=os.path.getctime)
    print(f"监控日志文件: {latest_log}")
else:
    print("未找到日志文件，等待实验开始...")
    exit()

print("\n" + "="*70)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70 + "\n")

# 监控进度
last_lines = []
method_pattern = re.compile(r'\[(\d{2}:\d{2}:\d{2})\] 开始方法: (.+)')
accuracy_pattern = re.compile(r'\[(\d{2}:\d{2}:\d{2})\] 准确率: ([\d.]+)%')

current_method = None
start_time = None

while True:
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 获取新行
        new_lines = lines[len(last_lines):]

        if new_lines:
            for line in new_lines:
                print(line.strip())

                # 检测方法开始
                m = method_pattern.search(line)
                if m:
                    current_method = m.group(2)
                    start_time = m.group(1)
                    print(f"\n{'='*70}")
                    print(f"开始运行: {current_method}")
                    print(f"开始时间: {start_time}")
                    print(f"{'='*70}\n")

                # 检测准确率结果
                a = accuracy_pattern.search(line)
                if a and "准确率:" in line:
                    accuracy = a.group(2)
                    end_time = a.group(1)
                    print(f"\n{'='*70}")
                    print(f"✅ {current_method} 完成!")
                    print(f"准确率: {accuracy}%")
                    print(f"结束时间: {end_time}")
                    print(f"{'='*70}\n")

            last_lines = lines

        # 检查是否完成
        if "✅ OK-VQA Baselines对比实验完成！" in "".join(lines[-10:]):
            print("\n" + "="*70)
            print("🎉 所有实验已完成！")
            print("="*70)
            break

        # 等待10秒
        import time
        time.sleep(10)

    except Exception as e:
        print(f"监控错误: {e}")
        import time
        time.sleep(10)