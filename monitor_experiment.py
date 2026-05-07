#!/usr/bin/env python
"""监控100样本实验进度"""

import time
import os
from datetime import datetime

log_file = "/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines/baselines_20251217_123144.log"

print("监控100样本实验进度...")
print("=" * 70)
print(f"日志文件: {log_file}")
print("=" * 70)

while True:
    if os.path.exists(log_file):
        # 读取最后20行
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                print("\n" + "-" * 70)
                print(f"更新时间: {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 70)

                # 显示最后10行
                for line in lines[-10:]:
                    print(line.rstrip())

                # 检查是否有完成的方法
                completed = []
                for line in lines:
                    if "实验完成" in line and "准确率:" in line:
                        # 提取方法名和准确率
                        parts = line.split("准确率:")
                        if len(parts) > 1:
                            method_line = lines[lines.index(line) - 3]
                            if "开始方法:" in method_line:
                                method = method_line.split("开始方法:")[1].strip()
                                accuracy = parts[1].split("%")[0].strip()
                                completed.append(f"{method}: {accuracy}%")

                if completed:
                    print("\n已完成的方法:")
                    for method in completed:
                        print(f"  ✅ {method}")

                # 检查是否所有方法都完成了
                if len(completed) >= 7:
                    print("\n🎉 所有7个方法都已完成！")
                    break
    else:
        print(f"\n{datetime.now().strftime('%H:%M:%S')} - 日志文件尚未创建...")

    # 等待30秒再次检查
    print(f"\n等待30秒后再次检查... ({datetime.now().strftime('%H:%M:%S')})")
    time.sleep(30)