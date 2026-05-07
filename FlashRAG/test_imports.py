#!/usr/bin/env python
"""测试导入是否正常"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

try:
    print("测试统一数据集加载器...")
    from flashrag.dataset.unified_dataset_loader import load_unified_dataset
    print("✓ 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")

try:
    print("\n测试统一评测器...")
    from flashrag.evaluator.unified_evaluator import evaluate_unified
    print("✓ 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")

try:
    print("\n测试原始脚本导入...")
    # 只导入配置部分
    exec(open('/data0/home/zqwang/ACL/FlashRAG/experiments/run_all_baselines_100samples.py').read().split('# 导入Baseline类')[0])
    print("✓ 基础导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")

print("\n完成！")