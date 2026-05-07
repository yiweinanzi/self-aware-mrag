#!/usr/bin/env python3
"""
快速测试所有7个方法的导入和基本功能
"""

import sys
import os

# 设置路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

print("="*60)
print("测试所有7个baseline方法的导入")
print("="*60)

methods = [
    ('Self-Aware-MRAG', 'experiments.baselines.selfaware_multimodal', 'SelfAwareMultimodalPipeline'),
    ('MuRAG', 'experiments.baselines.murag_enhanced', 'MuRAGEnhanced'),
    ('VisRAG', 'experiments.baselines.visrag_enhanced', 'VisRAGEnhanced'),
    ('ViDoRAG', 'experiments.baselines.vidorag_pipeline', 'ViDoRAGPipeline'),
    ('RagVL', 'experiments.baselines.ragvl_enhanced', 'RagVLEnhanced'),
    ('SAM-RAG', 'experiments.baselines.samrag_adapted', 'SAMRAGAdapted'),
    ('mR²AG', 'experiments.baselines.mr2ag_enhanced', 'MR2AGEnhanced')
]

success_count = 0
fail_count = 0

for name, module_path, class_name in methods:
    print(f"\n测试 {name}...")
    try:
        module = __import__(module_path, fromlist=[class_name])
        pipeline_class = getattr(module, class_name)
        print(f"  ✅ {class_name} 导入成功")
        success_count += 1

        # 测试是否有 run_single 方法
        if hasattr(pipeline_class, 'run_single'):
            print(f"  ✅ 有 run_single 方法")
        else:
            print(f"  ⚠️  没有 run_single 方法")

    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        fail_count += 1

print("\n" + "="*60)
print(f"测试结果: {success_count} 成功, {fail_count} 失败")
print("="*60)

# 检查修复情况
print("\n关键修复验证:")
print("1. max_new_tokens=20 - 需要运行时验证")
print("2. correct 字段计算 - 需要运行时验证")
print("3. Qwen3VL API 调用 - 需要运行时验证")
print("4. 返回格式 - 需要运行时验证")