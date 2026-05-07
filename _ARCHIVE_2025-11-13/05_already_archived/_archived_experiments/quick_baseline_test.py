#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速Baseline对比测试
100样本 + 300万Wikipedia
"""
import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

print("=" * 80)
print("🏆 Baseline对比实验 - 快速版")
print("=" * 80)
print("配置: 100样本 + 300万Wikipedia")
print("Baseline: MuRAG (简化版)")
print("Our Method: 使用68.90%的最佳配置")
print("=" * 80)

# 测试基本导入
try:
    from flashrag.dataset.okvqa_dataset_lazy import OKVQADataset
    from flashrag.modules.mllm_wrapper import LLaVAWrapper
    print("\n✅ 模块导入成功")
except Exception as e:
    print(f"\n❌ 导入失败: {e}")
    sys.exit(1)

# 加载数据集
print("\n加载数据集...")
config = {'data_dir': 'flashrag/data/VQA', 'max_samples': 100}
dataset = OKVQADataset(config)
print(f"✅ 数据集加载完成: {len(dataset)} 样本")

print("\n准备运行实验...")
print("预计时间: 约10-15分钟")
print("\n按Ctrl+C取消，或等待自动完成...")
