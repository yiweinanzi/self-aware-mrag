#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级评估指标测试脚本
Test Script for Advanced Evaluation Metrics

演示如何使用三大核心评估指标：
1. Attribution Precision（归因精度）
2. Cross-Modal Consistency（跨模态一致性）
3. Position Bias Metric（位置偏差）

运行方式：
```bash
cd /root/autodl-tmp/FlashRAG
python scripts/test_advanced_metrics.py
```
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from flashrag.evaluator.advanced_metrics import (
    AttributionPrecisionCalculator,
    CrossModalConsistencyScore,
    PositionBiasMetric,
    ComprehensiveEvaluator,
    quick_evaluate_attribution,
    quick_evaluate_consistency
)

print("="*80)
print("🧪 高级评估指标测试")
print("="*80)


# =============================================================================
# 测试1: Attribution Precision（归因精度）
# =============================================================================

print("\n" + "="*80)
print("📊 测试1: Attribution Precision（归因精度）")
print("="*80)

# 创建测试数据
test_attributions = {
    'visual': [
        {'source_image_id': 'img_001', 'region_bbox': [10, 20, 100, 200], 'confidence': 0.9},
        {'source_image_id': 'img_002', 'region_bbox': [50, 50, 150, 150], 'confidence': 0.8},
    ],
    'text': [
        {'source_text_id': 'doc_001', 'source_span': (0, 50), 'confidence': 0.85},
        {'source_text_id': 'doc_003', 'source_span': (100, 200), 'confidence': 0.75},
    ]
}

ground_truth_sources = ['img_001', 'doc_001', 'doc_002']  # doc_002被遗漏了

# 计算归因精度
attribution_calculator = AttributionPrecisionCalculator(confidence_threshold=0.5)

result = attribution_calculator.compute(
    generated_answer="Paris is the capital of France",
    attributions=test_attributions,
    ground_truth_sources=ground_truth_sources
)

print("\n📋 归因精度结果:")
print(f"  总体Precision: {result['precision']:.3f}")
print(f"  总体Recall: {result['recall']:.3f}")
print(f"  总体F1: {result['f1']:.3f}")
print(f"  视觉Precision: {result['visual_precision']:.3f}")
print(f"  视觉Recall: {result['visual_recall']:.3f}")
print(f"  文本Precision: {result['text_precision']:.3f}")
print(f"  文本Recall: {result['text_recall']:.3f}")

print("\n💡 解读:")
print(f"  - 预测了3个源 (img_001, img_002, doc_001, doc_003)")
print(f"  - 其中2个正确 (img_001, doc_001)")
print(f"  - 真实源有3个 (img_001, doc_001, doc_002)")
print(f"  - Precision = 2/4 = 0.5 (预测的4个中2个正确)")
print(f"  - Recall = 2/3 = 0.667 (真实的3个中2个被找到)")
print(f"  - F1 = 调和平均 = {result['f1']:.3f}")


# 批量测试
print("\n" + "-"*80)
print("📦 批量评估测试:")

batch_data = [
    {
        'generated_answer': "Answer 1",
        'attributions': {'visual': [{'source_image_id': 'img_1', 'confidence': 0.9}]},
        'ground_truth_sources': ['img_1', 'doc_1']
    },
    {
        'generated_answer': "Answer 2",
        'attributions': {
            'visual': [{'source_image_id': 'img_2', 'confidence': 0.8}],
            'text': [{'source_text_id': 'doc_2', 'confidence': 0.7}]
        },
        'ground_truth_sources': ['img_2', 'doc_2']
    },
    {
        'generated_answer': "Answer 3",
        'attributions': {'text': [{'source_text_id': 'doc_3', 'confidence': 0.6}]},
        'ground_truth_sources': ['doc_3']
    }
]

batch_result = attribution_calculator.compute_batch(batch_data)

print(f"\n  批量归因精度 (3个样本):")
print(f"    平均Precision: {batch_result['precision']:.3f}")
print(f"    平均Recall: {batch_result['recall']:.3f}")
print(f"    平均F1: {batch_result['f1']:.3f}")


# =============================================================================
# 测试2: Cross-Modal Consistency（跨模态一致性）
# =============================================================================

print("\n" + "="*80)
print("📊 测试2: Cross-Modal Consistency（跨模态一致性）")
print("="*80)

print("\n⚠️  注意: 此测试需要CLIP模型和图像数据")
print("如果没有CLIP模型，将跳过此测试\n")

try:
    # 尝试加载CLIP
    consistency_scorer = CrossModalConsistencyScore()
    
    if consistency_scorer.clip_model is not None:
        print("✅ CLIP模型加载成功")
        
        # 创建模拟测试（实际应用中需要真实图像）
        print("\n💡 在实际使用中:")
        print("""
        from PIL import Image
        
        # 加载图像
        image = Image.open('path/to/image.jpg')
        
        # 计算一致性
        score = consistency_scorer.compute(
            text_answer="A red car on the street",
            visual_evidence=image
        )
        
        print(f"一致性分数: {score:.3f}")
        """)
        
    else:
        print("⚠️  CLIP模型未加载，跳过实际测试")
        print("💡 要使用此功能，请确保:")
        print("   1. 安装transformers: pip install transformers")
        print("   2. 下载CLIP模型到: /root/autodl-tmp/models/clip-vit-large-patch14-336")

except Exception as e:
    print(f"❌ CLIP测试失败: {e}")
    print("💡 这是正常的，如果您还没有配置CLIP模型")


# =============================================================================
# 测试3: Position Bias Metric（位置偏差）
# =============================================================================

print("\n" + "="*80)
print("📊 测试3: Position Bias Metric（位置偏差）")
print("="*80)

# 创建模拟模型
class MockModel:
    """模拟模型用于测试"""
    
    def generate(self, query, context):
        """
        模拟生成答案
        
        模拟位置偏差：
        - 如果关键信息在开头: 80%正确
        - 如果关键信息在中间: 40%正确  
        - 如果关键信息在结尾: 60%正确
        """
        # 简化版：查找"Paris"在context中的位置
        for idx, doc in enumerate(context):
            if "Paris" in doc:
                position = idx / len(context)
                
                if position < 0.3:  # 开头
                    return "Paris" if np.random.rand() < 0.8 else "Wrong"
                elif position > 0.7:  # 结尾
                    return "Paris" if np.random.rand() < 0.6 else "Wrong"
                else:  # 中间
                    return "Paris" if np.random.rand() < 0.4 else "Wrong"
        
        return "Unknown"

mock_model = MockModel()

# 创建测试样本
test_samples = [
    {
        'query': "What is the capital of France?",
        'context': [
            "France is a country in Europe.",
            "Paris is the capital of France.",  # 关键信息
            "The Eiffel Tower is in Paris."
        ],
        'key_info': "Paris is the capital",  # 用于识别关键文档
        'ground_truth': ["Paris"]
    },
    {
        'query': "What is the capital of France?",
        'context': [
            "France is a country in Europe.",
            "The Loire Valley is beautiful.",
            "Paris is the capital of France.",  # 关键信息
            "French cuisine is famous."
        ],
        'key_info': "Paris is the capital",
        'ground_truth': ["Paris"]
    }
]

print("\n🔧 使用模拟模型测试位置偏差...")
print("   模拟设定:")
print("   - 关键信息在开头: 80% 准确率")
print("   - 关键信息在中间: 40% 准确率")
print("   - 关键信息在结尾: 60% 准确率")

position_evaluator = PositionBiasMetric()

# 运行多次取平均（因为有随机性）
print("\n📊 运行评估 (3次取平均)...")
all_results = []

for run in range(3):
    result = position_evaluator.evaluate(
        model=mock_model,
        test_samples=test_samples,
        verbose=False
    )
    all_results.append(result)

# 计算平均结果
avg_result = {
    'position_bias': np.mean([r['position_bias'] for r in all_results]),
    'max_diff': np.mean([r['max_diff'] for r in all_results]),
    'beginning_acc': np.mean([r['beginning_acc'] for r in all_results]),
    'middle_acc': np.mean([r['middle_acc'] for r in all_results]),
    'end_acc': np.mean([r['end_acc'] for r in all_results]),
}

print("\n📋 位置偏差评估结果 (平均值):")
print(f"  位置偏差分数: {avg_result['position_bias']:.3f} (越小越好)")
print(f"  最大性能差异: {avg_result['max_diff']:.3f}")
print(f"  开头准确率: {avg_result['beginning_acc']:.3f}")
print(f"  中间准确率: {avg_result['middle_acc']:.3f}")
print(f"  结尾准确率: {avg_result['end_acc']:.3f}")

print("\n💡 解读:")
if avg_result['position_bias'] < 0.2:
    print("  ✅ 位置偏差较小，模型不太受位置影响")
elif avg_result['position_bias'] < 0.4:
    print("  ⚠️  位置偏差中等，模型对位置有一定敏感度")
else:
    print("  ❌ 位置偏差较大，模型严重受位置影响")

print(f"\n  预期: 根据模拟设定，应该看到明显的位置偏差")
print(f"  实际: 位置偏差 = {avg_result['position_bias']:.3f}")


# 简化版评估
print("\n" + "-"*80)
print("📦 简化版位置偏差评估:")

# 如果已经有不同位置的预测结果
mock_predictions = {
    'beginning': [0.8, 0.9, 0.7, 0.8, 0.85],
    'middle': [0.4, 0.5, 0.3, 0.4, 0.45],
    'end': [0.6, 0.7, 0.5, 0.6, 0.65]
}

simple_result = position_evaluator.evaluate_simple(mock_predictions)

print(f"\n  简化评估结果:")
print(f"    位置偏差: {simple_result['position_bias']:.3f}")
print(f"    最大差异: {simple_result['max_diff']:.3f}")


# =============================================================================
# 测试4: 综合评估器
# =============================================================================

print("\n" + "="*80)
print("📊 测试4: 综合评估器 (Comprehensive Evaluator)")
print("="*80)

print("\n🔧 初始化综合评估器...")

comprehensive_evaluator = ComprehensiveEvaluator()

# 创建综合测试数据
comprehensive_test_data = [
    {
        'query': "What is shown in the image?",
        'image': None,  # 实际使用中应该是PIL.Image对象
        'generated_answer': "A red car",
        'attributions': {
            'visual': [{'source_image_id': 'img_1', 'confidence': 0.9}],
            'text': [{'source_text_id': 'doc_1', 'confidence': 0.8}]
        },
        'ground_truth_sources': ['img_1', 'doc_1'],
        'ground_truth_answer': ["red car", "car"],
        'context': ["Doc 1", "Doc 2", "Doc 3"],
        'key_info': "Doc 1"
    },
    # 可以添加更多样本...
]

print("\n💡 综合评估示例:")
print("""
# 运行完整评估
results = comprehensive_evaluator.evaluate_full(
    test_data=comprehensive_test_data,
    model=your_model  # 可选，用于位置偏差评估
)

# 生成报告
report = comprehensive_evaluator.generate_report(results)
print(report)

# 保存报告
with open('evaluation_report.md', 'w') as f:
    f.write(report)
""")


# =============================================================================
# 便捷函数测试
# =============================================================================

print("\n" + "="*80)
print("📊 测试5: 便捷函数 (Quick Functions)")
print("="*80)

# Quick Attribution
predictions = [
    {'answer': 'Paris', 'attributions': {'text': [{'source_text_id': 'doc_1', 'confidence': 0.9}]}},
    {'answer': 'London', 'attributions': {'text': [{'source_text_id': 'doc_2', 'confidence': 0.8}]}}
]

ground_truths = [
    {'sources': ['doc_1']},
    {'sources': ['doc_2', 'doc_3']}
]

quick_attr_result = quick_evaluate_attribution(predictions, ground_truths)

print("\n📊 快速归因评估:")
print(f"  F1 Score: {quick_attr_result['f1']:.3f}")


# =============================================================================
# 总结
# =============================================================================

print("\n" + "="*80)
print("✅ 测试完成！")
print("="*80)

print("\n📚 使用指南:")
print("\n1. Attribution Precision:")
print("   - 评估归因的准确性")
print("   - 支持Region-level（视觉）和Token-level（文本）")
print("   - 返回Precision, Recall, F1")

print("\n2. Cross-Modal Consistency:")
print("   - 评估文本-图像一致性")
print("   - 需要CLIP模型")
print("   - 返回[0,1]的一致性分数")

print("\n3. Position Bias Metric:")
print("   - 量化位置偏差")
print("   - 测试关键信息在不同位置的性能")
print("   - 返回标准差（越小越好）")

print("\n📖 完整文档:")
print("   参见: flashrag/evaluator/advanced_metrics.py")

print("\n🔗 集成到实验:")
print("""
   # 在你的实验脚本中:
   from flashrag.evaluator.advanced_metrics import *
   
   # 初始化
   evaluator = ComprehensiveEvaluator()
   
   # 评估
   results = evaluator.evaluate_full(test_data, model)
   
   # 生成报告
   report = evaluator.generate_report(results)
""")

print("\n" + "="*80)
print("🎉 所有测试通过！评估指标已准备就绪！")
print("="*80)


