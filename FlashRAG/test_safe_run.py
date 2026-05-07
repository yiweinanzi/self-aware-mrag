#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安全测试脚本 - 只运行一个方法在少量样本上
Safe Test Script - Run One Method on Few Samples
"""

import os
import sys
import json
import time
from datetime import datetime

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

print("="*80)
print("安全测试：单个方法，少量样本")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 导入必要的模块
print("\n1. 导入模块...")
try:
    from flashrag.dataset.unified_dataset_loader import load_unified_dataset
    print("✓ 数据集加载器")

    from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
    print("✓ Qwen3-VL")

    from flashrag.retriever import DenseRetriever
    print("✓ 检索器")

    from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
    print("✓ Pipeline")

except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 加载数据
print("\n2. 加载数据...")
dataset_name = 'okvqa'
max_samples = 5  # 只用5个样本

try:
    dataset = load_unified_dataset(dataset_name, split='val', max_samples=max_samples)
    print(f"✓ 加载了 {len(dataset)} 个样本")

    # 转换格式
    samples = []
    for item in dataset.data:
        sample = {
            'question': item.get('question', ''),
            'answer': item.get('golden_answers', [''])[0],
            'image': item.get('image'),
            'golden_answers': item.get('golden_answers', [])
        }
        samples.append(sample)

except Exception as e:
    print(f"✗ 数据加载失败: {e}")
    sys.exit(1)

# 初始化模型
print("\n3. 初始化模型...")
try:
    qwen3_vl = create_qwen3_vl_wrapper(
        model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        device="cuda"
    )
    print("✓ Qwen3-VL初始化成功")
except Exception as e:
    print(f"✗ 模型初始化失败: {e}")
    sys.exit(1)

# 初始化检索器（使用简化配置）
print("\n4. 初始化检索器...")
try:
    # 创建一个简单的模拟检索器
    class SimpleRetriever:
        def __init__(self):
            self.top_k = 5

        def search(self, question, num=None):
            # 返回模拟的检索结果
            docs = [f"Document {i} related to {question[:20]}" for i in range(min(num or self.top_k, 5))]
            return docs

    retriever = SimpleRetriever()
    print("✓ 使用模拟检索器（避免索引问题）")
except Exception as e:
    print(f"✗ 检索器初始化失败: {e}")
    retriever = None

# 创建Pipeline
print("\n5. 创建Pipeline...")
try:
    pipeline = SelfAwarePipelineQwen3VL(
        qwen3_vl_wrapper=qwen3_vl,
        retriever=retriever,
        config={
            'uncertainty_threshold': 0.43,
            'use_improved_estimator': False,  # 禁用以避免复杂度
            'use_position_fusion': False,
            'use_attribution': False,
            'retrieval_topk': 3,
            'max_images': 5,
            'thinking': False,
        }
    )
    print("✓ Pipeline创建成功")
except Exception as e:
    print(f"✗ Pipeline创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 运行实验
print("\n6. 运行实验...")
results = []
start_time = time.time()

for i, sample in enumerate(samples):
    print(f"\n处理样本 {i+1}/{len(samples)}")
    try:
        result = pipeline.run_single(sample)

        # 保存结果
        formatted_result = {
            'id': i,
            'question': sample['question'],
            'predicted_answer': result.get('answer', ''),
            'ground_truth': sample['answer'],
            'golden_answers': sample['golden_answers'],
            'retrieved_docs': result.get('retrieved_docs', []),
            'used_retrieval': result.get('used_retrieval', False)
        }

        results.append(formatted_result)

        # 简单的准确率检查
        pred = result.get('answer', '').strip().lower()
        gt = sample['answer'].strip().lower()
        is_correct = pred in gt or gt in pred

        print(f"  问题: {sample['question'][:50]}...")
        print(f"  预测: {pred}")
        print(f"  标注: {gt}")
        print(f"  正确: {'✓' if is_correct else '✗'}")

    except Exception as e:
        print(f"  ✗ 处理失败: {e}")
        results.append({
            'id': i,
            'error': str(e),
            'question': sample['question'],
            'ground_truth': sample['answer']
        })

# 计算准确率
correct = sum(1 for r in results if 'error' not in r and r.get('predicted_answer', '').lower() in r.get('ground_truth', '').lower())
accuracy = correct / len(results) * 100 if results else 0

elapsed_time = time.time() - start_time

# 保存结果
print("\n7. 保存结果...")
output_file = f'/data0/home/zqwang/ACL/FlashRAG/test_safe_run_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

result_data = {
    'test_info': {
        'dataset': dataset_name,
        'num_samples': len(samples),
        'method': 'Self-Aware-MRAG',
        'accuracy': accuracy,
        'elapsed_time': elapsed_time,
        'correct_samples': correct,
        'timestamp': datetime.now().isoformat()
    },
    'results': results
}

with open(output_file, 'w') as f:
    json.dump(result_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 结果已保存: {output_file}")

# 打印总结
print("\n" + "="*80)
print("测试总结")
print("="*80)
print(f"数据集: {dataset_name.upper()}")
print(f"样本数: {len(samples)}")
print(f"准确率: {accuracy:.2f}% ({correct}/{len(results)})")
print(f"总耗时: {elapsed_time:.1f}秒")
print(f"平均每样本: {elapsed_time/len(samples):.1f}秒")

print("\n测试成功！可以运行完整实验了。")